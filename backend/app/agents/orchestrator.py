"""OrchestratorAgent: opens the session, broadcasts the master plan,
records user comments globally, and handles audit overrides.

When the LLM is configured, the orchestrator asks it for a custom plan
tailored to the dataset's kind. When it isn't, the plan is computed
deterministically from the dataset metadata - so the system is honest
about the mode it's running in without hard-failing.

Crucially, every comment on the master plan is recorded as a *global*
directive (via wait_for_approval -> directives_service.record), so all
downstream agents read it before deciding anything.
"""
from __future__ import annotations

from app.agents.base import BaseAgent
from app.agents.schemas import MasterPlanResponse
from app.events.types import AgentEvent
from app.services import dataset_service, session_service, directives as directives_service


# Phase used by THIS agent's approval gate. Different from PipelineBuilder's
# phase="plan" so the two never collide on a single PhaseApproved event.
_PHASE = "master_plan"


class OrchestratorAgent(BaseAgent):
    name = "OrchestratorAgent"
    role = "Voice of the studio. Greets, plans, and records global directives."
    directive_scope = "global"
    allowed_tools = ()
    triggers = ("SessionStarted", "IntakeCompleted", "UserMessage", "AuditOverride")

    async def handle(self, event: AgentEvent) -> None:
        if event.kind == "SessionStarted":
            await self._open(event)
        elif event.kind == "IntakeCompleted":
            # Set a flag to wait for the user to reply with their goal.
            session = self.get_session(event.session_id)
            if session:
                session_service.attach_artifact(session, "awaiting_user_goal", True)
        elif event.kind == "UserMessage":
            session = self.get_session(event.session_id)
            if session and session.artifacts.get("awaiting_user_goal"):
                # Goal received! record it and draft the plan.
                goal_text = (event.payload or {}).get("text", "")
                directives_service.record(event.session_id, goal_text, scope="global")
                session_service.attach_artifact(session, "awaiting_user_goal", False)
                
                # Signal that we have a goal so downstream agents can wake up.
                await self.emit("GoalCaptured", event.session_id, payload={"goal": goal_text}, parent_event_id=event.id)
                
                await self._broadcast_plan(event.session_id, event.id)
        elif event.kind == "AuditOverride":
            await self._audit_override(event)

    async def _open(self, event: AgentEvent) -> None:
        session_id = event.session_id
        probe = (event.payload or {}).get("llm_probe") or {}
        mode = probe.get("mode", "deterministic")
        provider = probe.get("provider")
        model = probe.get("model")

        if mode == "full_agent" and provider and model:
            mode_note = (
                f"Connected to {provider} / {model} in {probe.get('latency_ms', 0):.0f}ms - "
                "running in full agent mode."
            )
        else:
            mode_note = (
                "Running in fallback mode. To unlock the full conversational "
                "AI harness and dynamic model selection, configure an LLM "
                "provider in Settings."
            )

        await self.emit_message(
            session_id,
            "FineTune Studio online. I am your lead architect. "
            "Upload a dataset to begin, and I will help you design a "
            "bespoke training pipeline.\n\n" + mode_note,
            parent=event.id,
        )

    async def _broadcast_plan(self, session_id: str, parent_id: str) -> None:
        """Called once the user's goal is captured to draft the execution path."""
        session = self.get_session(session_id)
        probe = (session.artifacts.get("llm_probe") or {}) if session else {}
        mode = probe.get("mode", "deterministic")
        
        # Pull any goal set by the user in directives
        all_directives = directives_service.read_for_scope(session_id, "global")
        goal = next((d.text for d in all_directives if d.scope == "global"), "Fine-tune a model") if session else "Fine-tune a model"

        steps = self._compute_default_steps(session)

        if mode == "full_agent":
            tailored = await self.call_llm_typed(
                session_id,
                (
                    f"User Goal: {goal}\n"
                    f"Default plan steps: {steps}\n\n"
                    "Refine these into a concrete 6-9 step execution plan. "
                    "Everything must be dynamic and agent-driven. "
                    "Return ONLY: {\"steps\": [\"step 1\", ...]}"
                ),
                MasterPlanResponse,
                system="You are the lead orchestrator for an autonomous AI training harness.",
                stream_thoughts=False,
                parent=parent_id,
            )
            if tailored and tailored.steps:
                steps = [str(s) for s in tailored.steps][:10]

        if session:
            session_service.attach_artifact(session, "master_plan", {"steps": steps})

        await self.announce(
            session_id,
            phase=_PHASE,
            title="Master plan",
            summary=f"I've drafted a plan to achieve: **{goal}**. Approve to proceed.",
            steps=steps,
            requires_approval=True,
            parent=parent_id,
        )

        comment = await self.wait_for_approval(session_id, _PHASE)
        # ... logic for revisions could go here if needed ...

        await self.complete(
            session_id,
            phase=_PHASE,
            summary="Plan confirmed. Initiating the agentic cascade.",
            parent=parent_id,
        )

    def _compute_default_steps(self, session) -> list[str]:
        """Compute the plan from the dataset kind. Raw docs add a
        restructure step; structured datasets skip it."""
        steps = ["Read dataset metadata"]
        ds = dataset_service.get_dataset(session.dataset_id) if session else None
        if dataset_service.is_raw_doc(ds):
            steps.append("Restructure raw text into training pairs")
        steps.extend([
            "Profile token lengths, duplicates, missing values",
            "Probe local hardware",
            "Search HuggingFace for a base model that fits",
            "Pick a training strategy (LoRA / QLoRA / DoRA)",
            "Draft and approve the pipeline graph",
            "Train, monitor, recover, evaluate",
            "Save locally or push to HF on your command",
        ])
        return steps

    async def _audit_override(self, event: AgentEvent) -> None:
        p = event.payload or {}
        await self.emit_message(
            event.session_id,
            f"Audit override - {p.get('summary', 'a critical concern was raised')}.\n\n"
            f"Recommendation: {p.get('advice', 'review the agent activity log')}.",
            parent=event.id,
        )
