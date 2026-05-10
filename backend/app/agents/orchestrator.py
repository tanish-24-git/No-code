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
from app.services import dataset_service, session_service


# Phase used by THIS agent's approval gate. Different from PipelineBuilder's
# phase="plan" so the two never collide on a single PhaseApproved event.
_PHASE = "master_plan"


class OrchestratorAgent(BaseAgent):
    name = "OrchestratorAgent"
    role = "Voice of the studio. Greets, plans, and records global directives."
    directive_scope = "global"
    allowed_tools = ()
    triggers = ("SessionStarted", "AuditOverride")

    async def handle(self, event: AgentEvent) -> None:
        if event.kind == "SessionStarted":
            await self._open(event)
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
            detail = probe.get("detail") or "no LLM provider configured"
            mode_note = (
                f"Running in deterministic mode ({detail}). "
                "The pipeline still works end-to-end. Configure an LLM "
                "provider in Settings to unlock data restructuring, "
                "agent reasoning, and natural-language directives."
            )

        await self.emit_message(
            session_id,
            "Session online. I will narrate every phase and pause for "
            "your approval on the critical ones.\n\n" + mode_note,
            parent=event.id,
        )

        # Compute the steps. If LLM is on, ask for a tailored plan;
        # otherwise use a plan computed from the dataset kind.
        session = self.get_session(session_id)
        steps = self._compute_default_steps(session)

        if mode == "full_agent":
            tailored = await self.call_llm_typed(
                session_id,
                (
                    f"Default plan steps: {steps}\n\n"
                    "Refine these into a concrete 6-9 step execution plan "
                    "for this user's session. Honor any standing user "
                    "directives. Return ONLY: {\"steps\": [\"step 1\", ...]}"
                ),
                MasterPlanResponse,
                system="You are the lead orchestrator for FineTune Studio.",
                stream_thoughts=False,
                parent=event.id,
            )
            if tailored and tailored.steps:
                steps = [str(s) for s in tailored.steps][:10]

        # Persist the plan as an artifact so every downstream agent can
        # reference it via the shared-context block.
        if session:
            session_service.attach_artifact(session, "master_plan", {"steps": steps})

        await self.announce(
            session_id,
            phase=_PHASE,
            title="Master plan",
            summary="Approve to kick the swarm off, or comment to adjust.",
            steps=steps,
            requires_approval=True,
            parent=event.id,
        )

        comment = await self.wait_for_approval(session_id, _PHASE)
        # Loop on comments, but cap at 3 revisions to avoid infinite loops.
        revisions = 0
        while comment and revisions < 3:
            revisions += 1
            tailored = await self.call_llm_typed(
                session_id,
                (
                    f"Original plan: {steps}\n"
                    f"User comment: {comment}\n\n"
                    "Adjust the plan accordingly. Return ONLY "
                    "{\"steps\": [\"...\"]}"
                ),
                MasterPlanResponse,
                system="You are a flexible lead orchestrator.",
                stream_thoughts=False,
                parent=event.id,
            )
            if tailored and tailored.steps:
                steps = [str(s) for s in tailored.steps][:10]
                if session:
                    session_service.attach_artifact(session, "master_plan", {"steps": steps})

            await self.announce(
                session_id,
                phase=_PHASE,
                title="Master plan (revised)",
                summary="Updated per your feedback. Approve or keep commenting.",
                steps=steps,
                requires_approval=True,
                parent=event.id,
            )
            comment = await self.wait_for_approval(session_id, _PHASE)

        await self.complete(
            session_id,
            phase=_PHASE,
            summary="Master plan approved. Starting the cascade.",
            parent=event.id,
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
