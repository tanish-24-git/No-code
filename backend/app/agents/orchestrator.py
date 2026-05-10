"""OrchestratorAgent: opens the session, broadcasts the master plan, and
handles audit overrides.

It is the only agent that runs synchronously inside the upload request
handler - everything else is event-driven from the bus.

When the LLM probe at session start succeeded ("full_agent" mode), the
orchestrator can ask the configured provider for a custom plan. When the
probe failed ("deterministic" mode), it falls back to the static plan
below so the user still gets a roadmap.
"""
from __future__ import annotations

import json
import re

from app.agents.base import BaseAgent
from app.api.schemas.session import FSMState
from app.events.types import AgentEvent
from app.services import session_service




_PLAN_SYSTEM_PROMPT = (
    "You are the lead orchestrator for FineTune Studio. Generate a concrete "
    "6-8 step execution plan for the user's fine-tuning session covering: "
    "data alchemy, hardware analysis, task inference, model search, strategy, "
    "pipeline construction, training, evaluation, and export. "
    "Return ONLY a JSON array of short strings."
)


class OrchestratorAgent(BaseAgent):
    name = "OrchestratorAgent"
    role = "Voice of the studio. Greets and broadcasts the master plan."
    allowed_tools = ()
    triggers = ("SessionStarted", "AuditOverride", "PhaseApproved")

    async def handle(self, event: AgentEvent) -> None:
        if event.kind == "SessionStarted":
            await self._open(event)
        elif event.kind == "AuditOverride":
            await self._audit_override(event)
        elif event.kind == "PhaseApproved":
            if event.payload.get("phase") == "plan":
                await self.complete(
                    event.session_id,
                    phase="plan",
                    summary="Master plan approved. Starting the cascade.",
                    parent=event.id,
                )

    async def _open(self, event: AgentEvent) -> None:
        session_id = event.session_id
        probe = (event.payload or {}).get("llm_probe") or {}
        mode = probe.get("mode", "deterministic")
        provider = probe.get("provider")
        model = probe.get("model")

        if mode == "full_agent" and provider and model:
            mode_note = (
                f"Connected to {provider} / {model} in {probe.get('latency_ms', 0):.0f} ms - "
                "running in full agent mode."
            )
        else:
            detail = probe.get("detail") or "no LLM provider configured"
            mode_note = (
                f"LLM probe failed ({detail}). I will run in deterministic mode - "
                "the pipeline still works, but I can't paraphrase, search, or reason "
                "with an LLM until a provider is configured in Settings."
            )

        await self.emit_message(
            session_id,
            "Session online. I'll narrate every phase and pause for your "
            "approval on the critical ones.\n\n" + mode_note,
            parent=event.id,
        )


        # In full-agent mode, ask the LLM for a custom plan.
        if mode != "full_agent":
             await self.emit_error(
                 session_id,
                 f"Agent activation failed: {probe.get('detail', 'no LLM provider configured')}. "
                 "Configure a valid OpenAI/Anthropic/Groq key in Settings to use the agentic pipeline."
             )
             return

        session = self.get_session(session_id)
        ds_id = session.dataset_id if session else "this dataset"
        
        await self.think(session_id, "Deliberating on a custom execution plan for your data...", parent=event.id)
        
        try:
            # We use a multi-stage prompt to ensure we get both reasoning and JSON.
            plan_prompt = (
                f"I am initializing a fine-tuning session for dataset {ds_id}.\n\n"
                "Task: Generate a concrete 6-8 step execution plan.\n"
                "Requirements:\n"
                "1. Cover intake, profiling, hardware probe, task inference, model search, strategy, and training.\n"
                "2. If raw docs are present, include a restructuring phase.\n"
                "3. Think like a SOTA AI engineer. Be precise.\n\n"
                "Output format:\n"
                "<reasoning>Your engineering thoughts here</reasoning>\n"
                "```json\n"
                "[\"step 1\", \"step 2\", ...]\n"
                "```"
            )
            
            raw_response = await self.call_llm(
                session_id,
                plan_prompt,
                system="You are the lead orchestrator for FineTune Studio.",
                stream_thoughts=True,
                parent=event.id,
            )
            
            # Extract JSON from code blocks or raw brackets
            m = re.search(r"```json\s*(\[.*\])\s*```", raw_response, re.DOTALL)
            if not m:
                m = re.search(r"(\[.*\])", raw_response, re.DOTALL)
            
            if not m:
                raise ValueError(f"Could not find JSON array in LLM response: {raw_response[:200]}...")
            
            parsed = json.loads(m.group(1))
            steps = [str(s) for s in parsed][:10]
        except Exception as e:
            await self.emit_error(
                session_id,
                f"Master Plan generation failed: {str(e)}. "
                "I'll fall back to a standard roadmap for now."
            )
            steps = [
                "Dataset intake and metadata check",
                "Deep profiling and health scan",
                "Hardware VRAM probe",
                "Task inference and goal setting",
                "Model search and ranking",
                "Training strategy optimization",
                "Pipeline construction and execution"
            ]

        await self.announce(
            session_id,
            phase="plan",
            title="Master plan",
            summary="I've deliberated on your project requirements and drafted this roadmap.",
            steps=steps,
            requires_approval=True,
            parent=event.id,
        )

        comment = await self.wait_for_approval(session_id, "plan")
        while comment:
            await self.think(session_id, f"Refining the roadmap based on your feedback: '{comment}'")
            try:
                refine_prompt = (
                    f"Original roadmap: {steps}\n"
                    f"User feedback: '{comment}'\n\n"
                    "Adjust the roadmap steps. Return ONLY the new JSON array of strings."
                )
                plan_text = await self.call_llm(
                    session_id,
                    refine_prompt,
                    system="You are a flexible lead orchestrator.",
                    stream_thoughts=False,
                    parent=event.id,
                )
                m = re.search(r"(\[.*\])", plan_text, re.DOTALL)
                if m:
                    steps = json.loads(m.group(1))
                    await self.think(session_id, "Roadmap revised. I've incorporated your instructions.")
            except Exception:
                pass

            
            await self.announce(
                session_id,
                phase="plan",
                title="Master plan (Revised)",
                summary="I have updated the plan based on your feedback. Does this look better?",
                steps=steps,
                requires_approval=True,
                parent=event.id,
            )
            comment = await self.wait_for_approval(session_id, "plan")

        await self.complete(
            session_id,
            phase="plan",
            summary="Master plan finalized. Starting the cascade.",
            parent=event.id,
        )


    async def _audit_override(self, event: AgentEvent) -> None:
        """The Critic vetoed something. Surface to the user immediately."""
        p = event.payload or {}
        await self.emit_message(
            event.session_id,
            f"Audit override - {p.get('summary', 'a critical concern was raised')}.\n\n"
            f"Recommendation: {p.get('advice', 'review the agent activity log')}.",
            parent=event.id,
        )


def _master_plan_md(steps: list[str], mode: str) -> str:
    lines = ["## Master plan", "", f"_Mode: {mode}_", "", "**Phases:**"]
    for i, s in enumerate(steps, 1):
        lines.append(f"{i}. {s}")
    return "\n".join(lines)
