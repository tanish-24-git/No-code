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
from app.events.types import AgentEvent


_OPENING_PLAN = [
    "Read dataset metadata and infer schema",
    "Profile token lengths, duplicates, missing values",
    "Probe local hardware (device + VRAM + throughput)",
    "Search HuggingFace for a base model that fits",
    "Pick a training strategy (LoRA / QLoRA / DoRA)",
    "Draft the pipeline graph and ask you to approve",
    "Train, monitor, recover, evaluate, sandbox",
    "Save locally or push to HF on your command",
]

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


        # In full-agent mode, ask the LLM for a custom plan; fall back to the
        # static plan if anything goes wrong.
        steps = _OPENING_PLAN
        if mode == "full_agent":
            session = self.get_session(session_id)
            ds_id = session.dataset_id if session else "this dataset"
            try:
                plan_text = await self.call_llm(
                    session_id,
                    f"Plan the fine-tuning session for dataset {ds_id}. Provide 6-8 clear steps.",
                    system=_PLAN_SYSTEM_PROMPT,
                    stream_thoughts=False,
                    parent=event.id,
                )
                m = re.search(r"\[.*\]", plan_text, re.DOTALL)
                if m:
                    parsed = json.loads(m.group(0))
                    if isinstance(parsed, list) and parsed:
                        steps = [str(s) for s in parsed][:10]
            except Exception:
                steps = _OPENING_PLAN

        await self.announce(
            session_id,
            phase="plan",
            title="Master plan",
            summary="I have drafted a custom execution plan for this dataset. Please review and approve.",
            steps=steps,
            requires_approval=True,
            parent=event.id,
        )

        # Wait for user to approve the master plan.
        comment = await self.wait_for_approval(session_id, "plan")
        if comment:
            await self.think(session_id, f"User added a note to the master plan: {comment}")
            # In a more advanced version, we would re-generate the plan here.

        await self.complete(
            session_id,
            phase="plan",
            summary="Master plan approved. Starting the cascade.",
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
