"""OrchestratorAgent: opens the session, broadcasts the master plan, and
posts the first user-visible message.

It is the only agent that runs synchronously inside the upload request
handler - everything else is event-driven from the bus.
"""
from __future__ import annotations

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


class OrchestratorAgent(BaseAgent):
    name = "OrchestratorAgent"
    role = "Voice of the studio. Greets and broadcasts the master plan."
    allowed_tools = ()
    triggers = ("SessionStarted",)

    async def handle(self, event: AgentEvent) -> None:
        session_id = event.session_id
        probe = (event.payload or {}).get("llm_probe") or {}
        mode = probe.get("mode", "deterministic")
        provider = probe.get("provider")
        model = probe.get("model")

        if mode == "full_agent" and provider and model:
            mode_note = (
                f"Connected to **{provider} / {model}** in {probe.get('latency_ms', 0):.0f} ms - "
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
        await self.emit(
            "PhasePlanProposed",
            session_id,
            payload={
                "phase": "intake",
                "title": "Master plan",
                "summary": "Here is everything I plan to do for this session.",
                "plan_markdown": _master_plan_md(_OPENING_PLAN, mode),
                "steps": _OPENING_PLAN,
                "requires_approval": False,
            },
            parent_event_id=event.id,
        )


def _master_plan_md(steps: list[str], mode: str) -> str:
    lines = ["## Master plan", "", f"_Mode: {mode}_", "", "**Phases:**"]
    for i, s in enumerate(steps, 1):
        lines.append(f"{i}. {s}")
    return "\n".join(lines)
