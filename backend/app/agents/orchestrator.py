"""OrchestratorAgent — the General (blueprint §2.1).

Responsibilities:
    1. Open every session with a brief, on-brand greeting that prepares
       the user for an interactive run.
    2. Stream a top-level plan ("here is what I will do") so the UI can
       render the bird's-eye view *before* tool execution begins.
    3. Watch for late-arriving free-text from the user and route it to the
       appropriate downstream agent (free-text guidance becomes a clarif-
       ication answer when one is pending).

Per the blueprint: the General is the "Voice of the Studio." Tone here
matters more than logic — every opening message should make the user feel
the agent has a strategy.
"""
from __future__ import annotations

from app.agents.base import BaseAgent
from app.events.types import AgentEvent


_OPENING_PLAN = [
    "Inspect your dataset and grade its health",
    "Probe local hardware (device, VRAM, throughput)",
    "Infer the task and ask if I'm under-confident",
    "Rank base models against the joint context",
    "Pick a SOTA training stack (DoRA / GaLore / Unsloth as warranted)",
    "Draft the pipeline graph node-by-node",
    "Run training with live anomaly + recovery monitoring",
    "Benchmark in an isolated sandbox and finalize export",
]


class OrchestratorAgent(BaseAgent):
    name = "OrchestratorAgent"
    role = "Voice of the studio. Greets the user and broadcasts the master plan."
    allowed_tools = ()
    triggers = ("SessionStarted", "AuditOverride")

    async def handle(self, event: AgentEvent) -> None:
        if event.kind == "SessionStarted":
            await self._open(event)
        elif event.kind == "AuditOverride":
            await self._audit_override(event)

    async def _open(self, event: AgentEvent) -> None:
        session_id = event.session_id
        await self.emit_message(
            session_id,
            "**Session online.** I'm the master orchestrator — I'll narrate "
            "every step, ask before high-impact decisions, and pause for "
            "your input whenever the data forces a judgment call.",
            parent=event.id,
        )
        await self.plan(session_id, _OPENING_PLAN, title="Master plan", parent=event.id)
        await self.think(
            session_id,
            "Booting the swarm: data alchemist, hardware analyst, task "
            "inference, model ranker, architectural designer, and audit critic "
            "are all subscribed and waiting on the dataset.",
            parent=event.id,
        )

    async def _audit_override(self, event: AgentEvent) -> None:
        """The Critic vetoed something. Surface to the user immediately."""
        p = event.payload or {}
        await self.emit_message(
            event.session_id,
            f"**Audit override** — {p.get('summary', 'a critical concern was raised')}.\n\n"
            f"Recommendation: {p.get('advice', 'review the agent activity log')}.",
            parent=event.id,
        )
