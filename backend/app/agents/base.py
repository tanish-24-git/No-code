"""BaseAgent: the contract every specialized agent implements.

An agent's life:
    1. Bus delivers an event of `triggers` kind.
    2. Agent reads session state.
    3. Agent calls tools.
    4. Agent emits new events (which schedule other agents).

Agents never call other agents directly — coupling is via the bus.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from app.events.bus import EventBus
from app.events.types import AgentEvent, EventKind
from app.services import session_service
from app.tools.registry import ToolContext, run_tool


log = logging.getLogger("finetune-studio.agents")


class BaseAgent:
    name: str = "BaseAgent"
    role: str = ""
    allowed_tools: tuple[str, ...] = ()
    triggers: tuple[EventKind, ...] = ()

    def __init__(self, bus: EventBus) -> None:
        self.bus = bus

    async def handle(self, event: AgentEvent) -> None:
        raise NotImplementedError

    # ── Helpers ────────────────────────────────────────────────────────────

    def _ctx(self, session_id: str) -> ToolContext:
        return ToolContext(
            session_id=session_id,
            bus=self.bus,
            extras={"agent": self.name},
        )

    async def call_tool(self, name: str, args: dict[str, Any], session_id: str) -> dict[str, Any]:
        if self.allowed_tools and name not in self.allowed_tools:
            return {"error": f"agent {self.name} not authorized for tool {name}"}
        return await run_tool(name, args, self._ctx(session_id))

    async def emit(
        self,
        kind: EventKind,
        session_id: str,
        *,
        payload: Optional[dict[str, Any]] = None,
        rationale: Optional[str] = None,
        confidence: Optional[float] = None,
        parent_event_id: Optional[str] = None,
        decision_id: Optional[str] = None,
    ) -> AgentEvent:
        ev = AgentEvent(
            session_id=session_id,
            kind=kind,
            actor=self.name,
            payload=payload or {},
            rationale=rationale,
            confidence=confidence,
            parent_event_id=parent_event_id,
            decision_id=decision_id,
        )
        await self.bus.publish(ev)
        return ev

    async def emit_message(self, session_id: str, text: str, *, parent: Optional[str] = None) -> AgentEvent:
        return await self.emit("AssistantMessage", session_id, payload={"text": text}, parent_event_id=parent)

    async def emit_error(self, session_id: str, message: str) -> AgentEvent:
        return await self.emit("Error", session_id, payload={"error": message})

    def get_session(self, session_id: str):
        return session_service.get(session_id)
