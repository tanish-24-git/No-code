"""BaseAgent: the contract every specialized agent implements.

An agent's life:
    1. Bus delivers an event of `triggers` kind.
    2. Agent reads session state.
    3. Agent calls tools.
    4. Agent emits new events (which schedule other agents).

Agents never call other agents directly - coupling is via the bus.

Phase narration (Antigravity-style): each agent calls ``announce_phase``
before doing work and ``complete_phase`` after. The frontend uses these
events to drive its node materialization and chat bubbles.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from app.events.bus import EventBus
from app.events.types import AgentEvent, EventKind
from app.services import session_service
from app.services.phase_service import (
    PhasePlan,
    announce_node,
    announce_phase,
    complete_phase,
)
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

    # ── Phase narration ────────────────────────────────────────────────────

    async def announce(
        self,
        session_id: str,
        *,
        phase: str,
        title: str,
        summary: str,
        steps: list[str] | None = None,
        inputs: dict[str, Any] | None = None,
        outputs: list[str] | None = None,
        requires_approval: bool = False,
        parent: Optional[str] = None,
    ) -> None:
        """Surface a phase plan to the user (chat bubble + node hint)."""
        plan = PhasePlan(
            phase=phase,
            title=title,
            summary=summary,
            steps=steps or [],
            inputs=inputs or {},
            outputs=outputs or [],
            requires_approval=requires_approval,
        )
        await announce_phase(
            self.bus,
            session_id=session_id,
            actor=self.name,
            phase=phase,
            plan=plan,
            parent_event_id=parent,
        )

    async def materialize_node(
        self,
        session_id: str,
        node: dict[str, Any],
        *,
        parent: Optional[str] = None,
    ) -> None:
        """Tell the canvas to pop a node into existence right now."""
        await announce_node(
            self.bus,
            session_id=session_id,
            actor=self.name,
            node=node,
            parent_event_id=parent,
        )

    async def complete(
        self,
        session_id: str,
        *,
        phase: str,
        summary: str,
        artifacts: Optional[dict[str, Any]] = None,
        parent: Optional[str] = None,
    ) -> None:
        await complete_phase(
            self.bus,
            session_id=session_id,
            actor=self.name,
            phase=phase,
            summary=summary,
            artifacts=artifacts,
            parent_event_id=parent,
        )
