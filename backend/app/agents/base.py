"""BaseAgent: the contract every specialized agent implements.

An agent's life:
    1. Bus delivers an event of `triggers` kind.
    2. Agent reads session state + the federated blackboard.
    3. Agent calls tools.
    4. Agent emits new events (which schedule other agents).

Agents never call other agents directly — coupling is via the bus.

Blueprint §3.1 streaming kinds (thinking / planning / asking / garnishing /
executing) are first-class helpers on this class so every agent emits a
consistent visual narrative without copying boilerplate.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from app.events.bus import EventBus
from app.events.types import AgentEvent, EventKind, StreamTone
from app.services import session_service
from app.services.blackboard import get_blackboard
from app.tools.registry import ToolContext, run_tool
from app.agents.providers import stream_chat
from app.agents.registry import get_spec
from app.utils.config import settings


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

    # ── Socratic / streaming helpers (blueprint §3.1) ──────────────────────

    async def think(
        self,
        session_id: str,
        thought: str,
        *,
        parent: Optional[str] = None,
        meta: Optional[dict[str, Any]] = None,
    ) -> AgentEvent:
        """Stream an internal-reasoning trace. Mirrored to the blackboard."""
        get_blackboard().post(session_id, "thoughts", agent=self.name, content=thought, meta=meta)
        return await self.emit(
            "AgentThinking",
            session_id,
            payload={"stream": "thinking", "agent": self.name, "text": thought, "meta": meta or {}},
            parent_event_id=parent,
        )

    async def think_delta(
        self,
        session_id: str,
        delta: str,
        *,
        is_final: bool = False,
        parent: Optional[str] = None,
    ) -> None:
        """Stream an incremental thought fragment for autoregressive UI feel."""
        await self.emit(
            "AgentThinking",
            session_id,
            payload={
                "stream": "thinking",
                "agent": self.name,
                "delta": delta,
                "is_final": is_final,
            },
            parent_event_id=parent,
        )

    async def call_llm(
        self,
        session_id: str,
        prompt: str,
        *,
        system: str = "",
        stream_thoughts: bool = True,
        parent: Optional[str] = None,
    ) -> str:
        """Execute a reasoning step using the session's configured LLM."""
        session = self.get_session(session_id)
        if not session:
            return ""

        full_text = ""
        # messages shape for the provider
        messages = [{"role": "user", "content": prompt}]

        # We pull the provider settings from the session record.
        # Defaults to gemini-2.0-flash if not set (or env fallback).
        provider = session.llm_provider or settings.llm_provider or "gemini"
        model = session.llm_model or settings.llm_model or "gemini-2.0-flash"
        base_url = session.llm_base_url or settings.llm_base_url or ""
        api_key = session.llm_api_key or settings.llm_api_key or ""

        # Run the stream
        for chunk in stream_chat(
            provider=provider,
            api_key=api_key,
            model=model,
            base_url=base_url,
            messages=messages,
            extra_system=system,
        ):
            full_text += chunk
            if stream_thoughts:
                await self.think_delta(session_id, chunk, parent=parent)

        if stream_thoughts:
            await self.think_delta(session_id, "", is_final=True, parent=parent)

        return full_text

    async def plan(
        self,
        session_id: str,
        steps: list[str],
        *,
        title: Optional[str] = None,
        parent: Optional[str] = None,
    ) -> AgentEvent:
        get_blackboard().post(
            session_id,
            "plans",
            agent=self.name,
            content=title or "plan",
            meta={"steps": steps},
        )
        return await self.emit(
            "AgentPlanning",
            session_id,
            payload={"stream": "planning", "agent": self.name, "title": title or "plan", "steps": steps},
            parent_event_id=parent,
        )

    async def ask(
        self,
        session_id: str,
        question: str,
        *,
        parent: Optional[str] = None,
        impact: str = "medium",
    ) -> AgentEvent:
        """Surface a Socratic question. Pairs with UserClarificationRequested
        when a structured catalog question is involved."""
        get_blackboard().post(
            session_id,
            "questions",
            agent=self.name,
            content=question,
            meta={"impact": impact},
        )
        return await self.emit(
            "AgentAsking",
            session_id,
            payload={"stream": "asking", "agent": self.name, "question": question, "impact": impact},
            parent_event_id=parent,
        )

    async def garnish(
        self,
        session_id: str,
        node: dict[str, Any],
        *,
        parent: Optional[str] = None,
    ) -> AgentEvent:
        """Stream a UI-side scaffold step. The frontend pops the node into
        the canvas with the prescribed ``glow`` class — see blueprint §3.2."""
        get_blackboard().post(session_id, "nodes", agent=self.name, content=node.get("type", "node"), meta=node)
        return await self.emit(
            "AgentGarnishing",
            session_id,
            payload={"stream": "garnishing", "agent": self.name, "node": node},
            parent_event_id=parent,
        )

    async def stream_executing(
        self,
        session_id: str,
        message: str,
        *,
        parent: Optional[str] = None,
    ) -> AgentEvent:
        return await self.emit(
            "AgentExecuting",
            session_id,
            payload={"stream": "executing", "agent": self.name, "text": message},
            parent_event_id=parent,
        )

    # Convenience: the blackboard surface, exposed read-only.

    def blackboard(self):
        return get_blackboard()

    def stream_kind_for(self, tone: StreamTone) -> EventKind:
        """Route a stream tone string to its concrete event kind."""
        return {
            "thinking": "AgentThinking",
            "planning": "AgentPlanning",
            "asking": "AgentAsking",
            "garnishing": "AgentGarnishing",
            "executing": "AgentExecuting",
        }[tone]
