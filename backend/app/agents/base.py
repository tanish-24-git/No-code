"""BaseAgent: the contract every specialized agent implements.

An agent's life:
    1. Bus delivers an event of `triggers` kind.
    2. Agent reads session state + the federated blackboard.
    3. Agent calls tools.
    4. Agent emits new events (which schedule other agents).

Agents never call other agents directly - coupling is via the bus.

Two narration patterns coexist on this class:

    * Phase narration (Antigravity-style)   ``announce`` / ``materialize_node``
                                            / ``complete``. The frontend renders
                                            these as chat-bubble plan cards with
                                            inline approve / comment buttons.

    * Streaming helpers                     ``think`` / ``plan`` / ``ask`` /
                                            ``garnish`` / ``stream_executing``.
                                            Mirrored to the federated blackboard
                                            and used by the audit / sandbox /
                                            alchemy agents.

Both patterns are additive - new agents can pick the one that fits their
job.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from app.agents.providers import stream_chat
from app.events.bus import EventBus
from app.events.types import AgentEvent, EventKind, StreamTone
from app.services import session_service
from app.services.blackboard import get_blackboard
from app.services.phase_service import (
    PhasePlan,
    announce_edge,
    announce_node,
    announce_phase,
    complete_phase,
)
from app.tools.registry import ToolContext, run_tool
from app.utils.config import settings


log = logging.getLogger("finetune-studio.agents")


class BaseAgent:
    name: str = "BaseAgent"
    role: str = ""
    allowed_tools: tuple[str, ...] = ()
    triggers: tuple[EventKind, ...] = ()

    def __init__(self, bus: EventBus) -> None:
        self.bus = bus
        # Tracks session IDs with an active handle() invocation.
        # Prevents duplicate concurrent execution when PhaseApproved both
        # resolves a wait_for_approval future AND re-triggers handle().
        self._active_sessions: set[str] = set()

    async def handle(self, event: AgentEvent) -> None:
        raise NotImplementedError

    async def safe_handle(self, event: AgentEvent) -> None:
        """Wrapper that prevents reentrant handle() for the same session."""
        sid = event.session_id
        if sid in self._active_sessions:
            log.debug("%s: skipping reentrant handle for session %s (event=%s)",
                      self.name, sid, event.kind)
            return
        self._active_sessions.add(sid)
        try:
            await self.handle(event)
        finally:
            self._active_sessions.discard(sid)

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
        text = self._clean_text(text)
        return await self.emit("AssistantMessage", session_id, payload={"text": text}, parent_event_id=parent)

    async def stream_message(
        self,
        session_id: str,
        delta: str,
        *,
        is_final: bool = False,
        parent: Optional[str] = None,
    ) -> None:
        """Stream an incremental assistant response."""
        await self.emit(
            "AssistantMessage",
            session_id,
            payload={
                "delta": self._clean_text(delta),
                "is_final": is_final,
            },
            parent_event_id=parent,
        )

    def _clean_text(self, text: str) -> str:
        """Strip markdown markers to keep the UI clean as per user request."""
        return text.replace("**", "").replace("* ", "• ").replace("*", "").strip()

    async def emit_error(self, session_id: str, message: str) -> AgentEvent:
        return await self.emit("Error", session_id, payload={"error": message})

    def get_session(self, session_id: str):
        return session_service.get(session_id)

    # ── Phase narration (Antigravity-style) ────────────────────────────────

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

    async def materialize_edge(
        self,
        session_id: str,
        edge: dict[str, Any],
        *,
        parent: Optional[str] = None,
    ) -> None:
        """Tell the canvas to draw a connection right now."""
        await announce_edge(
            self.bus,
            session_id=session_id,
            actor=self.name,
            edge=edge,
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

    async def wait_for_approval(self, session_id: str, phase: str) -> Optional[str]:
        """Pause execution until the user clicks Approve or submits a Comment.
        Returns the comment text if one was provided, or None if approved.

        Automatically transitions the session to AWAITING_APPROVAL so the
        frontend displays the approve/comment buttons, and restores the
        previous state once the user responds.
        """
        from app.api.schemas.session import FSMState

        # 1. Track this phase as pending so we don't restore state prematurely.
        session = self.get_session(session_id)
        if session:
            pending = set(session.artifacts.get("pending_approvals", []))
            pending.add(phase)
            session_service.attach_artifact(session, "pending_approvals", list(pending))

        # Transition to AWAITING_APPROVAL so the FE renders interaction buttons.
        prev_state = session.state if session else None
        if session and session.state != FSMState.AWAITING_APPROVAL:
            session_service.advance_state(
                session, FSMState.AWAITING_APPROVAL,
                reason=f"awaiting {phase} approval",
            )

        loop = asyncio.get_running_loop()
        future = loop.create_future()

        async def _on_approved(ev: AgentEvent) -> None:
            if ev.session_id == session_id and ev.payload.get("phase") == phase:
                if not future.done():
                    future.set_result(None)

        async def _on_commented(ev: AgentEvent) -> None:
            if ev.session_id == session_id and ev.payload.get("phase") == phase:
                if not future.done():
                    future.set_result(ev.payload.get("text"))

        self.bus.on("PhaseApproved", _on_approved)
        self.bus.on("PhaseCommented", _on_commented)

        try:
            result = await future
            # Mirror the user's feedback into the chat for visibility.
            if result:
                await self.emit_message(
                    session_id, 
                    f"**User Feedback ({phase}):** {result}",
                    parent=None # Root level message
                )
        finally:
            self.bus.off("PhaseApproved", _on_approved)
            self.bus.off("PhaseCommented", _on_commented)

            # 2. Remove this phase from pending.
            session = self.get_session(session_id)
            if session:
                pending = set(session.artifacts.get("pending_approvals", []))
                if phase in pending:
                    pending.remove(phase)
                session_service.attach_artifact(session, "pending_approvals", list(pending))
            else:
                pending = set()

        # 3. Restore the previous state ONLY if no other agent is waiting.
        if session and session.state == FSMState.AWAITING_APPROVAL and not pending:
            restore_to = prev_state if prev_state and prev_state != FSMState.AWAITING_APPROVAL else FSMState.PROFILING
            session_service.advance_state(
                session, restore_to,
                reason=f"{phase} {'feedback received' if result else 'approved'}",
            )

        return result


    # ── Streaming helpers (blackboard-mirrored) ────────────────────────────

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
        include_tools: bool = False,
        parent: Optional[str] = None,
    ) -> str:
        """Execute a reasoning step using the configured LLM provider.
        
        Set include_tools=True only if the agent needs to call search/analysis tools.
        """
        session = self.get_session(session_id)
        if not session:
            return ""

        provider = getattr(session, "llm_provider", None) or settings.llm_provider or ""
        model = getattr(session, "llm_model", None) or settings.llm_model or ""
        base_url = getattr(session, "llm_base_url", None) or settings.llm_base_url or ""
        api_key = getattr(session, "llm_api_key", None) or settings.llm_api_key or ""

        if not provider or not model:
            return ""

        full_text = ""
        messages = [{"role": "user", "content": prompt}]
        
        # If not using tools, explicitly tell the model to avoid tool-call syntax.
        if not include_tools:
            system = (system + "\n\n" if system else "") + "CRITICAL: Do NOT attempt to use any tools or functions. Return only plain text or JSON as requested."

        try:
            for chunk in stream_chat(
                provider=provider,
                api_key=api_key,
                model=model,
                base_url=base_url,
                messages=messages,
                use_tools=include_tools,
                extra_system=system,
            ):
                full_text += chunk
                if stream_thoughts:
                    await self.think_delta(session_id, chunk, parent=parent)
        except Exception as e:
            log.warning("call_llm failed: %s", e)
            return full_text

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
        the canvas with the prescribed ``glow`` class."""
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
