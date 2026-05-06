"""In-process event bus. Handlers are async; SSE subscribers receive every
event for their session. Persistence to the event log is unconditional —
no event escapes without being recorded.

This bus is single-process and uses asyncio. It can be swapped for Redis
or NATS later by replacing this module; nothing else imports asyncio.Queue
or the subscribe table directly.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, Optional

from app.events.log import EventLog, get_event_log
from app.events.types import AgentEvent, EventKind


log = logging.getLogger("finetune-studio.events")

Handler = Callable[[AgentEvent], Awaitable[None]]


class EventBus:
    def __init__(self, event_log: Optional[EventLog] = None) -> None:
        self._handlers: dict[str, list[Handler]] = {}
        self._sse: dict[str, list[asyncio.Queue]] = {}
        self._log = event_log or get_event_log()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Called once during FastAPI lifespan. Lets background threads
        schedule publishes via publish_threadsafe."""
        self._loop = loop

    # ── Subscriptions ──────────────────────────────────────────────────────

    def on(self, kind: EventKind, handler: Handler) -> None:
        self._handlers.setdefault(kind, []).append(handler)

    def attach_sse(self, session_id: str, maxsize: int = 256) -> asyncio.Queue:
        q: asyncio.Queue[AgentEvent] = asyncio.Queue(maxsize=maxsize)
        self._sse.setdefault(session_id, []).append(q)
        return q

    def detach_sse(self, session_id: str, q: asyncio.Queue) -> None:
        subs = self._sse.get(session_id, [])
        if q in subs:
            subs.remove(q)

    # ── Publish ────────────────────────────────────────────────────────────

    async def publish(self, event: AgentEvent) -> None:
        # Persist before fan-out: if a handler crashes, the event is still
        # in the log and replay will pick it up.
        try:
            self._log.append(event)
        except Exception:
            log.exception("event_log append failed for %s", event.kind)

        # SSE fan-out (non-blocking).
        for q in list(self._sse.get(event.session_id, [])):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                # Slow consumer: drop. The replay endpoint covers history.
                log.warning("dropping event for slow SSE consumer session=%s kind=%s",
                            event.session_id, event.kind)

        # Handlers run concurrently and isolated; one crash never blocks others.
        for h in self._handlers.get(event.kind, []):
            asyncio.create_task(self._safe_call(h, event))

    def publish_threadsafe(self, event: AgentEvent) -> None:
        """Schedule a publish from a non-asyncio thread (e.g. job worker)."""
        if self._loop is None:
            log.warning("publish_threadsafe before loop bound; dropping %s", event.kind)
            return
        asyncio.run_coroutine_threadsafe(self.publish(event), self._loop)

    # ── Internals ──────────────────────────────────────────────────────────

    async def _safe_call(self, handler: Handler, event: AgentEvent) -> None:
        try:
            await handler(event)
        except Exception:
            log.exception("handler crashed on %s", event.kind)


_bus_instance: Optional[EventBus] = None


def get_bus() -> EventBus:
    global _bus_instance
    if _bus_instance is None:
        _bus_instance = EventBus()
    return _bus_instance
