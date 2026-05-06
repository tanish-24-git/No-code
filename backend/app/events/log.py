"""Append-only JSONL log per session. The agent runtime is sourced from this
log, so reloading the session reconstructs the full state by replaying it.

Files live at data/events/<session_id>.jsonl. One line per AgentEvent.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Iterator, Optional

from app.events.types import AgentEvent
from app.utils.config import settings


class EventLog:
    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = root or (settings.data_dir / "events")
        self.root.mkdir(parents=True, exist_ok=True)
        # One lock per session id; protects against torn writes from threads.
        self._locks: dict[str, threading.Lock] = {}

    def _path(self, session_id: str) -> Path:
        return self.root / f"{session_id}.jsonl"

    def _lock(self, session_id: str) -> threading.Lock:
        lock = self._locks.get(session_id)
        if lock is None:
            lock = threading.Lock()
            self._locks[session_id] = lock
        return lock

    def append(self, event: AgentEvent) -> None:
        line = json.dumps(event.model_dump(mode="json"), default=str)
        with self._lock(event.session_id):
            with self._path(event.session_id).open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def read(self, session_id: str) -> Iterator[AgentEvent]:
        p = self._path(session_id)
        if not p.exists():
            return
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield AgentEvent(**json.loads(line))
                except Exception:
                    # A corrupt line should not take down the replay.
                    continue


_log_instance: Optional[EventLog] = None


def get_event_log() -> EventLog:
    global _log_instance
    if _log_instance is None:
        _log_instance = EventLog()
    return _log_instance
