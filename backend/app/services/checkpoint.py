"""Per-session checkpoint + circuit breaker.

Blueprint §6 (System Stability & Robustness):
    - Every event flushes the relevant slice of session state to
      ``data/sessions/<id>/checkpoint.json`` so a crashed runtime can be
      revived. The event log already replays state — checkpoints are the
      O(1) snapshot that lets us answer "what is the agent thinking *right
      now*?" without scanning the JSONL.

    - Circuit breakers track how many times the same event kind has fired
      against a session. If the count exceeds ``loop_threshold``, the next
      handler invocation is short-circuited and a ``CircuitBreakerTripped``
      event is published. This guards against agent loops where two agents
      keep responding to each other without progress.

The checkpoint file is opened atomically (write to .tmp, rename) so a
half-written file can never be observed by a reader.
"""
from __future__ import annotations

import json
import threading
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app.utils.config import settings


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class CheckpointStore:
    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = root or (settings.data_dir / "checkpoints")
        self.root.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, threading.Lock] = {}

    def _path(self, session_id: str) -> Path:
        return self.root / f"{session_id}.json"

    def _lock(self, session_id: str) -> threading.Lock:
        lk = self._locks.get(session_id)
        if lk is None:
            lk = threading.Lock()
            self._locks[session_id] = lk
        return lk

    def save(self, session_id: str, snapshot: dict[str, Any]) -> Path:
        snapshot = {**snapshot, "saved_at": _now()}
        p = self._path(session_id)
        tmp = p.with_suffix(".tmp")
        with self._lock(session_id):
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(snapshot, f, default=str, ensure_ascii=False, indent=2)
            tmp.replace(p)
        return p

    def load(self, session_id: str) -> Optional[dict[str, Any]]:
        p = self._path(session_id)
        if not p.exists():
            return None
        try:
            with p.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None


class CircuitBreaker:
    """Detects pathological loops (same kind firing N times in M seconds)."""

    def __init__(self, *, loop_threshold: int = 12, window_seconds: int = 30) -> None:
        self.loop_threshold = loop_threshold
        self.window_seconds = window_seconds
        # session_id → kind → deque[ts]
        self._hits: dict[str, dict[str, deque[float]]] = defaultdict(lambda: defaultdict(deque))
        self._tripped: set[tuple[str, str]] = set()
        self._lock = threading.Lock()

    def record(self, session_id: str, kind: str) -> bool:
        """Return True if circuit is currently tripped for this (session, kind)."""
        from time import time

        now = time()
        with self._lock:
            dq = self._hits[session_id][kind]
            dq.append(now)
            cutoff = now - self.window_seconds
            while dq and dq[0] < cutoff:
                dq.popleft()
            if len(dq) > self.loop_threshold:
                self._tripped.add((session_id, kind))
                return True
            return (session_id, kind) in self._tripped

    def reset(self, session_id: str, kind: Optional[str] = None) -> None:
        with self._lock:
            if kind is None:
                self._hits.pop(session_id, None)
                self._tripped = {(s, k) for (s, k) in self._tripped if s != session_id}
            else:
                self._hits[session_id].pop(kind, None)
                self._tripped.discard((session_id, kind))

    def is_tripped(self, session_id: str, kind: str) -> bool:
        return (session_id, kind) in self._tripped


_store: Optional[CheckpointStore] = None
_breaker: Optional[CircuitBreaker] = None


def get_checkpoint_store() -> CheckpointStore:
    global _store
    if _store is None:
        _store = CheckpointStore()
    return _store


def get_circuit_breaker() -> CircuitBreaker:
    global _breaker
    if _breaker is None:
        _breaker = CircuitBreaker()
    return _breaker
