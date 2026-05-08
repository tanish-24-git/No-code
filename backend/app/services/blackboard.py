"""Federated Blackboard — shared cognitive workspace per session.

Blueprint §2.1: all agents post their thoughts, plans, and questions to a
shared blackboard. Other agents read it before deciding what to do next.
This is what makes the swarm feel like a hive instead of a pipeline.

Backed by JSON-on-disk so a session survives a process restart and can be
replayed faithfully — see also ``checkpoint_service``.

Concurrency: a per-session ``threading.Lock`` protects multi-writer access.
The asyncio runtime is single-threaded, but background trainers run in
worker threads and may publish via ``publish_threadsafe`` which can race.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app.utils.config import settings


# Per-section limits keep the blackboard from growing unboundedly when an
# agent gets stuck in a thought-cycle. Older entries are dropped FIFO.
_MAX_PER_SECTION = 64


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Blackboard:
    """Append-only cognitive log per session.

    Sections:
        thoughts   — internal reasoning posts (AgentThinking)
        plans      — step-by-step plans (AgentPlanning)
        questions  — open dialogue items (AgentAsking)
        critiques  — AuditAgent vetoes / cautions (AuditCritique)
        decisions  — chosen actions ⇒ feed downstream agents
        nodes      — UI scaffolding the GarnishingAgent has materialized
    """

    SECTIONS = ("thoughts", "plans", "questions", "critiques", "decisions", "nodes")

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = root or (settings.data_dir / "blackboards")
        self.root.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, threading.Lock] = {}
        # In-memory cache to avoid disk on every read.
        self._cache: dict[str, dict[str, list[dict[str, Any]]]] = {}

    # ── lifecycle ────────────────────────────────────────────────────────

    def _path(self, session_id: str) -> Path:
        return self.root / f"{session_id}.json"

    def _lock(self, session_id: str) -> threading.Lock:
        lk = self._locks.get(session_id)
        if lk is None:
            lk = threading.Lock()
            self._locks[session_id] = lk
        return lk

    def _empty(self) -> dict[str, list[dict[str, Any]]]:
        return {s: [] for s in self.SECTIONS}

    def _load(self, session_id: str) -> dict[str, list[dict[str, Any]]]:
        if session_id in self._cache:
            return self._cache[session_id]
        p = self._path(session_id)
        if not p.exists():
            data = self._empty()
        else:
            try:
                with p.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                # Heal missing sections on schema upgrades.
                for s in self.SECTIONS:
                    data.setdefault(s, [])
            except Exception:
                data = self._empty()
        self._cache[session_id] = data
        return data

    def _save(self, session_id: str, data: dict[str, list[dict[str, Any]]]) -> None:
        p = self._path(session_id)
        tmp = p.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, default=str, ensure_ascii=False, indent=2)
        tmp.replace(p)

    # ── post / read ──────────────────────────────────────────────────────

    def post(
        self,
        session_id: str,
        section: str,
        *,
        agent: str,
        content: str,
        tags: Optional[list[str]] = None,
        meta: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        if section not in self.SECTIONS:
            raise ValueError(f"unknown blackboard section: {section}")
        entry = {
            "id": f"{section}_{int(datetime.now(timezone.utc).timestamp() * 1000)}",
            "agent": agent,
            "content": content,
            "tags": tags or [],
            "meta": meta or {},
            "ts": _now(),
        }
        with self._lock(session_id):
            data = self._load(session_id)
            data[section].append(entry)
            # FIFO trim per blueprint §6 circuit-breaker spirit.
            if len(data[section]) > _MAX_PER_SECTION:
                data[section] = data[section][-_MAX_PER_SECTION:]
            self._save(session_id, data)
        return entry

    def read(self, session_id: str, section: Optional[str] = None) -> Any:
        with self._lock(session_id):
            data = self._load(session_id)
            if section is None:
                return {k: list(v) for k, v in data.items()}
            return list(data.get(section, []))

    def latest(self, session_id: str, section: str, n: int = 1) -> list[dict[str, Any]]:
        return self.read(session_id, section)[-n:]

    def clear(self, session_id: str) -> None:
        with self._lock(session_id):
            self._cache.pop(session_id, None)
            p = self._path(session_id)
            if p.exists():
                p.unlink()


_instance: Optional[Blackboard] = None


def get_blackboard() -> Blackboard:
    global _instance
    if _instance is None:
        _instance = Blackboard()
    return _instance
