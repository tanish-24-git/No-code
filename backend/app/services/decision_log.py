"""Structured audit of every non-trivial agent decision. The event log says
WHAT happened; the decision log says WHY. Each AgentDecisionRecorded event
links to exactly one Decision via decision_id."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field

from app.storage import store


_COLLECTION = "decisions"


class Decision(BaseModel):
    id: str
    session_id: str
    agent: str
    kind: str                              # "task_inference", "model_choice", "recovery_diff", ...
    inputs: dict[str, Any] = Field(default_factory=dict)
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    chosen: Any = None
    confidence: float = 0.0
    rationale: str = ""
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    created_at: datetime


def record(
    *,
    session_id: str,
    agent: str,
    kind: str,
    inputs: Optional[dict[str, Any]] = None,
    candidates: Optional[list[dict[str, Any]]] = None,
    chosen: Any = None,
    confidence: float = 0.0,
    rationale: str = "",
    tool_calls: Optional[list[dict[str, Any]]] = None,
) -> Decision:
    d = Decision(
        id=store.new_id(),
        session_id=session_id,
        agent=agent,
        kind=kind,
        inputs=inputs or {},
        candidates=candidates or [],
        chosen=chosen,
        confidence=confidence,
        rationale=rationale,
        tool_calls=tool_calls or [],
        created_at=datetime.now(timezone.utc),
    )
    store.write(_COLLECTION, d.id, d.model_dump(mode="json"))
    return d


def get(decision_id: str) -> Optional[Decision]:
    raw = store.read(_COLLECTION, decision_id)
    if not raw:
        return None
    return Decision(**raw)


def list_for_session(session_id: str) -> list[Decision]:
    out: list[Decision] = []
    for raw in store.list_all(_COLLECTION):
        try:
            d = Decision(**raw)
        except Exception:
            continue
        if d.session_id == session_id:
            out.append(d)
    out.sort(key=lambda d: d.created_at)
    return out
