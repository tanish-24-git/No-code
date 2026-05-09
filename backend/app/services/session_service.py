"""AgentSession lifecycle. Persists to data/sessions/<id>.json via the JSON
store. State transitions are validated against ALLOWED_TRANSITIONS so an
errant agent can't push the FSM somewhere illegal."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from app.api.schemas.session import (
    ALLOWED_TRANSITIONS,
    AgentSession,
    ClarificationAnswer,
    ClarificationQuestion,
    FSMState,
    SessionListItem,
)
from app.services import dataset_service
from app.storage import store


log = logging.getLogger("finetune-studio.sessions")
_COLLECTION = "sessions"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _persist(s: AgentSession) -> None:
    s.updated_at = _now()
    store.write(_COLLECTION, s.id, s.model_dump(mode="json"))


# ── Read-side ──────────────────────────────────────────────────────────────

def get(session_id: str) -> Optional[AgentSession]:
    raw = store.read(_COLLECTION, session_id)
    if not raw:
        return None
    try:
        return AgentSession(**raw)
    except Exception:
        log.exception("session %s is corrupt", session_id)
        return None


def list_all() -> list[SessionListItem]:
    out: list[SessionListItem] = []
    for raw in store.list_all(_COLLECTION):
        try:
            s = AgentSession(**raw)
            out.append(SessionListItem(
                id=s.id, dataset_id=s.dataset_id, state=s.state,
                pipeline_id=s.pipeline_id, job_id=s.job_id,
                confidence=s.confidence,
                created_at=s.created_at, updated_at=s.updated_at,
            ))
        except Exception:
            continue
    out.sort(key=lambda x: x.updated_at, reverse=True)
    return out


def get_by_dataset(dataset_id: str) -> Optional[AgentSession]:
    """Return the most recent session for a dataset, if any."""
    latest: Optional[AgentSession] = None
    for raw in store.list_all(_COLLECTION):
        try:
            s = AgentSession(**raw)
        except Exception:
            continue
        if s.dataset_id != dataset_id:
            continue
        if latest is None or s.created_at > latest.created_at:
            latest = s
    return latest


# ── Write-side ─────────────────────────────────────────────────────────────

def start_for_dataset(dataset_id: str) -> AgentSession:
    sid = store.new_id()
    now = _now()
    
    # Try to get a meaningful name from the dataset
    name = "New Session"
    ds = dataset_service.get_dataset(dataset_id)
    if ds:
        name = f"Session: {ds.name}"

    s = AgentSession(
        id=sid, name=name, dataset_id=dataset_id,
        state=FSMState.INIT, state_entered_at=now,
        created_at=now, updated_at=now,
    )
    _persist(s)
    return s


def advance_state(session: AgentSession, target: FSMState, *, reason: str = "") -> AgentSession:
    """Validated transition. Raises ValueError if not allowed."""
    if target == session.state:
        return session
    allowed = ALLOWED_TRANSITIONS.get(session.state, set())
    if target not in allowed:
        raise ValueError(
            f"Illegal transition {session.state.value} -> {target.value} "
            f"(allowed: {sorted(s.value for s in allowed)})"
        )
    log.info("session %s: %s -> %s (%s)", session.id, session.state.value, target.value, reason or "no reason")
    session.state = target
    session.state_entered_at = _now()
    _persist(session)
    return session


def attach_artifact(session: AgentSession, key: str, value: Any) -> AgentSession:
    session.artifacts[key] = value
    _persist(session)
    return session


def attach_pipeline(session: AgentSession, pipeline_id: str) -> AgentSession:
    session.pipeline_id = pipeline_id
    _persist(session)
    return session


def attach_job(session: AgentSession, job_id: str) -> AgentSession:
    session.job_id = job_id
    _persist(session)
    return session


def set_confidence(session: AgentSession, confidence: float) -> AgentSession:
    session.confidence = max(0.0, min(1.0, confidence))
    _persist(session)
    return session


def set_pending_question(session: AgentSession, q: Optional[ClarificationQuestion]) -> AgentSession:
    session.pending_question = q
    if q is not None:
        session.question_budget_used += 1
    _persist(session)
    return session


def record_answer(session: AgentSession, answer: ClarificationAnswer) -> AgentSession:
    session.clarifications.append(answer)
    session.pending_question = None
    _persist(session)
    return session


def fail(session: AgentSession, message: str) -> AgentSession:
    session.error_message = message
    advance_state(session, FSMState.FAILED, reason=message)
    return session
