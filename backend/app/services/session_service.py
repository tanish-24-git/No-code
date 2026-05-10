"""AgentSession lifecycle. Persists to data/sessions/<id>.json via the JSON
store. State transitions are validated against ALLOWED_TRANSITIONS so an
errant agent can't push the FSM somewhere illegal."""
from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any, Callable, Optional

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
_session_locks: dict[str, threading.Lock] = {}

def _get_lock(session_id: str) -> threading.Lock:
    if session_id not in _session_locks:
        _session_locks[session_id] = threading.Lock()
    return _session_locks[session_id]



def _now() -> datetime:
    return datetime.now(timezone.utc)


def _persist(s: AgentSession) -> None:
    s.updated_at = _now()
    store.write(_COLLECTION, s.id, s.model_dump(mode="json"))


def _update_atomic(session: AgentSession, updater: Callable[[AgentSession], None]) -> AgentSession:
    with _get_lock(session.id):
        latest = get(session.id)
        if not latest:
            latest = session
        updater(latest)
        _persist(latest)
        session.__dict__.update(latest.__dict__)
        return session


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
                id=s.id, name=s.name, dataset_id=s.dataset_id, state=s.state,
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
    def _updater(s: AgentSession) -> None:
        if target == s.state:
            return
        allowed = ALLOWED_TRANSITIONS.get(s.state, set())
        if target not in allowed:
            raise ValueError(
                f"Illegal transition {s.state.value} -> {target.value} "
                f"(allowed: {sorted(st.value for st in allowed)})"
            )
        log.info("session %s: %s -> %s (%s)", s.id, s.state.value, target.value, reason or "no reason")
        s.state = target
        s.state_entered_at = _now()
    return _update_atomic(session, _updater)


def attach_artifact(session: AgentSession, key: str, value: Any) -> AgentSession:
    def _updater(s: AgentSession) -> None:
        s.artifacts[key] = value
    return _update_atomic(session, _updater)


def attach_pipeline(session: AgentSession, pipeline_id: str) -> AgentSession:
    def _updater(s: AgentSession) -> None:
        s.pipeline_id = pipeline_id
    return _update_atomic(session, _updater)


def set_dataset(session: AgentSession, dataset_id: str) -> AgentSession:
    """Re-bind the session to a new dataset. Used by the restructurer when
    it converts a raw doc into a structured dataset and wants the rest of
    the pipeline to run against the new dataset transparently."""
    def _updater(s: AgentSession) -> None:
        s.dataset_id = dataset_id
    return _update_atomic(session, _updater)


def attach_job(session: AgentSession, job_id: str) -> AgentSession:
    def _updater(s: AgentSession) -> None:
        s.job_id = job_id
    return _update_atomic(session, _updater)


def set_confidence(session: AgentSession, confidence: float) -> AgentSession:
    def _updater(s: AgentSession) -> None:
        s.confidence = max(0.0, min(1.0, confidence))
    return _update_atomic(session, _updater)


def set_pending_question(session: AgentSession, q: Optional[ClarificationQuestion]) -> AgentSession:
    def _updater(s: AgentSession) -> None:
        s.pending_question = q
        if q is not None:
            s.question_budget_used += 1
    return _update_atomic(session, _updater)


def record_answer(session: AgentSession, answer: ClarificationAnswer) -> AgentSession:
    def _updater(s: AgentSession) -> None:
        s.clarifications.append(answer)
        s.pending_question = None
    return _update_atomic(session, _updater)


def fail(session: AgentSession, message: str) -> AgentSession:
    def _updater(s: AgentSession) -> None:
        s.error_message = message
        # Inlined advance_state logic so we only lock once and update atomically
        if FSMState.FAILED != s.state:
            allowed = ALLOWED_TRANSITIONS.get(s.state, set())
            if FSMState.FAILED in allowed:
                log.info("session %s: %s -> %s (%s)", s.id, s.state.value, FSMState.FAILED.value, message)
                s.state = FSMState.FAILED
                s.state_entered_at = _now()
    return _update_atomic(session, _updater)
