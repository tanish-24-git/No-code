"""AgentSession API.

Endpoints:
    GET   /api/sessions                       list
    GET   /api/sessions/{id}                  detail (includes artifacts)
    GET   /api/sessions/{id}/events           SSE stream (replay + live)
    POST  /api/sessions/{id}/messages         send a free-text user message
    POST  /api/sessions/{id}/clarifications/{qid}  reply to a clarifying question
    POST  /api/sessions/{id}/approve          approve / reject pipeline draft
    POST  /api/sessions/{id}/export           submit export choice
    POST  /api/sessions/{id}/retry            approve / deny a recovery diff
    POST  /api/sessions/{id}/cancel           hard cancel
    GET   /api/sessions/{id}/audit            decision log
    GET   /api/tools                          tool registry introspection
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.api.schemas.session import (
    AgentSession,
    ApprovalDecision,
    ClarificationAnswer,
    ClarificationReply,
    ExportChoice,
    FSMState,
    RetryDecision,
    SessionListItem,
    SessionMessageRequest,
)
from app.events.bus import get_bus
from app.events.log import get_event_log
from app.events.types import AgentEvent
from app.services import decision_log, session_service
from app.tools.registry import list_descriptors
from app.utils.config import settings


router = APIRouter(tags=["sessions"])


def _missing_session(session_id: str) -> HTTPException:
    """Return 410 Gone when an events log exists but the session JSON is
    missing (the session was wiped after the fact — the frontend should
    drop its cached state). Otherwise return 404 Not Found.

    Lets the UI distinguish "you typed a bad URL" from "your session was
    cleared on the server" without polling, so a stale browser tab can
    reset cleanly instead of looping on retries.
    """
    try:
        events_path = settings.data_dir / "events" / f"{session_id}.jsonl"
        if events_path.exists():
            return HTTPException(410, "Session wiped")
    except Exception:
        pass
    return HTTPException(404, "Session not found")


@router.get("/api/sessions", response_model=list[SessionListItem])
def list_sessions() -> list[SessionListItem]:
    return session_service.list_all()


@router.get("/api/sessions/{session_id}", response_model=AgentSession)
def get_session(session_id: str) -> AgentSession:
    s = session_service.get(session_id)
    if not s:
        raise _missing_session(session_id)
    return s


@router.get("/api/sessions/{session_id}/events")
async def stream_session_events(session_id: str) -> StreamingResponse:
    s = session_service.get(session_id)
    if not s:
        raise _missing_session(session_id)

    bus = get_bus()
    log = get_event_log()

    async def gen() -> AsyncIterator[str]:
        # Replay backlog first.
        for ev in log.read(session_id):
            yield ev.as_sse()

        # Tail.
        if s.state in (FSMState.DONE, FSMState.FAILED, FSMState.CANCELLED):
            yield "event: SessionClosed\ndata: {}\n\n"
            return

        q = bus.attach_sse(session_id)
        try:
            while True:
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=15.0)
                    yield ev.as_sse()
                    if ev.kind == "SessionClosed":
                        return
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            bus.detach_sse(session_id, q)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable buffering in some proxies
        }
    )


@router.post("/api/sessions/{session_id}/messages")
async def send_message(session_id: str, payload: SessionMessageRequest) -> dict:
    s = session_service.get(session_id)
    if not s:
        raise _missing_session(session_id)
    bus = get_bus()
    await bus.publish(AgentEvent(
        session_id=session_id,
        kind="UserMessage",
        actor="user",
        payload={"text": payload.text},
    ))
    return {"ok": True}


@router.post("/api/sessions/{session_id}/clarifications/{question_id}")
async def reply_clarification(session_id: str, question_id: str, payload: ClarificationReply) -> dict:
    s = session_service.get(session_id)
    if not s:
        raise _missing_session(session_id)
    if not s.pending_question or s.pending_question.question_id != question_id:
        raise HTTPException(409, "No pending question with that id")

    answer = ClarificationAnswer(
        question_id=question_id,
        value=payload.value,
        answered_at=datetime.now(timezone.utc),
    )
    session_service.record_answer(s, answer)
    bus = get_bus()
    await bus.publish(AgentEvent(
        session_id=session_id,
        kind="UserClarificationReceived",
        actor="user",
        payload={"answer": answer.model_dump(mode="json")},
    ))
    return {"ok": True}


@router.post("/api/sessions/{session_id}/approve")
async def approve_pipeline(session_id: str, payload: ApprovalDecision) -> dict:
    s = session_service.get(session_id)
    if not s:
        raise _missing_session(session_id)
    if s.state != FSMState.AWAITING_APPROVAL:
        raise HTTPException(409, f"Session not awaiting approval (state={s.state.value})")

    bus = get_bus()
    if payload.approve:
        await bus.publish(AgentEvent(
            session_id=session_id, kind="PipelineApproved", actor="user",
            payload={"pipeline_id": s.pipeline_id, "auto": False},
            rationale=payload.reason or "user approved",
        ))
    else:
        await bus.publish(AgentEvent(
            session_id=session_id, kind="PipelineRejected", actor="user",
            payload={"pipeline_id": s.pipeline_id, "reason": payload.reason},
        ))
    return {"ok": True}


@router.post("/api/sessions/{session_id}/export")
async def submit_export(session_id: str, payload: ExportChoice) -> dict:
    s = session_service.get(session_id)
    if not s:
        raise _missing_session(session_id)
    if s.state != FSMState.AWAITING_EXPORT_CHOICE:
        raise HTTPException(409, f"Session not awaiting export (state={s.state.value})")
    if payload.choice not in ("local", "hf", "both"):
        raise HTTPException(400, "choice must be local|hf|both")
    if payload.choice in ("hf", "both") and not payload.hf_repo_id:
        raise HTTPException(400, "hf_repo_id required for hf|both")

    bus = get_bus()

    # Mark what we're waiting for so ExportAgent knows when to close the session.
    pending: list[str] = []
    if payload.choice in ("local", "both"):
        pending.append("local")
    if payload.choice in ("hf", "both"):
        pending.append("hf")
    session_service.attach_artifact(s, "export_pending", pending)

    if payload.choice in ("local", "both"):
        await bus.publish(AgentEvent(
            session_id=session_id, kind="SaveLocalRequested", actor="user", payload={},
        ))
    if payload.choice in ("hf", "both"):
        await bus.publish(AgentEvent(
            session_id=session_id, kind="PushToHFRequested", actor="user",
            payload={"repo_id": payload.hf_repo_id},
        ))
    return {"ok": True, "pending": pending}


@router.post("/api/sessions/{session_id}/retry")
async def submit_retry(session_id: str, payload: RetryDecision) -> dict:
    s = session_service.get(session_id)
    if not s:
        raise _missing_session(session_id)
    bus = get_bus()
    kind = "RetryApproved" if payload.approve else "RetryDenied"
    await bus.publish(AgentEvent(
        session_id=session_id, kind=kind, actor="user",
        payload={"diff_id": payload.diff_id, "reason": payload.reason},
    ))
    return {"ok": True}


@router.post("/api/sessions/{session_id}/phase/{phase}/approve")
async def approve_phase(session_id: str, phase: str) -> dict:
    """User clicked Approve on a phase plan card. Lets the agent proceed."""
    s = session_service.get(session_id)
    if not s:
        raise _missing_session(session_id)
    bus = get_bus()
    await bus.publish(AgentEvent(
        session_id=session_id, kind="PhaseApproved", actor="user",
        payload={"phase": phase},
    ))
    return {"ok": True}


class PhaseComment(BaseModel):
    text: str


@router.post("/api/sessions/{session_id}/phase/{phase}/comment")
async def comment_phase(session_id: str, phase: str, payload: PhaseComment) -> dict:
    """User typed a comment under a phase plan card. The agent will read it
    on the blackboard and adjust before continuing."""
    s = session_service.get(session_id)
    if not s:
        raise _missing_session(session_id)
    bus = get_bus()
    await bus.publish(AgentEvent(
        session_id=session_id, kind="PhaseCommented", actor="user",
        payload={"phase": phase, "text": payload.text},
    ))
    return {"ok": True}


@router.post("/api/sessions/{session_id}/cancel")
async def cancel_session(session_id: str) -> dict:
    s = session_service.get(session_id)
    if not s:
        raise _missing_session(session_id)
    if s.state in (FSMState.DONE, FSMState.FAILED, FSMState.CANCELLED):
        return {"ok": True, "already": s.state.value}
    try:
        session_service.advance_state(s, FSMState.CANCELLED, reason="user cancelled")
    except ValueError as e:
        raise HTTPException(409, str(e)) from e
    bus = get_bus()
    await bus.publish(AgentEvent(
        session_id=session_id, kind="SessionClosed", actor="user", payload={"reason": "cancelled"},
    ))
    return {"ok": True}


@router.get("/api/sessions/{session_id}/audit")
def session_audit(session_id: str) -> dict:
    s = session_service.get(session_id)
    if not s:
        raise _missing_session(session_id)
    decisions = decision_log.list_for_session(session_id)
    return {
        "session_id": session_id,
        "decisions": [d.model_dump(mode="json") for d in decisions],
    }


@router.get("/api/sessions/{session_id}/blackboard")
def session_blackboard(session_id: str) -> dict:
    """Federated Blackboard contents — thoughts, plans, questions,
    critiques, decisions, nodes. Powers the Cognition tab in the UI."""
    s = session_service.get(session_id)
    if not s:
        raise _missing_session(session_id)
    from app.services.blackboard import get_blackboard

    bb = get_blackboard().read(session_id)
    return {"session_id": session_id, "blackboard": bb}


@router.get("/api/sessions/{session_id}/checkpoint")
def session_checkpoint(session_id: str) -> dict:
    """Latest persisted checkpoint snapshot for crash-recovery debugging."""
    s = session_service.get(session_id)
    if not s:
        raise _missing_session(session_id)
    from app.services.checkpoint import get_checkpoint_store

    snap = get_checkpoint_store().load(session_id)
    return {"session_id": session_id, "checkpoint": snap}


# Tool introspection — used by the UI's debug panel.

tools_router = APIRouter(tags=["tools"])


@tools_router.get("/api/tools")
def list_tools() -> dict:
    # Trigger registration via import side-effect.
    import app.tools  # noqa: F401
    return {"tools": list_descriptors()}
