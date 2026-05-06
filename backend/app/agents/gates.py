"""Gates: tiny non-LLM routers that consume an event and decide whether to
advance the workflow or kick off a clarification / approval. Keeping these
small and deterministic is what makes the orchestration testable."""
from __future__ import annotations

from app.agents.base import BaseAgent
from app.api.schemas.session import FSMState
from app.events.types import AgentEvent
from app.orchestration.confidence import THRESHOLDS
from app.orchestration.policy import decide_plan_approval
from app.services import session_service


class ConfidenceGate(BaseAgent):
    """After TaskInferred: either ask a clarification or kick off planning."""

    name = "ConfidenceGate"
    role = "Route TaskInferred → plan or clarify"
    allowed_tools = ()
    triggers = ("TaskInferred",)

    async def handle(self, event: AgentEvent) -> None:
        session_id = event.session_id
        session = self.get_session(session_id)
        if not session:
            return
        ti = (session.artifacts.get("task_inference") or {})
        confidence = float(ti.get("confidence") or 0.0)
        missing = list(ti.get("missing_signals") or [])

        # Question budget exhausted? Force progress with current confidence.
        out_of_budget = session.question_budget_used >= session.question_budget_max

        if confidence >= THRESHOLDS.proceed_to_plan and not missing:
            if session.state in (FSMState.PROFILING, FSMState.CLARIFYING):
                session_service.advance_state(session, FSMState.PLANNING, reason="confidence ok")
            await self.emit(
                "PipelineDraftRequested",
                session_id,
                payload={"task": ti},
                parent_event_id=event.id,
                confidence=confidence,
            )
            return

        if out_of_budget:
            await self.emit_message(
                session_id,
                "Proceeding with my best guess — we can iterate on the plan after seeing it.",
                parent=event.id,
            )
            if session.state in (FSMState.PROFILING, FSMState.CLARIFYING):
                session_service.advance_state(session, FSMState.PLANNING, reason="budget exhausted")
            await self.emit(
                "PipelineDraftRequested",
                session_id,
                payload={"task": ti, "forced": True},
                parent_event_id=event.id,
                confidence=confidence,
            )
            return

        # Otherwise: ask.
        if session.state == FSMState.PROFILING:
            session_service.advance_state(session, FSMState.CLARIFYING, reason="low confidence")
        await self.emit(
            "IntentConfidenceLow",
            session_id,
            payload={"missing": missing, "task": ti},
            parent_event_id=event.id,
            confidence=confidence,
        )


class ApprovalGate(BaseAgent):
    """After PipelineDraftCreated: auto-approve or ask user."""

    name = "ApprovalGate"
    role = "Route PipelineDraftCreated → approve or request approval"
    allowed_tools = ()
    triggers = ("PipelineDraftCreated",)

    async def handle(self, event: AgentEvent) -> None:
        session_id = event.session_id
        session = self.get_session(session_id)
        if not session:
            return

        est = float(event.payload.get("estimated_minutes") or 999.0)
        confidence = float(session.confidence or 0.0)
        decision = decide_plan_approval(est, confidence)
        if decision == "approve":
            await self.emit(
                "PipelineApproved",
                session_id,
                actor="system",
                payload={"pipeline_id": session.pipeline_id, "auto": True},
                parent_event_id=event.id,
                rationale="auto-approved: short job, high confidence",
            ) if False else await self.emit(
                "PipelineApproved",
                session_id,
                payload={"pipeline_id": session.pipeline_id, "auto": True},
                parent_event_id=event.id,
                rationale="auto-approved: short job, high confidence",
            )
            return

        if session.state == FSMState.PLANNING:
            session_service.advance_state(session, FSMState.AWAITING_APPROVAL, reason="needs user approval")
        await self.emit(
            "PipelineApprovalRequested",
            session_id,
            payload={
                "pipeline_id": session.pipeline_id,
                "summary": event.payload.get("summary"),
                "estimated_minutes": est,
            },
            parent_event_id=event.id,
        )
