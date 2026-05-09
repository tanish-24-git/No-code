"""ExecutionAgent: actually starts the training job once the pipeline has
been approved (auto or by user)."""
from __future__ import annotations

from app.agents.base import BaseAgent
from app.api.schemas.session import FSMState
from app.events.types import AgentEvent
from app.services import job_service, session_service


class ExecutionAgent(BaseAgent):
    name = "ExecutionAgent"
    role = "Start the training job for an approved pipeline."
    allowed_tools = ("job.start", "job.stop", "audit.write")
    triggers = ("PipelineApproved",)

    async def handle(self, event: AgentEvent) -> None:
        session_id = event.session_id
        session = self.get_session(session_id)
        if not session or not session.pipeline_id:
            return

        if session.state in (FSMState.PLANNING, FSMState.AWAITING_APPROVAL):
            session_service.advance_state(session, FSMState.EXECUTING, reason="starting job")

        await self.announce(
            session_id,
            phase="execute",
            title="Starting training",
            summary="Approved. Spinning up the training worker now.",
            steps=["Allocate the worker", "Stream loss + step metrics live"],
            outputs=["job_id"],
            parent=event.id,
        )
        result = await self.call_tool(
            "job.start", {"pipeline_id": session.pipeline_id}, session_id
        )
        if "error" in result:
            await self.emit_error(session_id, result["error"])
            session_service.fail(session, result["error"])
            return

        job_id = result["job_id"]
        session_service.attach_job(session, job_id)
        job_service.bind_job_to_session(job_id, session_id)
        await self.emit_message(session_id, f"Training started (job `{job_id[:8]}`).", parent=event.id)
        await self.complete(
            session_id,
            phase="execute",
            summary=f"job {job_id[:8]} dispatched",
            artifacts={"job_id": job_id},
            parent=event.id,
        )
        await self.emit(
            "PipelineExecutionStarted",
            session_id,
            payload={"job_id": job_id, "pipeline_id": session.pipeline_id},
            parent_event_id=event.id,
        )
