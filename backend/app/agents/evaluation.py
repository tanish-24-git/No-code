"""EvaluationAgent: post-training metrics summary. Emits EvaluationCompleted
which the ExportAgent listens for."""
from __future__ import annotations

from app.agents.base import BaseAgent
from app.api.schemas.session import FSMState
from app.events.types import AgentEvent
from app.services import session_service


class EvaluationAgent(BaseAgent):
    name = "EvaluationAgent"
    role = "Run an evaluation suite and summarise results."
    allowed_tools = ("eval.run_suite", "eval.compare_baseline", "audit.write")
    triggers = ("TrainingCompleted",)

    async def handle(self, event: AgentEvent) -> None:
        session_id = event.session_id
        session = self.get_session(session_id)
        if not session or not session.job_id:
            return

        if session.state == FSMState.MONITORING:
            session_service.advance_state(session, FSMState.EVALUATING, reason="evaluating output")

        await self.announce(
            session_id,
            phase="evaluate",
            title="Evaluating the trained model",
            summary="Running the basic eval suite and comparing to the baseline.",
            steps=["Load held-out validation slice", "Compute final loss + score", "Delta vs. baseline"],
            outputs=["evaluation artifact"],
            parent=event.id,
        )
        await self.materialize_node(
            session_id,
            {"id": "evaluate", "type": "evaluate", "position": {"x": 1380, "y": 80},
             "data": {"label": "evaluate"}},
            parent=event.id,
        )
        await self.emit("EvaluationStarted", session_id, payload={"job_id": session.job_id})

        result = await self.call_tool("eval.run_suite", {"job_id": session.job_id, "suite": "basic"}, session_id)
        if "error" in result:
            await self.emit_error(session_id, result["error"])
            session_service.fail(session, result["error"])
            return

        compare = await self.call_tool("eval.compare_baseline", {"trained_score": result.get("score") or 0.5}, session_id)
        evaluation = {**result, "vs_baseline": compare}
        session_service.attach_artifact(session, "evaluation", evaluation)

        await self.emit_message(
            session_id,
            f"Evaluation: final loss {result.get('final_loss')}, score "
            f"{result.get('score')}; baseline delta {compare.get('delta')}.",
            parent=event.id,
        )
        await self.complete(
            session_id,
            phase="evaluate",
            summary=f"score {result.get('score')}, baseline delta {compare.get('delta')}",
            artifacts={"score": result.get("score")},
            parent=event.id,
        )
        await self.emit(
            "EvaluationCompleted",
            session_id,
            payload={"evaluation": evaluation},
            parent_event_id=event.id,
        )
