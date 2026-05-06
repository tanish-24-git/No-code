"""TaskInferenceAgent: classifies the dataset's task type using deterministic
signals from profiling. The deterministic path is sufficient for most cases;
when an LLM is configured we may upgrade this in a follow-up."""
from __future__ import annotations

from app.agents.base import BaseAgent
from app.events.types import AgentEvent
from app.services import decision_log, session_service


class TaskInferenceAgent(BaseAgent):
    name = "TaskInferenceAgent"
    role = "Infer task type from dataset signals; expose confidence."
    allowed_tools = ("task.classify", "task.score_candidates", "audit.write")
    triggers = ("DatasetProfileCompleted", "UserClarificationReceived")

    async def handle(self, event: AgentEvent) -> None:
        session_id = event.session_id
        session = self.get_session(session_id)
        if not session:
            return

        facts = session.artifacts.get("dataset_facts") or {}
        profile = session.artifacts.get("profile") or {}
        buckets = facts.get("field_buckets") or {}
        info = facts.get("info") or {}
        imbalance = profile.get("imbalance") or {}

        # If user has answered q_task_type, take their answer as ground truth.
        forced_task = self._user_chose_task(session)
        if forced_task:
            chosen = forced_task
            scores = {chosen: 1.0}
            confidence = 1.0
            missing = self._missing_after_user(session, buckets, imbalance)
        else:
            result = await self.call_tool(
                "task.classify",
                {
                    "field_buckets": buckets,
                    "column_types": info.get("column_types", {}),
                    "row_count": info.get("row_count", 0),
                    "imbalance": imbalance,
                },
                session_id,
            )
            if "error" in result:
                await self.emit_error(session_id, result["error"])
                return
            chosen = result["chosen"]
            scores = result["scores"]
            confidence = float(result["confidence"])
            missing = list(result.get("missing_signals") or [])

        # Persist + audit.
        task_inference = {
            "chosen": chosen, "scores": scores, "confidence": confidence,
            "missing_signals": missing,
        }
        session_service.attach_artifact(session, "task_inference", task_inference)
        session_service.set_confidence(session, confidence)
        d = decision_log.record(
            session_id=session_id,
            agent=self.name,
            kind="task_inference",
            inputs={"buckets": buckets, "imbalance": imbalance},
            candidates=[{"task": k, "score": v} for k, v in scores.items()],
            chosen=chosen,
            confidence=confidence,
            rationale=f"Field-bucket heuristic; missing={missing}",
        )

        await self.emit(
            "TaskInferred",
            session_id,
            payload={"task": task_inference},
            confidence=confidence,
            decision_id=d.id,
            parent_event_id=event.id,
        )

    # ── Helpers ────────────────────────────────────────────────────────────

    def _user_chose_task(self, session) -> str | None:
        for a in reversed(session.clarifications):
            if a.question_id == "q_task_type":
                return str(a.value)
        return None

    def _missing_after_user(self, session, buckets, imbalance) -> list[str]:
        answered_qids = {a.question_id for a in session.clarifications}
        out: list[str] = []
        if "q_target_field" not in answered_qids and not buckets.get("output_like") and not imbalance.get("label_field"):
            out.append("target_field")
        if "q_input_fields" not in answered_qids and not buckets.get("input_like") and not buckets.get("instruction_like"):
            out.append("input_field")
        return out
