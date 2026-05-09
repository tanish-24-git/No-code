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
            await self.think_delta(session_id, f"Using user-specified task: {chosen}", is_final=True, parent=event.id)
        else:
            await self.think_delta(session_id, "Analyzing dataset structure to infer the best training task...", is_final=False, parent=event.id)
            
            prompt = f"Given a dataset with {info.get('row_count', 0)} rows and column types {info.get('column_types', {})}, what is the best task type (e.g. instruction, chat, regression, classification, qa)? Return ONLY a JSON object with 'chosen' (string), 'scores' (dict of string:float), and 'confidence' (float)."
            system_prompt = "You are a machine learning data architect. You analyze raw dataset signals and infer the task type."
            
            result_str = await self.call_llm(session_id, prompt, system=system_prompt, stream_thoughts=True, parent=event.id)
            
            # Fallback parsing in case of poor JSON formatting
            import json
            try:
                # Strip markdown code blocks if any
                if "```json" in result_str:
                    result_str = result_str.split("```json")[1].split("```")[0]
                elif "```" in result_str:
                    result_str = result_str.split("```")[1].split("```")[0]
                
                result = json.loads(result_str.strip())
                chosen = result.get("chosen", "instruction").lower()
                scores = result.get("scores", {chosen: 0.9})
                confidence = float(result.get("confidence", 0.9))
                missing = []
            except Exception:
                chosen = "instruction"
                scores = {"instruction": 0.8}
                confidence = 0.8
                missing = []
            
            await self.think_delta(session_id, f"\nInferred task type: {chosen} with {confidence*100:.0f}% confidence.", is_final=True, parent=event.id)

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
