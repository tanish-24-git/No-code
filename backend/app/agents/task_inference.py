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
    triggers = ("HardwareProfileCompleted", "UserClarificationReceived")

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

        await self.announce(
            session_id,
            phase="task",
            title="Inferring training task",
            summary="Categorizing dataset purpose based on field patterns and class distribution.",
            steps=[
                "Check for instruction/response pairs",
                "Detect potential classification labels",
                "Analyze class balance for classification candidates",
            ],
            outputs=["task_inference artifact"],
            requires_approval=True,
            parent=event.id,
        )

        comment = await self.wait_for_approval(session_id, "task")
        
        await self.think_delta(session_id, "Analyzing dataset structure and incorporating feedback to infer the best training task...", is_final=False, parent=event.id)
        
        # Build a dynamic prompt that includes user feedback
        prompt = (
            f"Dataset Summary: {info.get('row_count', 0)} rows, columns: {info.get('column_types', {})}.\n"
            f"User Feedback: '{comment}'\n\n"
            "Based on the data signals and user feedback, what is the best task type for fine-tuning? "
            "Possible types: instruction, chat, regression, classification, qa, summarization, extraction, etc. "
            "If the user suggested a transformation (e.g. 'convert to chat'), pick that as the 'chosen' task. "
            "Return ONLY a JSON object with 'chosen' (string), 'scores' (dict), and 'confidence' (float)."
        )
        system_prompt = "You are a machine learning data architect. You are adaptive, reasoning like a human about the user's goals."
        
        try:
            result_str = await self.call_llm(session_id, prompt, system=system_prompt, stream_thoughts=True, parent=event.id)
            import json, re
            m = re.search(r"\{.*\}", result_str, re.DOTALL)
            if not m:
                raise ValueError("LLM returned non-JSON task inference")
            
            result = json.loads(m.group(0))
            chosen = result.get("chosen", "instruction").lower()
            scores = result.get("scores", {chosen: 1.0})
            confidence = float(result.get("confidence", 0.9))
            missing = []
        except Exception as e:
            await self.emit_error(
                session_id,
                f"Task inference failed: {str(e)}. "
                "The agent was unable to classify the dataset task type."
            )
            return
        
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
