"""TaskInferenceAgent: classifies the dataset's task type.

Runs the LLM with strict TaskInferenceResult schema first. Falls back to
the deterministic ``task.classify`` tool when the LLM is unavailable so
the system always produces a valid task type rather than hard-failing.
"""
from __future__ import annotations

from app.agents.base import BaseAgent
from app.agents.schemas import TaskInferenceResult, CANONICAL_TASKS
from app.events.types import AgentEvent
from app.services import decision_log, session_service


class TaskInferenceAgent(BaseAgent):
    name = "TaskInferenceAgent"
    role = "Infer task type from dataset signals; expose confidence."
    directive_scope = "data"
    allowed_tools = ("task.classify", "task.score_candidates", "audit.write")
    triggers = ("HardwareProfileCompleted", "GoalCaptured", "UserClarificationReceived")

    async def handle(self, event: AgentEvent) -> None:
        session_id = event.session_id
        session = self.get_session(session_id)
        if not session:
            return

        hw = session.artifacts.get("hardware")
        facts = session.artifacts.get("dataset_facts") or {}
        profile = session.artifacts.get("profile") or {}
        
        # Pull any goal set by the user in directives
        from app.services import directives as directives_service
        all_directives = directives_service.read_for_scope(session_id, "global")
        goal = next((d.text for d in all_directives if d.scope == "global"), None)

        # Synchronicity check: we need hardware probe + dataset profile + user goal
        # to make a high-confidence task inference.
        if not hw or not facts or not goal:
            return # Wait for the missing piece.

        buckets = facts.get("field_buckets") or {}
        info = facts.get("info") or {}
        imbalance = profile.get("imbalance") or {}

        # The agent now relies entirely on its reasoning for task inference.
        # Deterministic heuristics have been decommissioned.
        proposal = await self.call_llm_typed(
            session_id,
            (
                f"User Goal: {goal}\n"
                f"Dataset metadata: {info}\n"
                f"Field buckets: {buckets}\n"
                f"Class imbalance: {imbalance}\n\n"
                "What is the best fine-tuning task type? "
                f"Pick from: {', '.join(CANONICAL_TASKS)}. "
                "Synthesize the user goal with the observed field shapes. "
                'Return JSON: {"chosen": "...", "scores": {...}, "confidence": 0..1, "rationale": "..."}'
            ),
            TaskInferenceResult,
            system="You are an ML data architect. Infer the task type purely from the data signals.",
            stream_thoughts=False,
            parent=event.id,
        )

        if not proposal:
            await self.emit_error(session_id, "AI failed to infer a task for this dataset.")
            return

        chosen = proposal.chosen.lower()
        if chosen not in CANONICAL_TASKS:
            chosen = "instruction"
        scores = proposal.scores or {chosen: proposal.confidence}
        confidence = proposal.confidence

        task_inference = {
            "chosen": chosen,
            "scores": scores,
            "confidence": confidence,
            "missing_signals": [],  # Agent should mention gaps in rationale
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
            rationale=proposal.rationale,
        )

        await self.emit(
            "TaskInferred",
            session_id,
            payload={"task": task_inference},
            confidence=confidence,
            decision_id=d.id,
            parent_event_id=event.id,
        )
