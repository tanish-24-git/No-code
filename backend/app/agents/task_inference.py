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

        # 1. LLM-typed inference (when configured).
        proposal = await self.call_llm_typed(
            session_id,
            (
                f"Dataset metadata: {info}\n"
                f"Field buckets: {buckets}\n"
                f"Class imbalance: {imbalance}\n\n"
                "What is the best fine-tuning task type? "
                f"Pick from: {', '.join(CANONICAL_TASKS)}. "
                'Return JSON: {"chosen": "...", "scores": {...}, "confidence": 0..1, "rationale": "..."}'
            ),
            TaskInferenceResult,
            system="You are an ML data architect. Be strict about the canonical task vocabulary.",
            stream_thoughts=False,
            parent=event.id,
        )

        if proposal:
            chosen = proposal.chosen.lower()
            if chosen not in CANONICAL_TASKS:
                # Soft-coerce: pick the closest canonical.
                chosen = "instruction"
            scores = proposal.scores or {chosen: proposal.confidence}
            confidence = proposal.confidence
        else:
            # Deterministic fallback - the legacy task.classify tool.
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
            chosen = (result.get("chosen") or "instruction").lower()
            if chosen not in CANONICAL_TASKS:
                chosen = "instruction"
            scores = result.get("scores") or {chosen: 1.0}
            confidence = float(result.get("confidence") or 0.7)

        task_inference = {
            "chosen": chosen,
            "scores": scores,
            "confidence": confidence,
            "missing_signals": [],
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
            rationale=getattr(proposal, "rationale", "") if proposal else "deterministic",
        )

        await self.emit(
            "TaskInferred",
            session_id,
            payload={"task": task_inference},
            confidence=confidence,
            decision_id=d.id,
            parent_event_id=event.id,
        )
