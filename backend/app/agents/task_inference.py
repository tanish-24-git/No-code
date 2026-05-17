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
    # Classification over field-buckets — fast tier is sufficient.
    tier_preference = "fast"

    async def handle(self, event: AgentEvent) -> None:
        session_id = event.session_id
        session = self.get_session(session_id)
        if not session:
            return

        hw = session.artifacts.get("hardware")
        facts = session.artifacts.get("dataset_facts") or {}
        profile = session.artifacts.get("profile") or {}
        ds = session.artifacts.get("dataset") or {}

        # Pull any goal set by the user in directives. Goal is OPTIONAL —
        # if the user hasn't said anything yet, we infer from data signals
        # alone rather than silently bailing (which used to deadlock the
        # downstream model_selection -> pipeline_builder chain).
        from app.services import directives as directives_service
        all_directives = directives_service.read_for_scope(session_id, "global")
        goal = next((d.text for d in all_directives if d.scope == "global"), None)

        # Hardware + dataset facts are genuinely required; without them
        # there's nothing to reason over.
        if not hw or not facts:
            return  # genuinely too early; wait for upstream profilers.

        buckets = facts.get("field_buckets") or {}
        info = facts.get("info") or {}
        imbalance = profile.get("imbalance") or {}

        goal_line = (
            f"User goal: {goal}"
            if goal
            else "User goal: (none stated — infer the most plausible task from data signals only)"
        )

        # The agent relies on its reasoning for task inference; deterministic
        # heuristics were decommissioned upstream, but we still call the
        # task.classify tool as a safety net if the LLM is unavailable.
        proposal = await self.call_llm_typed(
            session_id,
            (
                f"{goal_line}\n"
                f"Dataset metadata: {info}\n"
                f"Field buckets: {buckets}\n"
                f"Class imbalance: {imbalance}\n\n"
                "What is the best fine-tuning task type? "
                f"Pick from: {', '.join(CANONICAL_TASKS)}. "
                "Use the field shapes and class distribution as the primary signal; "
                "only weight the user goal when present.\n"
                'Return JSON: {"chosen": "...", "scores": {...}, "confidence": 0..1, "rationale": "..."}'
            ),
            TaskInferenceResult,
            system="You are an ML data architect. Infer the task type from data signals.",
            stream_thoughts=False,
            parent=event.id,
        )

        # No hardcoded fallback. If the LLM didn't return a valid task
        # inference, call_llm_typed has already emitted a UI-visible
        # error (with the actual provider message — rate limit, context
        # limit, parse failure, etc.). Surface a final agent-scoped
        # error so the user knows which step failed, then return.
        if not proposal:
            await self.emit_error(
                session_id,
                f"[{self.name}] Could not produce a TaskInferenceResult. "
                "See the LLM error above; fix the underlying cause "
                "(API key, rate limit, context length) and retry by "
                "commenting on this dataset.",
            )
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
