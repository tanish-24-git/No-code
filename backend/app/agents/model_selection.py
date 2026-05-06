"""ModelSelectionAgent: ranks base models for the joint
(hardware × profile × task) context, then picks the top one."""
from __future__ import annotations

from app.agents.base import BaseAgent
from app.events.types import AgentEvent
from app.services import decision_log, session_service


class ModelSelectionAgent(BaseAgent):
    name = "ModelSelectionAgent"
    role = "Rank curated base models against hardware, dataset, and task."
    allowed_tools = ("model.rank_candidates", "model.estimate_fit", "audit.write")
    # Triggered when both hardware AND task are available; we listen to both
    # and self-gate on artifact presence.
    triggers = ("HardwareProfileCompleted", "PipelineDraftRequested")

    async def handle(self, event: AgentEvent) -> None:
        session_id = event.session_id
        session = self.get_session(session_id)
        if not session:
            return
        hw = session.artifacts.get("hardware")
        profile = session.artifacts.get("profile") or {}
        task = session.artifacts.get("task_inference")
        if not hw or not task:
            return  # Will fire again on the other event.
        if session.artifacts.get("candidate_models"):
            return  # Already done.

        ranked = await self.call_tool(
            "model.rank_candidates",
            {"hardware": hw, "profile": profile, "task": task, "top_n": 5},
            session_id,
        )
        if "error" in ranked:
            await self.emit_error(session_id, ranked["error"])
            return
        candidates = ranked.get("candidates") or []
        if not candidates:
            await self.emit_error(session_id, "no candidate models fit the hardware")
            return

        chosen = candidates[0]
        session_service.attach_artifact(session, "candidate_models", candidates)
        session_service.attach_artifact(session, "chosen_model", chosen)

        d = decision_log.record(
            session_id=session_id,
            agent=self.name,
            kind="model_choice",
            inputs={"hardware": hw, "profile": profile, "task": task},
            candidates=candidates,
            chosen=chosen.get("repo_id"),
            confidence=float(chosen.get("score", 0.0)),
            rationale="; ".join(chosen.get("reasons", [])),
        )

        await self.emit_message(
            session_id,
            f"Top candidate: **{chosen['label']}** ({chosen['repo_id']}) — "
            f"reasons: {', '.join(chosen.get('reasons', [])[:3])}.",
            parent=event.id,
        )
        await self.emit(
            "CandidateModelsRanked",
            session_id,
            payload={"candidates": candidates, "chosen": chosen},
            parent_event_id=event.id,
            decision_id=d.id,
        )
