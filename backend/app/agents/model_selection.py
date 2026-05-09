"""ModelSelectionAgent.

Two-stage selection:

    1. Search the HuggingFace Hub (``model.search_hf``) for instruct-tuned
       base models that fit the user's task and hardware budget.
    2. Score the returned candidates with the deterministic ranker
       (``model.rank_candidates``) so the chosen model has a defensible
       rationale even when the LLM is in deterministic mode.

If the Hub is unreachable, ``model.search_hf`` falls back to the curated
catalogue automatically; this agent does not need to know which path was
taken. The result is the same shape either way.
"""
from __future__ import annotations

from app.agents.base import BaseAgent
from app.events.types import AgentEvent
from app.services import decision_log, session_service


class ModelSelectionAgent(BaseAgent):
    name = "ModelSelectionAgent"
    role = "Search the Hub, then rank candidates against hardware, dataset, and task."
    allowed_tools = ("model.search_hf", "model.rank_candidates", "model.estimate_fit", "audit.write")
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

        await self.announce(
            session_id,
            phase="model_search",
            title="Searching HuggingFace for the right base model",
            summary=(
                f"Scanning the Hub for instruct-tuned candidates that fit your "
                f"{hw.get('device', 'cpu').upper()} budget and "
                f"{task.get('chosen', 'task')} workload."
            ),
            steps=[
                "Filter by task and 'instruct' tag",
                "Cap parameter count to fit your VRAM",
                "Score each candidate against (hardware, profile, task)",
                "Pick the highest-scoring fit",
            ],
            outputs=["chosen_model artifact", "candidate_models shortlist"],
            requires_approval=False,
            parent=event.id,
        )

        # Stage 1: live Hub search (or fallback catalogue).
        searched = await self.call_tool(
            "model.search_hf",
            {
                "task": task.get("chosen", "instruction"),
                "hardware": hw,
                "instruct_only": True,
                "top_n": 12,
            },
            session_id,
        )
        if "error" in searched and not searched.get("candidates"):
            await self.emit_error(session_id, searched["error"])
            return

        hub_candidates = searched.get("candidates") or []
        await self.emit_message(
            session_id,
            (
                f"Searched HuggingFace ({searched.get('source', 'hub')}): "
                f"{len(hub_candidates)} candidate base models within "
                f"{searched.get('max_params_b', '?')}B parameters."
            ),
            parent=event.id,
        )
        if not hub_candidates:
            await self.emit_error(
                session_id,
                "no candidate models fit the device budget - "
                "consider attaching a GPU or asking for a smaller model family",
            )
            return

        # Stage 2: deterministic ranking against the joint context.
        ranked = await self.call_tool(
            "model.rank_candidates",
            {
                "hardware": hw,
                "profile": profile,
                "task": task,
                "top_n": 5,
                "shortlist": hub_candidates,
            },
            session_id,
        )
        if "error" in ranked:
            await self.emit_error(session_id, ranked["error"])
            return

        candidates = ranked.get("candidates") or []
        if not candidates:
            await self.emit_error(session_id, "no candidate models scored above the fit threshold")
            return

        chosen = candidates[0]

        session_service.attach_artifact(session, "candidate_models", candidates)
        session_service.attach_artifact(session, "chosen_model", chosen)

        d = decision_log.record(
            session_id=session_id,
            agent=self.name,
            kind="model_choice",
            inputs={"hardware": hw, "profile": profile, "task": task, "search_source": searched.get("source")},
            candidates=candidates,
            chosen=chosen.get("repo_id"),
            confidence=float(chosen.get("score", 0.0)),
            rationale="; ".join(chosen.get("reasons", [])),
        )

        await self.emit_message(
            session_id,
            f"Top candidate: **{chosen['label']}** ({chosen['repo_id']}) - "
            f"reasons: {', '.join(chosen.get('reasons', [])[:3])}.",
            parent=event.id,
        )
        await self.complete(
            session_id,
            phase="model_search",
            summary=f"chose {chosen['repo_id']} ({chosen['params_b']}B params)",
            artifacts={"chosen_repo_id": chosen.get("repo_id"), "score": chosen.get("score")},
            parent=event.id,
        )
        await self.emit(
            "CandidateModelsRanked",
            session_id,
            payload={"candidates": candidates, "chosen": chosen, "search_source": searched.get("source")},
            parent_event_id=event.id,
            decision_id=d.id,
        )
