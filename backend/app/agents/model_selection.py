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

        # Stage 1: live Hub search.
        await self.think(session_id, "Searching HuggingFace Hub for models that fit your data...", parent=event.id)
        
        chosen_task = task.get("chosen", "instruction")
        # If the task is specialized (e.g. stock price, medical), use it as a query.
        is_specialized = chosen_task not in ("instruction", "chat", "qa", "classification", "extraction")
        
        searched = await self.call_tool(
            "model.search_hf",
            {
                "task": chosen_task,
                "query": chosen_task if is_specialized else None,
                "hardware": hw,
                "instruct_only": not is_specialized, # Don't limit to instruct if specialized
                "top_n": 12,
            },
            session_id,
        )


        if "error" in searched and not searched.get("candidates"):
            await self.emit_error(session_id, searched["error"])
            return

        hub_candidates = searched.get("candidates") or []
        
        # Stage 2: deterministic ranking against the joint context.
        ranked = await self.call_tool(
            "model.rank_candidates",
            {
                "hardware": hw,
                "profile": profile,
                "task": task,
                "top_n": 10,
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

        # Present the candidates to the user.
        candidate_list_str = "\n".join([f"{i+1}. {c['label']} ({c['repo_id']}) - {c['params_b']}B" for i, c in enumerate(candidates[:5])])
        
        await self.announce(
            session_id,
            phase="model_search",
            title="Model Selection",
            summary=(
                f"I've found {len(candidates)} candidates. The top recommendation is **{candidates[0]['label']}**. "
                f"You can approve this or comment to pick another one from the list:\n\n{candidate_list_str}"
            ),
            steps=[
                "Filter Hub for instruct-tuned models",
                "Rank by VRAM and parameter efficiency",
                "Validate chat template compatibility"
            ],
            requires_approval=True,
            parent=event.id,
        )

        chosen = candidates[0]
        comment = await self.wait_for_approval(session_id, "model_search")

        
        if comment and session.llm_provider:
            await self.think(session_id, f"Processing your feedback: '{comment}'")
            # Use LLM to resolve the chosen model from the comment.
            prompt = (
                f"User feedback: '{comment}'\n\n"
                f"Top Candidates:\n{candidate_list_str}\n\n"
                "The user wants to pick a model. Identify the HuggingFace repo_id (e.g., 'meta-llama/Llama-3.2-1B-Instruct') "
                "they are asking for. Return ONLY the repo_id. If they aren't asking for a specific model, "
                "return the current choice."
            )
            resolved_id = await self.call_llm(session_id, prompt, system="You are an expert model identifier.", parent=event.id)
            resolved_id = resolved_id.strip().strip('"').strip("'").split(" ")[0] # Grab first word/id
            
            # 1. Check top candidates
            match = next((c for c in candidates if resolved_id.lower() in c['repo_id'].lower()), None)
            
            # 2. Check full search results if not in top candidates
            if not match:
                full_list = searched.get("candidates", [])
                match = next((c for c in full_list if resolved_id.lower() in c['repo_id'].lower()), None)
            
            if match:
                chosen = match
                await self.think(session_id, f"Switching choice to **{chosen['label']}** ({chosen['repo_id']}) based on your feedback.")
            else:
                await self.think(session_id, f"I couldn't find a model matching '{resolved_id}' in the Hub search results. Keeping **{chosen['label']}**.")

        session_service.attach_artifact(session, "candidate_models", candidates)
        session_service.attach_artifact(session, "chosen_model", chosen)

        d = decision_log.record(
            session_id=session_id,
            agent=self.name,
            kind="model_choice",
            inputs={"hardware": hw, "profile": profile, "task": task, "comment": comment},
            candidates=candidates,
            chosen=chosen.get("repo_id"),
            confidence=float(chosen.get("score", 0.0)),
            rationale="; ".join(chosen.get("reasons", [])),
        )

        await self.complete(
            session_id,
            phase="model_search",
            summary=f"Selected {chosen['repo_id']}",
            artifacts={"chosen_repo_id": chosen.get("repo_id")},
            parent=event.id,
        )

        await self.emit(
            "CandidateModelsRanked",
            session_id,
            payload={"candidates": candidates, "chosen": chosen},
            parent_event_id=event.id,
            decision_id=d.id,
        )

