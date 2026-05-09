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

        await self.think_delta(session_id, "Evaluating available models against hardware constraints and dataset profile...", is_final=False, parent=event.id)
        
        prompt = f"Given hardware: {hw}, dataset profile: {profile}, and inferred task: {task}, suggest the best open-source model repository ID (e.g., 'Qwen/Qwen2.5-0.5B-Instruct', 'meta-llama/Llama-3.2-1B-Instruct') and provide 3 short reasons. Return ONLY a JSON object with 'repo_id' (string), 'label' (string), 'score' (float 0-1), and 'reasons' (list of strings)."
        system_prompt = "You are a machine learning systems architect. You select the optimal base model for fine-tuning based on hardware VRAM, dataset complexity, and task type."
        
        result_str = await self.call_llm(session_id, prompt, system=system_prompt, stream_thoughts=True, parent=event.id)
        
        import json
        try:
            if "```json" in result_str:
                result_str = result_str.split("```json")[1].split("```")[0]
            elif "```" in result_str:
                result_str = result_str.split("```")[1].split("```")[0]
            
            chosen = json.loads(result_str.strip())
            if "repo_id" not in chosen:
                raise ValueError("Missing repo_id")
        except Exception:
            # Fallback to a safe default if LLM fails
            chosen = {
                "repo_id": "Qwen/Qwen2.5-0.5B-Instruct",
                "label": "Qwen2.5 0.5B Instruct",
                "score": 0.85,
                "reasons": ["tiny model — runnable on CPU", "seq-len OK", "chat template available"]
            }
            
        candidates = [chosen]
        await self.think_delta(session_id, f"\nSelected model: {chosen.get('label')} ({chosen.get('repo_id')}).", is_final=True, parent=event.id)

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
