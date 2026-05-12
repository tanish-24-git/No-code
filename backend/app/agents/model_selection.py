"""ModelSelectionAgent.

Honors any user directive ("use llama", "stick to 1B models") that was
captured anywhere upstream by reading the global directives bus before
searching, and again before resolving a comment on its own approval card.

Three stages:

    1. Pull family / size hints from directives.
    2. Live HF Hub search filtered by those hints + hardware budget.
    3. Deterministic ranker against (hardware * profile * task).

If the user comments on the proposed top candidate ("give me Llama 3.2 1B"),
we use ``call_llm_typed(ModelChoiceResolution)`` to extract a structured
intent and pick the closest match across the FULL search results. Falls
back to fuzzy matching when no LLM is configured.
"""
from __future__ import annotations

from app.agents.base import BaseAgent
from app.agents.schemas import ModelChoiceResolution
from app.events.types import AgentEvent
from app.services import decision_log, session_service
from app.services import directives as directives_service


class ModelSelectionAgent(BaseAgent):
    name = "ModelSelectionAgent"
    role = "Search the Hub, then rank candidates against hardware, dataset, and task."
    directive_scope = "model"
    allowed_tools = ("model.search_hf", "model.rank_candidates", "model.estimate_fit", "audit.write")
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
        if session.artifacts.get("chosen_model"):
            return  # Already settled.

        await self.materialize_node(
            session_id,
            {"id": "model", "type": "model", "position": {"x": 440, "y": 80},
             "data": {"label": "model"}},
            parent=event.id,
        )
        await self.materialize_edge(
            session_id,
            {"id": "e-preprocess-model", "source": "preprocess", "target": "model", "animated": True},
            parent=event.id,
        )

        # 1. Read user directives (e.g. "use Llama") and the overall session goal.
        hints = directives_service.model_hints(session_id)
        family = hints.get("family")
        size_b = hints.get("size_b")
        repo_id_hint = hints.get("repo_id")
        
        # Pull the global goal as a fallback query
        all_directives = directives_service.read_for_scope(session_id, "global")
        goal = next((d.text for d in all_directives if d.scope == "global"), None)

        await self.think(
            session_id,
            (
                "Searching HuggingFace Hub. "
                + (f"Honoring user hint: {family or repo_id_hint}. " if (family or repo_id_hint) else "")
                + (f"Querying for: '{goal}'. " if goal else "Broad search. ")
            ),
            parent=event.id,
        )

        # 2. Live search.
        chosen_task = task.get("chosen", "instruction")
        searched = await self.call_tool(
            "model.search_hf",
            {
                "task": chosen_task,
                "query": goal if not family else None,
                "family_hint": family,
                "size_hint_b": size_b,
                "hardware": hw,
                "precision": (session.artifacts.get("strategy") or {}).get("precision", "bf16"),
                "method": (session.artifacts.get("strategy") or {}).get("method", "lora"),
                "instruct_only": True,
                "top_n": 15,
            },
            session_id,
        )
        if "error" in searched and not searched.get("candidates"):
            await self.emit_error(session_id, searched["error"])
            return

        hub_candidates = searched.get("candidates") or []
        if not hub_candidates:
            await self.emit_error(
                session_id,
                "no candidate models fit the device budget - "
                "try a smaller family hint or attach a GPU",
            )
            return

        # 3. Deterministic rank.
        ranked = await self.call_tool(
            "model.rank_candidates",
            {
                "hardware": hw,
                "profile": profile,
                "task": task,
                "top_n": 8,
                "shortlist": hub_candidates,
            },
            session_id,
        )
        candidates = ranked.get("candidates") or []
        if not candidates:
            await self.emit_error(session_id, "no candidate scored above the fit threshold")
            return

        # If a specific repo_id was hinted, prefer the exact match.
        if repo_id_hint:
            match = next((c for c in candidates if c["repo_id"].lower() == repo_id_hint.lower()), None)
            if match:
                candidates = [match] + [c for c in candidates if c is not match]
        elif family or size_b:
            # Boost candidates matching family or size hints so user intent
            # isn't buried by the raw ranker's CPU-efficiency bias.
            def _hint_score(c):
                s = 0.0
                if family and family in c["repo_id"].lower():
                    s += 2.0
                if size_b and abs(c["params_b"] - float(size_b)) < 0.1:
                    s += 1.0
                return s
            candidates.sort(key=lambda c: (c["score"] + _hint_score(c)), reverse=True)

        candidate_list_str = "\n".join(
            f"{i + 1}. {c['label']} ({c['repo_id']}) - {c['params_b']}B"
            for i, c in enumerate(candidates[:5])
        )

        await self.announce(
            session_id,
            phase="model_search",
            title="Model selection",
            summary=(
                f"Found {len(candidates)} candidates"
                + (f" matching '{family}'" if family else "")
                + f". Top recommendation: **{candidates[0]['label']}**.\n\n"
                f"{candidate_list_str}\n\n"
                "Approve to use the top one, or comment to pick a "
                "specific model (e.g. 'use llama 3.2 1B')."
            ),
            steps=[
                "Filter Hub for instruct-tuned models",
                "Apply user directive hints",
                "Rank by VRAM and parameter efficiency",
                "Validate chat template compatibility",
            ],
            requires_approval=True,
            parent=event.id,
        )

        chosen = candidates[0]
        comment = await self.wait_for_approval(session_id, "model_search")

        if comment:
            chosen = await self._resolve_comment(session_id, comment, candidates, searched, event.id) or chosen

        session_service.attach_artifact(session, "candidate_models", candidates)
        session_service.attach_artifact(session, "chosen_model", chosen)

        d = decision_log.record(
            session_id=session_id,
            agent=self.name,
            kind="model_choice",
            inputs={"hardware": hw, "profile": profile, "task": task, "comment": comment, "hints": hints},
            candidates=candidates,
            chosen=chosen.get("repo_id"),
            confidence=float(chosen.get("score", 0.0)),
            rationale="; ".join(chosen.get("reasons", [])),
        )

        await self.emit_message(
            session_id,
            f"Picked **{chosen['label']}** ({chosen['repo_id']}) - "
            f"{', '.join(chosen.get('reasons', [])[:3])}.",
            parent=event.id,
        )
        await self.complete(
            session_id,
            phase="model_search",
            summary=f"chose {chosen['repo_id']} ({chosen['params_b']}B)",
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

    async def _resolve_comment(
        self,
        session_id: str,
        comment: str,
        candidates: list[dict],
        searched: dict,
        parent: str,
    ) -> dict | None:
        """Map a user comment like 'use llama 3.2 1B' onto the closest
        candidate, using the LLM with strict JSON validation when available
        and fuzzy substring matching otherwise."""
        await self.think(session_id, f"Mapping your feedback to a concrete model id: '{comment}'", parent=parent)

        # Try LLM-typed resolution first.
        resolution = await self.call_llm_typed(
            session_id,
            (
                f"User feedback: {comment}\n\n"
                f"Available candidates:\n" + "\n".join(
                    f"- {c['repo_id']} ({c['params_b']}B)" for c in candidates[:8]
                ) + "\n\n"
                "Identify which candidate matches the user's request. "
                "Return JSON with: repo_id (exact id from the list when possible), "
                "family (e.g. 'llama'), size_b (number), rationale."
            ),
            ModelChoiceResolution,
            system="You map natural-language model requests onto repo ids.",
            stream_thoughts=False,
            parent=parent,
        )
        target_repo: str | None = None
        target_family: str | None = None
        target_size: float | None = None
        if resolution:
            target_repo = resolution.repo_id
            target_family = (resolution.family or "").lower() or None
            target_size = resolution.size_b

        # Fallback: extract from the comment itself.
        if not target_family and not target_repo:
            lower = comment.lower()
            for fam in ("llama", "qwen", "phi", "mistral", "gemma", "smollm",
                         "deepseek", "yi", "tinyllama"):
                if fam in lower:
                    target_family = fam
                    break

        # Hunt the candidate list, then the full search results.
        full_list = (searched.get("candidates") or []) + candidates
        if target_repo:
            for c in full_list:
                if c["repo_id"].lower() == target_repo.lower():
                    return c
            for c in full_list:
                if target_repo.lower() in c["repo_id"].lower():
                    return c
        if target_family:
            best = None
            for c in full_list:
                if target_family in c["repo_id"].lower():
                    if target_size is None:
                        return c
                    if best is None or abs(c["params_b"] - target_size) < abs(best["params_b"] - target_size):
                        best = c
            if best:
                return best

        await self.think(session_id, "Could not match the requested model in the search results.", parent=parent)
        return None
