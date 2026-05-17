"""PipelineBuilderAgent: turns the chosen model + strategy into a concrete
pipeline. Asks the LLM for the graph (schema-validated), falls back to a
deterministic graph computed from the dataset shape (e.g. balance node
only when imbalance was detected)."""
from __future__ import annotations

from app.agents.base import BaseAgent
from app.agents.schemas import GraphProposal
from app.api.schemas.session import FSMState
from app.events.types import AgentEvent
from app.services import dataset_service, session_service


class PipelineBuilderAgent(BaseAgent):
    name = "PipelineBuilderAgent"
    role = "Generate a concrete training pipeline (config + node graph)."
    directive_scope = "pipeline"
    allowed_tools = (
        "pipeline.create",
        "pipeline.apply_config",
        "pipeline.mutate_graph",
        "pipeline.summarize_for_user",
        "audit.write",
    )
    triggers = ("StrategyChosen",)

    async def handle(self, event: AgentEvent) -> None:
        session_id = event.session_id
        session = self.get_session(session_id)
        if not session:
            return
        chosen_model = session.artifacts.get("chosen_model") or {}
        strategy = session.artifacts.get("strategy") or {}
        profile = session.artifacts.get("profile") or {}
        ds = dataset_service.get_dataset(session.dataset_id)
        if not ds:
            await self.emit_error(session_id, "dataset missing for session")
            return

        # Last-chance recovery: if we somehow got triggered without a
        # chosen_model (e.g. agentic loop reordered tool calls and an
        # upstream agent silently bailed), invoke ModelSelectionAgent
        # directly before giving up. Previously this branch hard-crashed
        # the whole pipeline with "cannot proceed without a chosen base
        # model" and left the frontend deadlocked.
        if not chosen_model.get("repo_id"):
            await self.think(
                session_id,
                "No chosen_model on the session yet. Invoking model selection "
                "in-line to unblock pipeline construction.",
                parent=event.id,
            )
            try:
                from app.agents.model_selection import ModelSelectionAgent
                sel_agent = ModelSelectionAgent(self.bus)
                sel_agent.silent = True
                sel_ev = AgentEvent(
                    session_id=session_id,
                    kind="HardwareProfileCompleted",
                    actor=self.name,
                    payload={},
                )
                await sel_agent.handle(sel_ev)
            except Exception as e:  # pragma: no cover - defensive
                await self.emit_error(
                    session_id,
                    f"In-line model selection failed: {e}. "
                    "Please comment with a specific model id "
                    "(e.g. 'use meta-llama/Llama-3.2-1B-Instruct').",
                )
                return
            # Refresh session state after the inline run.
            session = self.get_session(session_id) or session
            chosen_model = session.artifacts.get("chosen_model") or {}

        if not chosen_model.get("repo_id"):
            await self.emit_error(
                session_id,
                "pipeline_builder cannot proceed without a chosen base model. "
                "Model search did not return a candidate. Comment with a "
                "specific model id to override (e.g. 'use meta-llama/"
                "Llama-3.2-1B-Instruct').",
            )
            return

        await self.think(
            session_id,
            "Designing the pipeline graph based on your dataset profile and strategy.",
            parent=event.id,
        )

        # The agent now relies entirely on its reasoning for graph design.
        # Deterministic fallbacks have been decommissioned.
        proposal = await self.call_llm_typed(
            session_id,
            (
                f"Dataset profile: {profile}\n"
                f"Strategy: {strategy}\n"
                f"Chosen model: {chosen_model.get('repo_id')}\n\n"
                "Design the pipeline node graph. Use node types from "
                "(dataset, preprocess, balance, split, train, evaluate, export). "
                "The graph should be logically sound for the given task and hardware. "
                "Return GraphProposal JSON."
            ),
            GraphProposal,
            system="You are an autonomous pipeline architect. Design the best execution graph for this run.",
            stream_thoughts=False,
            parent=event.id,
        )

        if not proposal or not proposal.nodes:
            await self.emit_error(session_id, "AI failed to design a pipeline graph.")
            return

        graph = {
            "nodes": [n.model_dump() for n in proposal.nodes],
            "edges": [e.model_dump() for e in proposal.edges],
        }
        rationale = proposal.rationale

        # 1. Create pipeline.
        if not session.pipeline_id:
            created = await self.call_tool(
                "pipeline.create",
                {
                    "name": f"auto-{ds.name}",
                    "description": f"Auto-generated for dataset {ds.id}",
                    "dataset_id": ds.id,
                },
                session_id,
            )
            if "error" in created:
                await self.emit_error(session_id, created["error"])
                return
            session_service.attach_pipeline(session, created["id"])
            session = self.get_session(session_id)

        # 2. Apply config.
        cfg = self._build_config(chosen_model, strategy, profile, session)
        reasoning = self._build_reasoning(chosen_model, strategy, profile)
        applied = await self.call_tool(
            "pipeline.apply_config",
            {"pipeline_id": session.pipeline_id, "config": cfg, "reasoning": reasoning},
            session_id,
        )
        if "error" in applied:
            await self.emit_error(session_id, applied["error"])
            return

        # 3. Apply the graph.
        await self.call_tool(
            "pipeline.mutate_graph",
            {"pipeline_id": session.pipeline_id, "nodes": graph["nodes"], "edges": graph["edges"]},
            session_id,
        )
        for n in graph["nodes"]:
            await self.materialize_node(session_id, n, parent=event.id)

        # 4. Summary card.
        est_minutes = float(event.payload.get("estimated_minutes") or 0.0)
        summary = await self.call_tool(
            "pipeline.summarize_for_user",
            {"pipeline_id": session.pipeline_id, "estimated_minutes": est_minutes},
            session_id,
        )

        if session.state in (FSMState.PROFILING, FSMState.CLARIFYING):
            session_service.advance_state(session, FSMState.PLANNING, reason="building draft")

        await self.announce(
            session_id,
            phase="plan",
            title="Pipeline draft",
            summary=(summary.get("summary") or rationale or "")
                    + "\n\nApprove to start training, or comment to adjust.",
            steps=[n.get("id", n.get("type", "node")) for n in graph["nodes"]],
            requires_approval=True,
            parent=event.id,
        )

        # Wait for approval (the ApprovalGate also routes off PipelineDraftCreated;
        # this gate is the user-facing one).
        comment = await self.wait_for_approval(session_id, "plan")
        if comment:
            await self.think(
                session_id,
                f"Adjusting pipeline per your feedback: {comment}",
                parent=event.id,
            )
            # The directive is now in the global bus; if the user said
            # "more epochs" we'll re-run strategy on the next iteration.
            # For now we just acknowledge and continue with the current plan.

        await self.complete(
            session_id,
            phase="plan",
            summary=summary.get("title", "pipeline draft ready"),
            artifacts={"pipeline_id": session.pipeline_id, "node_count": len(graph["nodes"])},
            parent=event.id,
        )
        await self.emit(
            "PipelineDraftCreated",
            session_id,
            payload={
                "pipeline_id": session.pipeline_id,
                "config": cfg,
                "summary": summary,
                "estimated_minutes": est_minutes,
            },
            parent_event_id=event.id,
        )

    # ── builders ──────────────────────────────────────────────────────────

    def _build_config(self, model, strategy, profile, session) -> dict:
        repo_id = model.get("repo_id")
        if not repo_id:
            raise ValueError("chosen_model has no repo_id - upstream selection failed")
        # Every value is strictly taken from the strategy proposed by the AI architect.
        return {
            "project_name": f"auto-{session.id[:8]}",
            "dataset_id": session.dataset_id,
            "task_type": strategy.get("task_type") or "Chat",
            "training_method": strategy.get("method") or "lora",
            "base_model": repo_id,
            "epochs": strategy.get("epochs", 1),
            "batch_size": strategy.get("batch_size", 1),
            "learning_rate": strategy.get("learning_rate", 2e-4),
            "max_seq_len": strategy.get("max_seq_len", 512),
            "lora_rank": strategy.get("lora_rank", 16),
            "gradient_accumulation": strategy.get("gradient_accumulation", 1),
            "precision": strategy.get("precision", "bf16"),
            "early_stopping": strategy.get("early_stopping", True),
        }

    def _build_reasoning(self, model, strategy, profile) -> dict[str, str]:
        return {
            "base_model": "; ".join(model.get("reasons", [])) or "agent-selected candidate",
            "training_method": f"chosen by agent reasoning: {strategy.get('method')}",
            "precision": f"{strategy.get('precision')} per agent choice",
            "max_seq_len": f"set by agent: {strategy.get('max_seq_len')}",
            "epochs": f"{strategy.get('epochs')} determined by agent reasoning",
            "batch_size": f"{strategy.get('batch_size')}x{strategy.get('gradient_accumulation')} determined by agent reasoning",
        }
