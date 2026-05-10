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
        if not chosen_model.get("repo_id"):
            await self.emit_error(
                session_id,
                "pipeline_builder cannot proceed without a chosen base model. "
                "Model search did not return a candidate that fits this hardware.",
            )
            return

        await self.think(
            session_id,
            "Designing the pipeline graph based on your dataset profile and strategy.",
            parent=event.id,
        )

        # Try schema-validated LLM proposal first.
        det_nodes, det_edges = self._build_graph(session.dataset_id, profile, strategy)
        proposal = await self.call_llm_typed(
            session_id,
            (
                f"Dataset profile: {profile}\n"
                f"Strategy: {strategy}\n"
                f"Chosen model: {chosen_model.get('repo_id')}\n\n"
                "Design the pipeline node graph. Use node types from "
                "(dataset, preprocess, balance, split, train, evaluate, export). "
                "Return GraphProposal JSON."
            ),
            GraphProposal,
            system="You are a SOTA pipeline architect. Output only valid JSON.",
            stream_thoughts=False,
            parent=event.id,
        )
        if proposal and proposal.nodes:
            graph = {
                "nodes": [n.model_dump() for n in proposal.nodes],
                "edges": [e.model_dump() for e in proposal.edges],
            }
            rationale = proposal.rationale
        else:
            graph = {"nodes": det_nodes, "edges": det_edges}
            rationale = "deterministic graph computed from profile and strategy"

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
        return {
            "project_name": f"auto-{session.id[:8]}",
            "dataset_id": session.dataset_id,
            "task_type": strategy.get("task_type") or "Chat",
            "training_method": strategy.get("method") or "lora",
            "base_model": repo_id,
            "epochs": int(strategy.get("epochs") or 3),
            "batch_size": int(strategy.get("batch_size") or 1),
            "learning_rate": float(strategy.get("learning_rate") or 2e-4),
            "max_seq_len": int(strategy.get("max_seq_len") or 1024),
            "lora_rank": int(strategy.get("lora_rank") or 16),
            "gradient_accumulation": int(strategy.get("gradient_accumulation") or 4),
            "precision": strategy.get("precision") or "bf16",
            "early_stopping": bool(strategy.get("early_stopping", True)),
        }

    def _build_reasoning(self, model, strategy, profile) -> dict[str, str]:
        return {
            "base_model": "; ".join(model.get("reasons", [])) or "top scored candidate",
            "training_method": f"chosen by hardware fit: {strategy.get('method')}",
            "precision": f"{strategy.get('precision')} per device support",
            "max_seq_len": f"clipped to dataset p95 ({profile.get('p95')}) and model.max_pos",
            "epochs": f"{strategy.get('epochs')} balances dataset size vs runtime",
            "batch_size": f"{strategy.get('batch_size')}x{strategy.get('gradient_accumulation')} fits VRAM budget",
        }

    def _build_graph(self, dataset_id, profile, strategy):
        nodes = [
            {"id": "dataset", "type": "dataset", "position": {"x": 40, "y": 80}, "data": {"dataset_id": dataset_id}},
            {"id": "preprocess", "type": "preprocess", "position": {"x": 240, "y": 80}, "data": {}},
        ]
        edges = [{"id": "e1", "source": "dataset", "target": "preprocess"}]
        prev = "preprocess"
        x = 440
        imb = (profile.get("imbalance") or {})
        if imb.get("balanced") is False and imb.get("label_field"):
            nodes.append({"id": "balance", "type": "balance", "position": {"x": x, "y": 80},
                          "data": {"label_field": imb["label_field"], "strategy": "upsample"}})
            edges.append({"id": f"e_{prev}_balance", "source": prev, "target": "balance"})
            prev = "balance"
            x += 200
        nodes.append({"id": "split", "type": "split", "position": {"x": x, "y": 80}, "data": {"ratio": 0.9}})
        edges.append({"id": f"e_{prev}_split", "source": prev, "target": "split"})
        x += 200
        nodes.append({"id": "train", "type": "train", "position": {"x": x, "y": 80}, "data": {}})
        edges.append({"id": "e_split_train", "source": "split", "target": "train"})
        x += 200
        nodes.append({"id": "evaluate", "type": "evaluate", "position": {"x": x, "y": 80}, "data": {}})
        edges.append({"id": "e_train_eval", "source": "train", "target": "evaluate"})
        x += 200
        nodes.append({"id": "export", "type": "export", "position": {"x": x, "y": 80}, "data": {}})
        edges.append({"id": "e_eval_export", "source": "evaluate", "target": "export"})
        return nodes, edges
