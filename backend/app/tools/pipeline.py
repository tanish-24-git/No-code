"""Pipeline tools. Wrap pipeline_service so the agent operates through
schemas, never raw Pydantic mutation."""
from __future__ import annotations

from typing import Any

from app.api.schemas.pipeline import NodeGraph, PipelineCreate
from app.services import pipeline_service
from app.tools.registry import ToolContext, tool


@tool(
    name="pipeline.create",
    description="Create a fresh pipeline bound to a dataset.",
    input_schema={
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "description": {"type": "string"},
            "dataset_id": {"type": "string"},
        },
        "required": ["name", "dataset_id"],
    },
    side_effect="write_resource",
)
async def pipeline_create(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    rec = pipeline_service.create(PipelineCreate(
        name=args["name"],
        description=args.get("description"),
        dataset_id=args["dataset_id"],
    ))
    return rec.model_dump(mode="json")


@tool(
    name="pipeline.mutate_graph",
    description="Replace or extend the pipeline node graph. Operations: insert_node, remove_node, set_edges.",
    input_schema={
        "type": "object",
        "properties": {
            "pipeline_id": {"type": "string"},
            "nodes": {"type": "array"},
            "edges": {"type": "array"},
        },
        "required": ["pipeline_id"],
    },
    side_effect="write_resource",
)
async def pipeline_mutate_graph(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    pid = args["pipeline_id"]
    rec = pipeline_service.get(pid)
    if not rec:
        return {"error": "pipeline not found"}
    if "nodes" in args or "edges" in args:
        graph = NodeGraph(
            nodes=args.get("nodes") or rec.node_graph.nodes,
            edges=args.get("edges") or rec.node_graph.edges,
            viewport=rec.node_graph.viewport,
        )
        rec.node_graph = graph
        from datetime import datetime, timezone
        rec.updated_at = datetime.now(timezone.utc)
        from app.storage import store
        store.write("pipelines", rec.id, rec.model_dump(mode="json"))
    return rec.model_dump(mode="json")


@tool(
    name="pipeline.apply_config",
    description="Merge a config patch into the pipeline (with reasoning per field).",
    input_schema={
        "type": "object",
        "properties": {
            "pipeline_id": {"type": "string"},
            "config": {"type": "object"},
            "reasoning": {"type": "object"},
        },
        "required": ["pipeline_id", "config"],
    },
    side_effect="write_resource",
)
async def pipeline_apply_config(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    rec = pipeline_service.apply_agent_config(
        args["pipeline_id"],
        args.get("config", {}),
        args.get("reasoning", {}),
    )
    if not rec:
        return {"error": "pipeline not found"}
    return rec.model_dump(mode="json")


@tool(
    name="pipeline.summarize_for_user",
    description="Build a short, human-readable summary card for the chat panel.",
    input_schema={
        "type": "object",
        "properties": {
            "pipeline_id": {"type": "string"},
            "estimated_minutes": {"type": "number"},
        },
        "required": ["pipeline_id"],
    },
)
async def pipeline_summarize_for_user(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    pid = args["pipeline_id"]
    rec = pipeline_service.get(pid)
    if not rec:
        return {"error": "pipeline not found"}
    cfg = rec.config
    bullet = [
        f"**Base model:** {cfg.base_model}",
        f"**Method:** {cfg.training_method}, precision {cfg.precision}",
        f"**Batch:** {cfg.batch_size} (×{cfg.gradient_accumulation} grad accum)",
        f"**Seq len:** {cfg.max_seq_len}, LR {cfg.learning_rate}, {cfg.epochs} epoch(s)",
        f"**Task:** {cfg.task_type} → {cfg.output_type}",
    ]
    est = args.get("estimated_minutes")
    if est:
        bullet.append(f"**Estimated runtime:** ~{est:.0f} min")
    return {
        "title": f"Pipeline draft: {rec.name}",
        "summary": "\n".join(f"- {line}" for line in bullet),
        "pipeline_id": rec.id,
        "config": cfg.model_dump(mode="json"),
        "reasoning": rec.reasoning,
    }
