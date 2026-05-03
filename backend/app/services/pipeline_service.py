from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.api.schemas.pipeline import (
    NodeGraph,
    PipelineConfig,
    PipelineCreate,
    PipelineRecord,
    PipelineUpdate,
)
from app.storage import store


def _default_graph(dataset_id: str | None = None) -> NodeGraph:
    """Default 5-node DAG for a fresh pipeline. The frontend can rewire freely."""
    nodes = [
        {"id": "dataset", "type": "dataset", "position": {"x": 40, "y": 80}, "data": {"dataset_id": dataset_id}},
        {"id": "preprocess", "type": "preprocess", "position": {"x": 360, "y": 80}, "data": {}},
        {"id": "train", "type": "train", "position": {"x": 700, "y": 60}, "data": {}},
        {"id": "evaluate", "type": "evaluate", "position": {"x": 1040, "y": 80}, "data": {}},
        {"id": "export", "type": "export", "position": {"x": 1380, "y": 80}, "data": {}},
    ]
    edges = [
        {"id": "e1", "source": "dataset", "target": "preprocess"},
        {"id": "e2", "source": "preprocess", "target": "train"},
        {"id": "e3", "source": "train", "target": "evaluate"},
        {"id": "e4", "source": "evaluate", "target": "export"},
    ]
    return NodeGraph(nodes=nodes, edges=edges)


def create(payload: PipelineCreate) -> PipelineRecord:
    record_id = store.new_id()
    now = datetime.now(timezone.utc)
    cfg = PipelineConfig(project_name=payload.name, dataset_id=payload.dataset_id)
    rec = PipelineRecord(
        id=record_id,
        name=payload.name,
        description=payload.description,
        node_graph=_default_graph(payload.dataset_id),
        config=cfg,
        created_at=now,
        updated_at=now,
    )
    store.write("pipelines", record_id, rec.model_dump(mode="json"))
    return rec


def get(pipeline_id: str) -> PipelineRecord | None:
    raw = store.read("pipelines", pipeline_id)
    if not raw:
        return None
    return PipelineRecord(**raw)


def list_all() -> list[PipelineRecord]:
    out: list[PipelineRecord] = []
    for raw in store.list_all("pipelines"):
        try:
            out.append(PipelineRecord(**raw))
        except Exception:
            continue
    out.sort(key=lambda r: r.updated_at, reverse=True)
    return out


def update(pipeline_id: str, payload: PipelineUpdate) -> PipelineRecord | None:
    raw = store.read("pipelines", pipeline_id)
    if not raw:
        return None
    rec = PipelineRecord(**raw)
    data = payload.model_dump(exclude_none=True)
    for k, v in data.items():
        setattr(rec, k, v)
    rec.updated_at = datetime.now(timezone.utc)
    store.write("pipelines", pipeline_id, rec.model_dump(mode="json"))
    return rec


def apply_agent_config(pipeline_id: str, config_patch: dict[str, Any], reasoning: dict[str, str]) -> PipelineRecord | None:
    raw = store.read("pipelines", pipeline_id)
    if not raw:
        return None
    rec = PipelineRecord(**raw)
    merged = {**rec.config.model_dump(), **config_patch}
    rec.config = PipelineConfig(**merged)
    rec.reasoning = {**rec.reasoning, **reasoning}
    rec.is_agent_configured = True
    rec.updated_at = datetime.now(timezone.utc)
    store.write("pipelines", pipeline_id, rec.model_dump(mode="json"))
    return rec


def delete(pipeline_id: str) -> bool:
    return store.delete("pipelines", pipeline_id)
