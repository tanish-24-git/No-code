"""Dataset management routes. The upload endpoint is the canonical entry
point of the agent runtime: a successful upload starts an AgentSession and
publishes DatasetUploaded on the bus, which kicks off the whole pipeline.

The response carries `session_id` so the frontend can immediately attach
the SSE stream and render the agent activity panel before the user types.
"""
from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Response, UploadFile
from pydantic import BaseModel

from app.api.schemas.dataset import DatasetSchema
from app.events.bus import get_bus
from app.events.types import AgentEvent
from app.services import dataset_service, session_service
from app.tools.llm import ping_llm


router = APIRouter(prefix="/api/datasets", tags=["datasets"])


class LLMProbe(BaseModel):
    ok: bool
    mode: str           # "full_agent" | "deterministic"
    provider: str | None = None
    model: str | None = None
    latency_ms: float = 0.0
    detail: str = ""


class DatasetUploadResponse(BaseModel):
    dataset: DatasetSchema
    session_id: str
    llm_probe: LLMProbe


@router.post("/upload", response_model=DatasetUploadResponse, status_code=201)
async def upload_dataset(file: UploadFile = File(...)) -> DatasetUploadResponse:
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")
    try:
        dataset = dataset_service.save_uploaded(file.filename or "dataset", contents)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    # Probe the configured LLM provider before booting the swarm. The result
    # determines whether agents run in full LLM-driven mode or deterministic
    # fallback mode; either way the user sees an explicit verdict in the UI.
    probe = await ping_llm()

    # Start the agent session and emit the first event. The orchestrator
    # posts the welcome message; downstream agents take it from there.
    session = session_service.start_for_dataset(dataset.id)
    # Persist the probe verdict on the session so agents can branch on it.
    session_service.attach_artifact(session, "llm_probe", probe)
    bus = get_bus()

    await bus.publish(AgentEvent(
        session_id=session.id, kind="SessionStarted", actor="system",
        payload={"dataset_id": dataset.id, "llm_probe": probe},
    ))
    await bus.publish(AgentEvent(
        session_id=session.id, kind="DatasetUploaded", actor="system",
        payload={"dataset_id": dataset.id, "name": dataset.name, "llm_probe": probe},
    ))

    return DatasetUploadResponse(
        dataset=dataset,
        session_id=session.id,
        llm_probe=LLMProbe(**probe),
    )


@router.get("", response_model=list[DatasetSchema])
def list_datasets() -> list[DatasetSchema]:
    return dataset_service.list_datasets()


@router.get("/{dataset_id}", response_model=DatasetSchema)
def get_dataset(dataset_id: str) -> DatasetSchema:
    d = dataset_service.get_dataset(dataset_id)
    if not d:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return d


@router.delete("/{dataset_id}", status_code=204, response_class=Response)
def delete_dataset(dataset_id: str) -> Response:
    if not dataset_service.delete_dataset(dataset_id):
        raise HTTPException(status_code=404, detail="Dataset not found")
    return Response(status_code=204)
