"""Dataset management routes. The upload endpoint is the canonical entry
point of the agent runtime: a successful upload starts an AgentSession and
publishes DatasetUploaded on the bus, which kicks off the whole pipeline.

v4.0 — Universal intake: supports both single files and multi-file
directory uploads. All file types accepted (extension-agnostic).
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Response, UploadFile
from pydantic import BaseModel

from app.api.schemas.dataset import DatasetSchema
from app.events.bus import get_bus
from app.events.types import AgentEvent
from app.services import dataset_service, session_service
from app.tools.llm import ping_llm
from app.utils.config import settings


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
    """Upload any file (extension-agnostic). The Universal Intake Engine
    sniffs the content to determine type and parses accordingly."""
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
    #
    # We do NOT defer this even though it adds a few seconds to the upload
    # response — the orchestrator branches on probe.mode at SessionStarted,
    # so a pending probe would lock the whole session into deterministic
    # mode. The ProviderGate in providers.py now serializes ping_llm
    # together with the subsequent intake/loop calls, so this synchronous
    # probe no longer participates in a burst-RPM exhaustion pattern.
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


@router.post("/upload-directory", response_model=DatasetUploadResponse, status_code=201)
async def upload_directory(files: list[UploadFile] = File(...)) -> DatasetUploadResponse:
    """Upload multiple files (simulating a directory upload). The Universal
    Intake Engine recursively scans, sniffs types, and aggregates all files
    into a single dataset — structured files are row-merged, raw docs are
    concatenated for cross-file synthesis."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    # Write all uploaded files to a temporary directory
    tmp_dir = Path(tempfile.mkdtemp(dir=settings.upload_dir, prefix="dir_"))
    try:
        for f in files:
            fname = f.filename or "unnamed"
            # Preserve subdirectory structure from filenames like "subdir/file.txt"
            target = tmp_dir / fname
            target.parent.mkdir(parents=True, exist_ok=True)
            content = await f.read()
            if content:
                target.write_bytes(content)

        dataset = dataset_service.save_uploaded_directory(tmp_dir)
    except ValueError as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(e)) from e
    finally:
        # Clean up temp dir (data has been written to uploads/)
        shutil.rmtree(tmp_dir, ignore_errors=True)

    probe = await ping_llm()
    session = session_service.start_for_dataset(dataset.id)
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
