"""Job orchestration + SSE log streaming."""
from __future__ import annotations

import asyncio
from typing import AsyncIterator

from fastapi import APIRouter, BackgroundTasks, HTTPException, Response
from fastapi.responses import StreamingResponse

from app.api.schemas.job import JobRecord, JobStartRequest, JobStartResponse
from app.services import job_service


router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.post("/start", response_model=JobStartResponse, status_code=201)
def start_job(payload: JobStartRequest, background: BackgroundTasks) -> JobStartResponse:
    try:
        job = job_service.create(payload.pipeline_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    background.add_task(job_service.run_in_background, job.id)
    return JobStartResponse(job_id=job.id, status="queued")


@router.get("", response_model=list[JobRecord])
def list_jobs() -> list[JobRecord]:
    return job_service.list_all()


@router.get("/{job_id}", response_model=JobRecord)
def get_job(job_id: str) -> JobRecord:
    job = job_service.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.post("/{job_id}/stop")
def stop_job(job_id: str) -> dict:
    if not job_service.request_stop(job_id):
        raise HTTPException(status_code=404, detail="Job not running or not found")
    return {"status": "stop_requested"}


@router.delete("/{job_id}", status_code=204, response_class=Response)
def delete_job(job_id: str) -> Response:
    try:
        deleted = job_service.delete(job_id)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    if not deleted:
        raise HTTPException(status_code=404, detail="Job not found")
    return Response(status_code=204)


@router.get("/{job_id}/logs")
async def stream_logs(job_id: str) -> StreamingResponse:
    job = job_service.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    async def gen() -> AsyncIterator[str]:
        # Replay backlog first.
        for line in job_service.get_buffered_logs(job_id):
            yield f"data: {line}\n\n"
        if job_service.is_terminal(job_id):
            yield "data: [DONE]\n\n"
            return

        q = job_service.subscribe(job_id)
        try:
            while True:
                try:
                    line = await asyncio.wait_for(q.get(), timeout=15.0)
                    yield f"data: {line}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                if job_service.is_terminal(job_id) and q.empty():
                    yield "data: [DONE]\n\n"
                    return
        finally:
            job_service.unsubscribe(job_id, q)

    return StreamingResponse(gen(), media_type="text/event-stream")
