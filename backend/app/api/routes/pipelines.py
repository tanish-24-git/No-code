from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response

from app.api.schemas.pipeline import PipelineCreate, PipelineRecord, PipelineUpdate
from app.services import pipeline_service


router = APIRouter(prefix="/api/pipelines", tags=["pipelines"])


@router.post("", response_model=PipelineRecord, status_code=201)
def create_pipeline(payload: PipelineCreate) -> PipelineRecord:
    return pipeline_service.create(payload)


@router.get("", response_model=list[PipelineRecord])
def list_pipelines() -> list[PipelineRecord]:
    return pipeline_service.list_all()


@router.get("/{pipeline_id}", response_model=PipelineRecord)
def get_pipeline(pipeline_id: str) -> PipelineRecord:
    rec = pipeline_service.get(pipeline_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    return rec


@router.put("/{pipeline_id}", response_model=PipelineRecord)
def update_pipeline(pipeline_id: str, payload: PipelineUpdate) -> PipelineRecord:
    rec = pipeline_service.update(pipeline_id, payload)
    if not rec:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    return rec


@router.delete("/{pipeline_id}", status_code=204, response_class=Response)
def delete_pipeline(pipeline_id: str) -> Response:
    if not pipeline_service.delete(pipeline_id):
        raise HTTPException(status_code=404, detail="Pipeline not found")
    return Response(status_code=204)
