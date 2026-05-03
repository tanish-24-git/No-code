"""User inference endpoints. Listed by /api/inferences; the agent can read
this list as a tool and recommend metrics tuned to each one."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response

from app.api.schemas.inference import (
    GenerateRequest,
    GenerateResponse,
    InferenceCreate,
    InferenceProbe,
    InferenceRecord,
    InferenceUpdate,
)
from app.services import inference_service


router = APIRouter(prefix="/api/inferences", tags=["inferences"])


@router.post("", response_model=InferenceRecord, status_code=201)
def create(payload: InferenceCreate) -> InferenceRecord:
    return inference_service.create(payload)


@router.get("", response_model=list[InferenceRecord])
def list_endpoints() -> list[InferenceRecord]:
    return inference_service.list_all()


@router.get("/{inference_id}", response_model=InferenceRecord)
def get_endpoint(inference_id: str) -> InferenceRecord:
    rec = inference_service.get(inference_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Inference endpoint not found")
    return rec


@router.put("/{inference_id}", response_model=InferenceRecord)
def update_endpoint(inference_id: str, payload: InferenceUpdate) -> InferenceRecord:
    rec = inference_service.update(inference_id, payload)
    if not rec:
        raise HTTPException(status_code=404, detail="Inference endpoint not found")
    return rec


@router.delete("/{inference_id}", status_code=204, response_class=Response)
def delete_endpoint(inference_id: str) -> Response:
    if not inference_service.delete(inference_id):
        raise HTTPException(status_code=404, detail="Inference endpoint not found")
    return Response(status_code=204)


@router.post("/{inference_id}/probe", response_model=InferenceProbe)
async def probe_endpoint(inference_id: str) -> InferenceProbe:
    return await inference_service.probe(inference_id)


@router.post("/generate", response_model=GenerateResponse)
async def generate(payload: GenerateRequest) -> GenerateResponse:
    try:
        return await inference_service.generate(
            payload.inference_id,
            payload.prompt,
            model=payload.model,
            max_tokens=payload.max_tokens,
            temperature=payload.temperature,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Upstream error: {e}") from e
