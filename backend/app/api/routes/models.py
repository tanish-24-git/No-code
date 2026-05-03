"""Trained-model registry + HF Hub pull/push."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services import hf_service
from app.storage import store


router = APIRouter(prefix="/api/models", tags=["models"])


class ModelRecord(BaseModel):
    id: str
    kind: str = "trained"  # "trained" | "base"
    job_id: str | None = None
    repo_id: str | None = None
    local_path: str | None = None
    hf_repo_id: str | None = None
    is_pushed_to_hub: bool = False
    push_status: str | None = None
    base_model: str | None = None
    training_method: str | None = None


class PullRequest(BaseModel):
    repo_id: str


class PushRequest(BaseModel):
    repo_id: str  # destination repo, e.g. "user/my-tuned-model"


@router.get("", response_model=list[ModelRecord])
def list_models() -> list[ModelRecord]:
    out: list[ModelRecord] = []
    for raw in store.list_all("models"):
        try:
            out.append(ModelRecord(**raw))
        except Exception:
            continue
    return out


@router.get("/{model_id}", response_model=ModelRecord)
def get_model(model_id: str) -> ModelRecord:
    raw = store.read("models", model_id)
    if not raw:
        raise HTTPException(status_code=404, detail="Model not found")
    return ModelRecord(**raw)


@router.post("/pull")
def pull_base_model(payload: PullRequest) -> dict:
    return hf_service.start_pull(payload.repo_id)


@router.get("/pull-status/{repo_id:path}")
def pull_status(repo_id: str) -> dict:
    return hf_service.pull_status(repo_id)


@router.post("/{model_id}/push-hub")
def push_to_hub(model_id: str, payload: PushRequest) -> dict:
    try:
        return hf_service.start_push(model_id, payload.repo_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("/{model_id}/push-status")
def push_status(model_id: str) -> dict:
    return hf_service.push_status(model_id)
