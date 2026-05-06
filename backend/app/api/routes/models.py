"""Trained-model registry + HF Hub pull/push + interactive test endpoint."""
from __future__ import annotations

import asyncio
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.services import hf_service, pipeline_service
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


# ─── Interactive test playground ────────────────────────────────────────────
#
# This stub does not load the trained adapter into a real inference runtime.
# It proxies the prompt through the configured LLM provider with a role-play
# system prompt that reflects what the model was trained to do, so the user
# gets a usable preview without needing Ollama / vLLM / HF Inference set up.
#
# When you wire up real inference (transformers + peft, or a side-car
# inference server), swap the implementation of `_stream_test` and keep this
# route shape — the frontend doesn't need to change.

class TestRequest(BaseModel):
    prompt: str
    system_prompt: str | None = None
    temperature: float = 0.7
    max_tokens: int = 512


@router.post("/{model_id}/test")
async def test_model(model_id: str, payload: TestRequest) -> StreamingResponse:
    raw = store.read("models", model_id)
    if not raw:
        raise HTTPException(status_code=404, detail="Model not found")

    # Try to recover task type / base model context from the originating job.
    pipeline_ctx = ""
    job_id = raw.get("job_id")
    if job_id:
        job = store.read("jobs", job_id) or {}
        pid = job.get("pipeline_id")
        if pid:
            p = pipeline_service.get(pid)
            if p:
                cfg = p.config
                pipeline_ctx = (
                    f"Trained from base model: {cfg.base_model}. "
                    f"Task: {cfg.task_type}. Output style: {cfg.output_type}. "
                    f"Domain: {cfg.domain}."
                )

    system = (payload.system_prompt or "").strip() or (
        "You are a fine-tuned assistant. " + pipeline_ctx +
        " Reply in the style and format you were tuned for. Keep answers grounded."
    )

    # Build a one-shot chat through the existing provider streaming path.
    from app.api.schemas.agent import ChatMessage
    from app.agents.providers import stream_chat
    from app.api.routes.settings import get_llm_config

    cfg = get_llm_config()
    if not cfg.provider or not cfg.model:
        raise HTTPException(
            status_code=400,
            detail="LLM provider is not configured. Set it on the Settings page first.",
        )

    messages = [{"role": "user", "content": payload.prompt}]

    async def gen() -> AsyncIterator[str]:
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=256)

        def producer() -> None:
            try:
                for chunk in stream_chat(
                    provider=cfg.provider,
                    api_key=cfg.api_key,
                    model=cfg.model,
                    base_url=cfg.base_url,
                    messages=messages,
                    extra_system=system,
                ):
                    asyncio.run_coroutine_threadsafe(queue.put(chunk), loop)
            except Exception as e:
                asyncio.run_coroutine_threadsafe(queue.put(f"\n[error: {e}]\n"), loop)
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)

        loop.run_in_executor(None, producer)
        while True:
            chunk = await queue.get()
            if chunk is None:
                yield "data: [DONE]\n\n"
                return
            for line in chunk.split("\n"):
                yield f"data: {line}\n"
            yield "\n"

    return StreamingResponse(gen(), media_type="text/event-stream")
