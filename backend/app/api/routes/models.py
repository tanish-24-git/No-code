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
# Two execution paths:
#
#   1. Real adapter inference (preferred). When the trained adapter directory
#      exists on disk and `transformers` + `peft` are importable, we load the
#      base model, attach the adapter, and stream tokens from generate(). The
#      reply is genuinely from the user's fine-tuned weights.
#
#   2. Provider proxy fallback. If real inference is unavailable (missing
#      deps, missing adapter dir, OOM at load time), we proxy the prompt
#      through the configured LLM provider with a role-play system prompt.
#      The response header `X-Inference-Source` tells the UI which path ran.
#
# Streamed as SSE so the existing frontend does not change.

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

    # Recover task type / base model context from the originating job.
    pipeline_ctx = ""
    base_model: str | None = None
    job_id = raw.get("job_id")
    if job_id:
        job = store.read("jobs", job_id) or {}
        pid = job.get("pipeline_id")
        if pid:
            p = pipeline_service.get(pid)
            if p:
                cfg = p.config
                base_model = cfg.base_model
                pipeline_ctx = (
                    f"Trained from base model: {cfg.base_model}. "
                    f"Task: {cfg.task_type}. Output style: {cfg.output_type}. "
                    f"Domain: {cfg.domain}."
                )

    adapter_dir = raw.get("local_path")

    system = (payload.system_prompt or "").strip() or (
        "You are a fine-tuned assistant. " + pipeline_ctx +
        " Reply in the style and format you were tuned for. Keep answers grounded."
    )

    # Try real adapter inference first.
    real_iter = _try_real_inference(
        adapter_dir=adapter_dir,
        base_model=base_model,
        prompt=payload.prompt,
        system=system,
        temperature=float(payload.temperature),
        max_tokens=int(payload.max_tokens),
    )
    source = "fine_tuned_adapter" if real_iter is not None else "provider_proxy"

    if real_iter is None:
        # Fall back to the configured provider.
        from app.api.routes.settings import get_llm_config
        from app.agents.providers import stream_chat

        cfg = get_llm_config()
        if not cfg.provider or not cfg.model:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Trained adapter not loadable here and no LLM provider configured. "
                    "Either install transformers + peft + torch, or set a provider in Settings."
                ),
            )

        def proxy_iter() -> AsyncIterator[str]:
            return _stream_provider(cfg, payload.prompt, system)

        gen = proxy_iter()
    else:
        gen = real_iter

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Inference-Source": source,
    }
    return StreamingResponse(gen, media_type="text/event-stream", headers=headers)


def _try_real_inference(
    *,
    adapter_dir: str | None,
    base_model: str | None,
    prompt: str,
    system: str,
    temperature: float,
    max_tokens: int,
) -> AsyncIterator[str] | None:
    """Return an async iterator that streams tokens from the trained adapter,
    or None if real inference is not available right now."""
    if not adapter_dir:
        return None
    from pathlib import Path

    adapter_path = Path(adapter_dir)
    if not adapter_path.exists():
        return None

    # Defensive imports - all of these can fail on a CPU-only laptop.
    try:
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer  # type: ignore
    except Exception:
        return None

    # Optional peft adapter wrap. If the adapter is a merged checkpoint, the
    # AutoModel call below is enough; otherwise we attach the adapter on top
    # of the base model.
    try:
        from peft import PeftModel  # type: ignore
        peft_available = True
    except Exception:
        peft_available = False

    async def gen() -> AsyncIterator[str]:
        try:
            tok_path = str(adapter_path) if (adapter_path / "tokenizer.json").exists() else (base_model or str(adapter_path))
            tok = AutoTokenizer.from_pretrained(tok_path)
            if tok.pad_token_id is None and tok.eos_token_id is not None:
                tok.pad_token_id = tok.eos_token_id

            # Heuristic: if config.json has model weights co-located, load the
            # adapter dir directly; otherwise load base + attach adapter.
            has_weights = any((adapter_path / f).exists() for f in ("model.safetensors", "pytorch_model.bin"))
            if has_weights:
                model = AutoModelForCausalLM.from_pretrained(
                    str(adapter_path), torch_dtype=torch.float32, low_cpu_mem_usage=True,
                )
            elif base_model and peft_available:
                base = AutoModelForCausalLM.from_pretrained(
                    base_model, torch_dtype=torch.float32, low_cpu_mem_usage=True,
                )
                model = PeftModel.from_pretrained(base, str(adapter_path))
            else:
                yield "data: [error] adapter directory does not contain weights and peft is not installed\n\n"
                yield "data: [DONE]\n\n"
                return

            model.eval()

            # Build chat-template input where the tokenizer supports it; fall
            # back to a vanilla prompt otherwise.
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            try:
                input_ids = tok.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
            except Exception:
                input_ids = tok(f"{system}\n\nUser: {prompt}\nAssistant:", return_tensors="pt").input_ids

            streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
            gen_kwargs = dict(
                input_ids=input_ids,
                streamer=streamer,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-3),
                top_p=0.95,
                pad_token_id=tok.pad_token_id,
            )

            import threading
            thread = threading.Thread(target=model.generate, kwargs=gen_kwargs, daemon=True)
            thread.start()

            for piece in streamer:
                if not piece:
                    continue
                # SSE expects each newline-separated chunk on its own data: line.
                for line in piece.split("\n"):
                    yield f"data: {line}\n"
                yield "\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: [error] {type(e).__name__}: {e}\n\n"
            yield "data: [DONE]\n\n"

    return gen()


async def _stream_provider(cfg, prompt: str, system: str) -> AsyncIterator[str]:
    """Existing provider-proxy path, lifted to its own coroutine."""
    from app.agents.providers import stream_chat

    loop = asyncio.get_event_loop()
    queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=256)

    def producer() -> None:
        try:
            for chunk in stream_chat(
                provider=cfg.provider,
                api_key=cfg.api_key,
                model=cfg.model,
                base_url=cfg.base_url,
                messages=[{"role": "user", "content": prompt}],
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
