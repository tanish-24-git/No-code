"""User-registered inference endpoints. We can probe Ollama, OpenAI-compatible
servers, and the HF Inference API. Keys are encrypted at rest."""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import httpx

from app.api.schemas.inference import (
    GenerateResponse,
    InferenceCreate,
    InferenceProbe,
    InferenceRecord,
    InferenceUpdate,
)
from app.storage import store
from app.utils import crypto


def _to_record(raw: dict[str, Any]) -> InferenceRecord:
    api_key = crypto.decrypt(raw.get("api_key_enc", "")) if raw.get("api_key_enc") else ""
    return InferenceRecord(
        id=raw["id"],
        name=raw["name"],
        kind=raw["kind"],
        base_url=raw["base_url"],
        api_key_masked=crypto.mask(api_key) or None,
        default_model=raw.get("default_model"),
        notes=raw.get("notes"),
        last_probe=InferenceProbe(**raw["last_probe"]) if raw.get("last_probe") else None,
        suggested_metrics=raw.get("suggested_metrics", {}),
        created_at=raw["created_at"],
        updated_at=raw["updated_at"],
    )


def _decrypt_key(record_id: str) -> str:
    raw = store.read("inferences", record_id)
    if not raw:
        return ""
    return crypto.decrypt(raw.get("api_key_enc", "")) if raw.get("api_key_enc") else ""


def create(payload: InferenceCreate) -> InferenceRecord:
    record_id = store.new_id()
    now = datetime.now(timezone.utc)
    raw: dict[str, Any] = {
        "id": record_id,
        "name": payload.name,
        "kind": payload.kind,
        "base_url": payload.base_url.rstrip("/"),
        "api_key_enc": crypto.encrypt(payload.api_key) if payload.api_key else "",
        "default_model": payload.default_model,
        "notes": payload.notes,
        "last_probe": None,
        "suggested_metrics": {},
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
    }
    store.write("inferences", record_id, raw)
    return _to_record(raw)


def update(record_id: str, payload: InferenceUpdate) -> InferenceRecord | None:
    raw = store.read("inferences", record_id)
    if not raw:
        return None
    data = payload.model_dump(exclude_none=True)
    if "api_key" in data:
        raw["api_key_enc"] = crypto.encrypt(data.pop("api_key")) if data["api_key"] else ""
    raw.update({k: v for k, v in data.items() if v is not None})
    raw["updated_at"] = datetime.now(timezone.utc).isoformat()
    store.write("inferences", record_id, raw)
    return _to_record(raw)


def get(record_id: str) -> InferenceRecord | None:
    raw = store.read("inferences", record_id)
    return _to_record(raw) if raw else None


def list_all() -> list[InferenceRecord]:
    out: list[InferenceRecord] = []
    for raw in store.list_all("inferences"):
        try:
            out.append(_to_record(raw))
        except Exception:
            continue
    out.sort(key=lambda r: r.created_at, reverse=True)
    return out


def delete(record_id: str) -> bool:
    return store.delete("inferences", record_id)


# ─── Probing ────────────────────────────────────────────────────────────────

async def probe(record_id: str) -> InferenceProbe:
    raw = store.read("inferences", record_id)
    if not raw:
        return InferenceProbe(reachable=False, detail="Not found")

    base_url = raw["base_url"].rstrip("/")
    kind = raw["kind"]
    api_key = crypto.decrypt(raw.get("api_key_enc", "")) if raw.get("api_key_enc") else ""

    started = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            if kind == "ollama":
                r = await client.get(f"{base_url}/api/tags")
                latency = (time.perf_counter() - started) * 1000
                if r.status_code != 200:
                    return _save_probe(raw, InferenceProbe(reachable=False, latency_ms=latency, detail=f"HTTP {r.status_code}"))
                models = [m.get("name") for m in r.json().get("models", []) if m.get("name")]
                return _save_probe(raw, InferenceProbe(reachable=True, latency_ms=latency, models=models))

            if kind == "openai_compat":
                headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
                r = await client.get(f"{base_url}/v1/models", headers=headers)
                latency = (time.perf_counter() - started) * 1000
                if r.status_code != 200:
                    return _save_probe(raw, InferenceProbe(reachable=False, latency_ms=latency, detail=f"HTTP {r.status_code}"))
                data = r.json().get("data", [])
                models = [m.get("id") for m in data if m.get("id")]
                return _save_probe(raw, InferenceProbe(reachable=True, latency_ms=latency, models=models))

            if kind == "huggingface_inference":
                # HF inference doesn't have a list endpoint per se; just hit the root.
                headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
                r = await client.get(f"{base_url}/", headers=headers)
                latency = (time.perf_counter() - started) * 1000
                models = [raw.get("default_model")] if raw.get("default_model") else []
                return _save_probe(
                    raw,
                    InferenceProbe(reachable=r.status_code < 500, latency_ms=latency, models=[m for m in models if m], detail=f"HTTP {r.status_code}"),
                )

            if kind == "anthropic":
                headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
                r = await client.get(f"{base_url}/v1/models", headers=headers)
                latency = (time.perf_counter() - started) * 1000
                if r.status_code != 200:
                    return _save_probe(raw, InferenceProbe(reachable=False, latency_ms=latency, detail=f"HTTP {r.status_code}"))
                data = r.json().get("data", [])
                models = [m.get("id") for m in data if m.get("id")]
                return _save_probe(raw, InferenceProbe(reachable=True, latency_ms=latency, models=models))

            return _save_probe(raw, InferenceProbe(reachable=False, detail=f"Unknown kind: {kind}"))
    except Exception as e:
        latency = (time.perf_counter() - started) * 1000
        return _save_probe(raw, InferenceProbe(reachable=False, latency_ms=latency, detail=str(e)))


def _save_probe(raw: dict[str, Any], probe: InferenceProbe) -> InferenceProbe:
    raw["last_probe"] = probe.model_dump(mode="json")
    raw["updated_at"] = datetime.now(timezone.utc).isoformat()
    store.write("inferences", raw["id"], raw)
    return probe


# ─── Generate (test prompt against an endpoint) ────────────────────────────

async def generate(record_id: str, prompt: str, *, model: str | None, max_tokens: int, temperature: float) -> GenerateResponse:
    raw = store.read("inferences", record_id)
    if not raw:
        raise ValueError("Inference endpoint not found")
    base_url = raw["base_url"].rstrip("/")
    kind = raw["kind"]
    api_key = crypto.decrypt(raw.get("api_key_enc", "")) if raw.get("api_key_enc") else ""
    use_model = model or raw.get("default_model")
    if not use_model:
        raise ValueError("No model specified and no default model set on this endpoint")

    started = time.perf_counter()
    async with httpx.AsyncClient(timeout=120.0) as client:
        if kind == "ollama":
            r = await client.post(
                f"{base_url}/api/generate",
                json={"model": use_model, "prompt": prompt, "stream": False, "options": {"num_predict": max_tokens, "temperature": temperature}},
            )
            r.raise_for_status()
            text = r.json().get("response", "")
        elif kind == "openai_compat":
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
            r = await client.post(
                f"{base_url}/v1/chat/completions",
                headers=headers,
                json={
                    "model": use_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
            )
            r.raise_for_status()
            data = r.json()
            text = data["choices"][0]["message"]["content"]
        elif kind == "huggingface_inference":
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
            r = await client.post(
                f"{base_url}/models/{use_model}",
                headers=headers,
                json={"inputs": prompt, "parameters": {"max_new_tokens": max_tokens, "temperature": temperature}},
            )
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list) and data and isinstance(data[0], dict):
                text = data[0].get("generated_text", "")
            else:
                text = str(data)
        elif kind == "anthropic":
            headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
            r = await client.post(
                f"{base_url}/v1/messages",
                headers=headers,
                json={
                    "model": use_model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            r.raise_for_status()
            data = r.json()
            text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        else:
            raise ValueError(f"Unknown kind: {kind}")

    latency = (time.perf_counter() - started) * 1000
    return GenerateResponse(text=text, model=use_model, latency_ms=latency)
