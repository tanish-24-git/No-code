"""Health endpoint. Returns hardware info and a config-state summary so the
frontend can decide whether to show its first-run setup wizard."""
from __future__ import annotations

from fastapi import APIRouter

from app.api.routes.settings import get_hf_token_resolved, get_llm_config
from app.utils.hardware import detect_hardware


router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    cfg = get_llm_config()
    hf_token, hf_source = get_hf_token_resolved()
    return {
        "status": "ok",
        "version": "2.0.0",
        "hardware": detect_hardware(),
        "llm": {
            "configured": bool(cfg.provider and cfg.model and (cfg.api_key or cfg.provider == "openai")),
            "provider": cfg.provider,
            "model": cfg.model,
            "base_url": cfg.base_url or None,
            "source": cfg.source,
        },
        "hf": {
            "configured": bool(hf_token),
            "source": hf_source,
        },
    }
