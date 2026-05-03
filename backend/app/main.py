"""FineTune Studio API. No Redis, no MinIO, no database, no Docker.

Just FastAPI + JSON files on disk. The LLM provider is configured via
.env (LLM_PROVIDER / LLM_API_KEY / LLM_MODEL / LLM_BASE_URL) or via the
Settings page in the UI; the UI takes precedence when both are set.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import (
    agent,
    datasets,
    health,
    inference,
    jobs,
    models,
    pipelines,
    settings as settings_routes,
)
from app.utils.config import settings


logging.basicConfig(level=settings.log_level)
log = logging.getLogger("finetune-studio")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Resolved config at boot. We log a single human-readable summary so it
    # is obvious whether the user still needs to configure anything.
    from app.api.routes.settings import get_hf_token_resolved, get_llm_config

    cfg = get_llm_config()
    hf_token, hf_source = get_hf_token_resolved()

    log.info("FineTune Studio API starting")
    log.info("data_dir=%s  upload_dir=%s  models_dir=%s",
             settings.data_dir, settings.upload_dir, settings.models_dir)

    if cfg.provider:
        log.info(
            "LLM provider: %s  model=%s  base_url=%s  source=%s  api_key=%s",
            cfg.provider, cfg.model, cfg.base_url or "(default)",
            cfg.source, "set" if cfg.api_key else "missing",
        )
    else:
        log.warning(
            "LLM is NOT configured. Either set LLM_PROVIDER, LLM_API_KEY, and "
            "LLM_MODEL in backend/.env, or open the UI and visit Settings."
        )

    if hf_token:
        log.info("Hugging Face token: set (source=%s)", hf_source)
    else:
        log.info("Hugging Face token: not set (only needed for model pull/push)")

    yield


app = FastAPI(
    title="FineTune Studio API",
    version="2.0.0",
    description="Local-first, open-source LLM fine-tuning + inference copilot.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(settings_routes.router)
app.include_router(datasets.router)
app.include_router(pipelines.router)
app.include_router(jobs.router)
app.include_router(models.router)
app.include_router(inference.router)
app.include_router(agent.router)


@app.get("/")
def root() -> dict:
    return {
        "name": "FineTune Studio API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.api_host, port=settings.api_port, reload=settings.debug)
