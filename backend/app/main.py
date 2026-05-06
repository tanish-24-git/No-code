"""FineTune Studio API.

Two surfaces live here:
    1. The legacy stateless chat endpoint at /api/agent/chat (kept for the
       existing UI; useful as an "ask" channel).
    2. The new session-based, event-driven, multi-agent runtime at
       /api/sessions/* — auto-triggered by /api/datasets/upload.

Both share the same provider config and JSON-on-disk store.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.agents.wiring import register_agents
from app.api.routes import (
    agent,
    datasets,
    health,
    jobs,
    models,
    pipelines,
    sessions,
    settings as settings_routes,
)
from app.events.bus import get_bus
from app.utils.config import settings


logging.basicConfig(level=settings.log_level)
log = logging.getLogger("finetune-studio")


@asynccontextmanager
async def lifespan(_app: FastAPI):
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

    # Bind the running event loop to the bus and register agents. This needs
    # to happen here (not at import time) because background threads will
    # publish via run_coroutine_threadsafe.
    bus = get_bus()
    bus.bind_loop(asyncio.get_running_loop())
    register_agents(bus)
    # Importing app.tools triggers tool registration as a side effect.
    import app.tools  # noqa: F401
    log.info("agent runtime ready")

    yield


app = FastAPI(
    title="FineTune Studio API",
    version="3.0.0",
    description="Local-first, open-source autonomous fine-tuning agent platform.",
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
app.include_router(agent.router)
app.include_router(sessions.router)
app.include_router(sessions.tools_router)


@app.get("/")
def root() -> dict:
    return {
        "name": "FineTune Studio API",
        "version": "3.0.0",
        "docs": "/docs",
        "health": "/health",
        "sessions": "/api/sessions",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.api_host, port=settings.api_port, reload=settings.debug)
