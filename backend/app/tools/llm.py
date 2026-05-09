"""LLM connectivity tools.

A small "say ok" probe that runs before every session starts. The agents
behave very differently depending on whether the configured provider is
actually reachable:

    full agent mode      provider replies within the timeout - the
                         orchestrator delegates planning, model search,
                         and rationale generation to the LLM.

    deterministic mode   provider missing, unreachable, or rate-limited -
                         the agents fall back to their hand-coded scoring
                         heuristics so the user still gets a result.

Either way, the user is told which mode they are in. No silent fallback.
"""
from __future__ import annotations

import time
from typing import Any

from app.tools.registry import ToolContext, tool


_PING_PROMPT = "Reply with the single word 'ok'. Nothing else."
_PING_TIMEOUT_S = 6.0


@tool(
    name="llm.ping",
    description=(
        "Send a minimal one-token probe to the configured LLM provider. "
        "Returns {ok, mode, latency_ms, provider, model, detail}. Never raises."
    ),
    input_schema={"type": "object", "properties": {}},
)
async def llm_ping(_args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    return await ping_llm()


async def ping_llm() -> dict[str, Any]:
    """Public helper used by the upload route before starting a session."""
    from app.api.routes.settings import get_llm_config
    from app.agents.providers import stream_chat

    cfg = get_llm_config()
    if not cfg.provider or not cfg.model:
        return {
            "ok": False,
            "mode": "deterministic",
            "provider": None,
            "model": None,
            "latency_ms": 0,
            "detail": "no LLM provider configured - running in deterministic mode",
        }

    started = time.perf_counter()
    chunks: list[str] = []
    try:
        # We run the synchronous provider stream in this thread because the
        # producer is short-lived and the call is bounded by the timeout below.
        deadline = started + _PING_TIMEOUT_S
        for chunk in stream_chat(
            provider=cfg.provider,
            api_key=cfg.api_key,
            model=cfg.model,
            base_url=cfg.base_url,
            messages=[{"role": "user", "content": _PING_PROMPT}],
            extra_system="You are a connectivity probe. Reply with one short word.",
        ):
            chunks.append(chunk)
            if time.perf_counter() >= deadline:
                break
            # A short reply is enough; bail early once we have content.
            if sum(len(c) for c in chunks) > 32:
                break
    except Exception as e:
        ms = round((time.perf_counter() - started) * 1000, 1)
        return {
            "ok": False,
            "mode": "deterministic",
            "provider": cfg.provider,
            "model": cfg.model,
            "latency_ms": ms,
            "detail": f"{type(e).__name__}: {str(e)[:200]}",
        }

    text = "".join(chunks).strip()
    ms = round((time.perf_counter() - started) * 1000, 1)
    if not text:
        return {
            "ok": False,
            "mode": "deterministic",
            "provider": cfg.provider,
            "model": cfg.model,
            "latency_ms": ms,
            "detail": "provider replied with no text within timeout",
        }
    return {
        "ok": True,
        "mode": "full_agent",
        "provider": cfg.provider,
        "model": cfg.model,
        "latency_ms": ms,
        "detail": text[:64],
    }
