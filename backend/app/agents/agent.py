"""High-level agent entry point. Reads provider config from settings, builds
the system prompt with any active context, and delegates to the right
provider's streaming function."""
from __future__ import annotations

from typing import Iterator

from app.agents.providers import stream_chat
from app.api.schemas.agent import ChatMessage


def _augment_system(pipeline_id: str | None, dataset_id: str | None) -> str:
    parts: list[str] = []
    if pipeline_id:
        parts.append(f"The user is currently on pipeline `{pipeline_id}`.")
    if dataset_id:
        parts.append(f"The user is referring to dataset `{dataset_id}`.")
    return "\n".join(parts)


def run_chat(
    messages: list[ChatMessage],
    *,
    pipeline_id: str | None = None,
    dataset_id: str | None = None,
) -> Iterator[str]:
    # Imported lazily so cold imports of this module don't read settings.
    from app.api.routes.settings import get_llm_config

    from app.agents.registry import get_spec

    cfg = get_llm_config()
    if not cfg.provider or not cfg.model:
        raise ValueError(
            "LLM is not configured. Set LLM_PROVIDER and LLM_MODEL in .env, "
            "or configure them in Settings."
        )
    spec = get_spec(cfg.provider)
    if spec and spec.needs_key and not cfg.api_key:
        raise ValueError(
            f"{spec.label} requires an API key. Set LLM_API_KEY in .env or in Settings."
        )

    history = [{"role": m.role, "content": m.content} for m in messages]
    yield from stream_chat(
        provider=cfg.provider,
        api_key=cfg.api_key,
        model=cfg.model,
        base_url=cfg.base_url,
        messages=history,
        extra_system=_augment_system(pipeline_id, dataset_id),
    )
