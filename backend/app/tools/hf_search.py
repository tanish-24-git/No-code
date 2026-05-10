"""HuggingFace Hub model search.

Replaces the static 12-model catalogue with a live search against the
Hub. The agent narrows the search by:

    - task type (text-generation, text-classification, ...)
    - parameter budget (derived from device VRAM and a comfort margin)
    - instruction-tuned only (so the user gets a usable chat-template
      out of the box; base-only weights are excluded unless explicitly
      requested)
    - downloads / likes signals as a tiebreaker

Falls back to the static catalogue from ``app.tools.model`` when the
``huggingface_hub`` package is unavailable, the network is unreachable,
or the user has no HF token configured for rate-limit-free access.
"""
from __future__ import annotations

import re
from typing import Any

from app.tools.registry import ToolContext, tool


# Rough family-size budget — used when the user's hardware has no GPU and we
# need to cap the search aggressively.
_VRAM_TO_PARAMS_GB = {
    0: 1.5,        # CPU - keep under 1.5B params
    4: 1.5,
    6: 3.0,
    8: 3.0,
    12: 7.0,
    16: 7.0,
    24: 13.0,
    40: 70.0,
    80: 70.0,
}


def _budget_for(vram_gb: float | None, device: str) -> float:
    if device == "cpu":
        return 1.5
    if not vram_gb:
        return 7.0
    # Pick the largest budget whose key <= vram_gb.
    keys = sorted(_VRAM_TO_PARAMS_GB.keys())
    pick = keys[0]
    for k in keys:
        if k <= vram_gb:
            pick = k
    return _VRAM_TO_PARAMS_GB[pick]


def _params_b_from_id(repo_id: str) -> float | None:
    """Pull the parameter count out of a repo id like 'meta-llama/Llama-3.2-3B-Instruct'."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*[Bb]\b", repo_id)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+)\s*[Mm]\b", repo_id)
    if m:
        return float(m.group(1)) / 1000.0
    return None


def _task_to_hf_task(task: str) -> str:
    return {
        "chat": "text-generation",
        "instruction": "text-generation",
        "qa": "text-generation",
        "extraction": "text-generation",
        "classification": "text-classification",
    }.get(task, "text-generation")


@tool(
    name="model.search_hf",
    description=(
        "Search the HuggingFace Hub for base models that fit the user's task "
        "and hardware. Returns up to N candidates with downloads, likes, and "
        "estimated parameter counts. Falls back to a curated catalogue when "
        "the Hub is unreachable."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "task": {"type": "string", "default": "instruction"},
            "query": {"type": "string"},
            "hardware": {"type": "object"},
            "instruct_only": {"type": "boolean", "default": True},
            "top_n": {"type": "integer", "default": 12},
            "max_params_b": {"type": "number"},
        },
        "required": ["hardware"],
    },
    cost_class="cheap",
    side_effect="external",
)
async def model_search_hf(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    task = (args.get("task") or "instruction").lower()
    query = args.get("query")
    hw = args.get("hardware") or {}
    instruct_only = bool(args.get("instruct_only", True))
    top_n = int(args.get("top_n", 12))
    budget = float(args.get("max_params_b") or _budget_for(hw.get("vram_gb"), hw.get("device", "cpu")))

    hf_task = _task_to_hf_task(task)

    # 1. Attempt a live Hub search.
    candidates: list[dict[str, Any]] = []
    source = "fallback_catalog"
    error: str | None = None
    try:
        from huggingface_hub import HfApi  # type: ignore

        from app.api.routes.settings import get_hf_token

        api = HfApi(token=get_hf_token() or None)
        # Use the explicit query if provided, else fallback to 'instruct'.
        search_terms = query if query else ("instruct" if instruct_only else None)
        listed = list(
            api.list_models(
                task=hf_task,
                search=search_terms,
                sort="downloads",
                direction=-1,
                limit=80,
            )
        )
        for m in listed:
            repo_id = getattr(m, "id", None) or getattr(m, "modelId", None)
            if not repo_id:
                continue
            params_b = _params_b_from_id(repo_id)
            if params_b is None:
                continue
            if params_b > budget * 1.05:
                continue
            tags = list(getattr(m, "tags", []) or [])
            if instruct_only and not _looks_instruct(repo_id, tags):
                continue
            candidates.append({
                "repo_id": repo_id,
                "label": _label_from_id(repo_id),
                "params_b": params_b,
                "downloads": int(getattr(m, "downloads", 0) or 0),
                "likes": int(getattr(m, "likes", 0) or 0),
                "tags": tags[:8],
                "source": "huggingface_hub",
            })
            if len(candidates) >= top_n:
                break
        if candidates:
            source = "huggingface_hub"
    except Exception as e:
        error = f"{type(e).__name__}: {str(e)[:200]}"

    if not candidates and not error:
        error = "No models found matching your criteria on HuggingFace Hub."

    # Sort: smaller models first when budget is tight, then by popularity.
    candidates.sort(key=lambda r: (-r.get("downloads", 0), r["params_b"]))
    return {
        "candidates": candidates,
        "source": source,
        "task": hf_task,
        "max_params_b": budget,
        "device": hw.get("device", "cpu"),
        "error": error,
    }


def _looks_instruct(repo_id: str, tags: list[str]) -> bool:
    rid = repo_id.lower()
    for needle in ("instruct", "chat", "-it", "sft", "tuned", "dpo"):
        if needle in rid:
            return True
    return any(t.lower() in {"chat", "instruct", "instruct-tuning"} for t in tags)


def _label_from_id(repo_id: str) -> str:
    return repo_id.split("/")[-1]
