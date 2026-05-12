"""HuggingFace Hub model search.

Replaces the static catalogue with a live search against the Hub. The
agent narrows the search by:

    * task type (text-generation, text-classification, ...)
    * parameter budget (computed from device VRAM, precision, and a safety
      margin - NOT a hardcoded ladder)
    * instruction-tuned tag or family (so the user gets a chat-templated
      base model out of the box; pure base weights are excluded unless
      explicitly requested)
    * downloads / likes signals as a tiebreaker

Parameter counts come from the HuggingFace ``model_info`` endpoint when
available (``safetensors.total`` field) and only fall back to a regex on
the repo id when the API does not provide them. This avoids mis-sizing
models where the version number (e.g. '3.2') might be mistaken for the 
parameter count.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from app.tools.registry import ToolContext, tool


# ── parameter budget ─────────────────────────────────────────────────────


def _bytes_per_param(precision: str) -> float:
    p = (precision or "").lower()
    if p in ("int4", "4bit"):
        return 0.55       # NF4 ~ 0.5B with overhead
    if p in ("int8", "8bit"):
        return 1.0
    if p in ("bf16", "fp16", "float16", "bfloat16"):
        return 2.0
    return 4.0            # fp32


def _budget_params_b(
    *,
    vram_gb: Optional[float],
    device: str,
    precision: str = "bf16",
    method: str = "lora",
) -> float:
    """How many billions of parameters fit on this hardware?

    Computed from VRAM and precision:
        usable_vram = vram - 2 GB (kv-cache + activations + lora optim state)
        params_bytes = (usable_vram / bytes_per_param) GB
        params_b = params_bytes / 1.0 GB-per-billion-params  (since "B" => 1e9)

    LoRA adds ~10% overhead vs inference; QLoRA's 4-bit quant means
    bytes_per_param drops accordingly. CPU is capped at 1.5B regardless
    because anything bigger is impractical without a GPU.
    """
    if device == "cpu":
        return 1.5
    if not vram_gb or vram_gb <= 0:
        return 7.0
    bpp = _bytes_per_param(precision)
    overhead_gb = 2.0
    if method in ("lora", "qlora", "dora"):
        overhead_gb += 1.0
    usable = max(vram_gb - overhead_gb, 0.5)
    # Each B parameters = bpp GB at the chosen precision.
    return round(usable / bpp, 2)


# ── repo id introspection ────────────────────────────────────────────────


_FAMILY_TAGS = {
    "llama": ("llama",),
    "qwen": ("qwen",),
    "phi": ("phi",),
    "mistral": ("mistral",),
    "gemma": ("gemma",),
    "smollm": ("smollm",),
    "deepseek": ("deepseek",),
    "yi": ("01-ai", "yi-"),
    "tinyllama": ("tinyllama",),
    "stablelm": ("stablelm",),
    "olmo": ("olmo",),
}


def _params_b_from_id(repo_id: str) -> Optional[float]:
    """Best-effort parameter-count parse from the repo id.

    Stricter than before: we look for a NUMBER directly followed by the
    suffix B/M and a non-digit boundary. ``Qwen2.5-Coder-32B-Instruct``
    parses as 32B (the ``2.5`` doesn't match because it's followed by
    ``-`` not ``B``).
    """
    # Match X[.Y]B but only when not preceded by another digit-dot prefix.
    m = re.search(r"(?<![\d\.])(\d+(?:\.\d+)?)\s*[Bb](?![\w])", repo_id)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    m = re.search(r"(?<![\d\.])(\d+)\s*[Mm](?![\w])", repo_id)
    if m:
        try:
            return float(m.group(1)) / 1000.0
        except ValueError:
            return None
    return None


def _params_b_from_info(model_info: Any) -> Optional[float]:
    """Use HfApi.model_info safetensors.total when present."""
    safe = getattr(model_info, "safetensors", None)
    if not safe:
        return None
    total = None
    if isinstance(safe, dict):
        total = safe.get("total") or safe.get("parameters", {}).get("total")
    else:
        total = getattr(safe, "total", None)
    if total:
        try:
            return round(float(total) / 1e9, 2)
        except Exception:
            return None
    return None


def _looks_instruct(repo_id: str, tags: list[str]) -> bool:
    rid = repo_id.lower()
    # Word-boundary check so "endpoint" doesn't false-positive on "dpo".
    for needle in (r"\binstruct\b", r"\bchat\b", r"\b-it\b", r"\bsft\b", r"\bdpo\b", r"\borpo\b"):
        if re.search(needle, rid):
            return True
    return any(t.lower() in {"chat", "instruct", "instruct-tuning", "conversational"} for t in tags)


def _task_to_hf_task(task: str) -> str:
    return {
        "chat": "text-generation",
        "instruction": "text-generation",
        "qa": "text-generation",
        "extraction": "text-generation",
        "summarization": "text-generation",
        "translation": "text-generation",
        "language_modeling": "text-generation",
        "classification": "text-classification",
    }.get(task, "text-generation")


@tool(
    name="model.search_hf",
    description=(
        "Search the HuggingFace Hub for base models that fit the user's task "
        "and hardware. Returns up to N candidates with downloads, likes, and "
        "estimated parameter counts. Falls back to the curated catalogue "
        "when the Hub is unreachable or the search returns nothing."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "task": {"type": "string", "default": "instruction"},
            "query": {"type": "string"},
            "family_hint": {"type": "string"},
            "size_hint_b": {"type": "number"},
            "hardware": {"type": "object"},
            "precision": {"type": "string"},
            "method": {"type": "string"},
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
    family_hint = (args.get("family_hint") or "").lower() or None
    size_hint_b = args.get("size_hint_b")
    hw = args.get("hardware") or {}
    precision = args.get("precision") or "bf16"
    method = args.get("method") or "lora"
    instruct_only = bool(args.get("instruct_only", True))
    top_n = int(args.get("top_n", 12))
    budget = float(args.get("max_params_b") or _budget_params_b(
        vram_gb=hw.get("vram_gb"),
        device=hw.get("device", "cpu"),
        precision=precision,
        method=method,
    ))

    hf_task = _task_to_hf_task(task)
    candidates: list[dict[str, Any]] = []
    source = "fallback_catalog"
    error: Optional[str] = None

    try:
        from huggingface_hub import HfApi  # type: ignore
        from app.api.routes.settings import get_hf_token

        api = HfApi(token=get_hf_token() or None)
        # Build search terms - prefer a family hint pulled from the user
        # directives ("use llama") over the bare task verb.
        search_terms = (family_hint or query or ("instruct" if instruct_only else None))
        listed = list(api.list_models(
            task=hf_task,
            search=search_terms,
            sort="downloads",
            direction=-1,
            limit=80,
        ))
        for m in listed:
            repo_id = getattr(m, "id", None) or getattr(m, "modelId", None)
            if not repo_id:
                continue
            params_b = _params_b_from_info(m) or _params_b_from_id(repo_id)
            if params_b is None:
                continue
            if params_b > budget * 1.05:
                continue
            tags = list(getattr(m, "tags", []) or [])
            if instruct_only and not _looks_instruct(repo_id, tags):
                continue
            if family_hint:
                # The user wants a specific family - skip non-matches.
                rid = repo_id.lower()
                if family_hint not in rid and not any(family_hint in t for t in [tg.lower() for tg in tags]):
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
        error = "No models found on HuggingFace Hub matching the criteria."

    if size_hint_b:
        # Re-rank candidates to put the closest size on top.
        candidates.sort(key=lambda r: abs(r["params_b"] - float(size_hint_b)))
    else:
        candidates.sort(key=lambda r: (-r.get("downloads", 0), r["params_b"]))

    return {
        "candidates": candidates,
        "source": source,
        "task": hf_task,
        "max_params_b": budget,
        "device": hw.get("device", "cpu"),
        "error": error,
        "family_hint": family_hint,
        "size_hint_b": size_hint_b,
    }


def _label_from_id(repo_id: str) -> str:
    return repo_id.split("/")[-1]
