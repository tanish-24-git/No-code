"""Model selection tools. The catalog is intentionally short and curated;
add models as you validate them. Scoring is deterministic and explainable."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.tools.registry import ToolContext, tool


@dataclass
class BaseModelSpec:
    repo_id: str
    label: str
    params_b: float           # parameter count in billions
    max_pos: int
    has_chat_template: bool
    family: str
    domain_tags: tuple[str, ...] = ()


# Hand-picked, broadly available, license-friendly defaults. The agent picks
# from this list; the user can always override.
CATALOG: list[BaseModelSpec] = [
    BaseModelSpec("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "TinyLlama 1.1B Chat", 1.1, 2048, True, "llama"),
    BaseModelSpec("Qwen/Qwen2.5-0.5B-Instruct", "Qwen2.5 0.5B Instruct", 0.5, 32768, True, "qwen"),
    BaseModelSpec("Qwen/Qwen2.5-1.5B-Instruct", "Qwen2.5 1.5B Instruct", 1.5, 32768, True, "qwen"),
    BaseModelSpec("Qwen/Qwen2.5-3B-Instruct", "Qwen2.5 3B Instruct", 3.0, 32768, True, "qwen"),
    BaseModelSpec("Qwen/Qwen2.5-7B-Instruct", "Qwen2.5 7B Instruct", 7.0, 32768, True, "qwen"),
    BaseModelSpec("microsoft/Phi-3.5-mini-instruct", "Phi-3.5 mini Instruct", 3.8, 131072, True, "phi"),
    BaseModelSpec("meta-llama/Llama-3.2-1B-Instruct", "Llama 3.2 1B Instruct", 1.0, 131072, True, "llama"),
    BaseModelSpec("meta-llama/Llama-3.2-3B-Instruct", "Llama 3.2 3B Instruct", 3.0, 131072, True, "llama"),
    BaseModelSpec("meta-llama/Llama-3.1-8B-Instruct", "Llama 3.1 8B Instruct", 8.0, 131072, True, "llama"),
    BaseModelSpec("google/gemma-2-2b-it", "Gemma 2 2B IT", 2.0, 8192, True, "gemma"),
    BaseModelSpec("HuggingFaceTB/SmolLM2-360M-Instruct", "SmolLM2 360M Instruct", 0.36, 8192, True, "smollm"),
    BaseModelSpec("HuggingFaceTB/SmolLM2-1.7B-Instruct", "SmolLM2 1.7B Instruct", 1.7, 8192, True, "smollm"),
]


@tool(
    name="model.rank_candidates",
    description=(
        "Rank base models for a given (hardware, dataset profile, task) joint "
        "context. If `shortlist` is provided, ranks those models; otherwise "
        "ranks the curated catalogue."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "hardware":  {"type": "object"},
            "profile":   {"type": "object"},
            "task":      {"type": "object"},
            "top_n":     {"type": "integer", "default": 5},
            "shortlist": {"type": "array"},
        },
        "required": ["hardware", "profile", "task"],
    },
)
async def model_rank_candidates(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    hw = args["hardware"] or {}
    pr = args["profile"] or {}
    tk = args["task"] or {}
    n = int(args.get("top_n", 5))
    shortlist = args.get("shortlist") or None

    pool: list[BaseModelSpec] = []
    if shortlist:
        for c in shortlist:
            pool.append(BaseModelSpec(
                repo_id=c.get("repo_id", ""),
                label=c.get("label") or c.get("repo_id", ""),
                params_b=float(c.get("params_b") or 0.0),
                max_pos=int(c.get("max_pos") or 4096),
                has_chat_template=True,            # search filtered to instruct-tuned
                family=_infer_family(c.get("repo_id", "")),
                domain_tags=tuple(c.get("tags") or ()),
            ))
    else:
        pool = list(CATALOG)

    rows: list[dict[str, Any]] = []
    for m in pool:
        if not m.repo_id:
            continue
        s = _score(m, hw, pr, tk)
        rows.append({
            "repo_id": m.repo_id,
            "label": m.label,
            "params_b": m.params_b,
            "max_pos": m.max_pos,
            "family": m.family,
            "score": round(s["score"], 4),
            "reasons": s["reasons"],
            "fits": s["fits"],
            "method": s["method"],
        })
    rows.sort(key=lambda r: r["score"], reverse=True)
    return {"candidates": rows[:n], "considered": len(pool)}


def _infer_family(repo_id: str) -> str:
    rid = repo_id.lower()
    if "llama" in rid:
        return "llama"
    if "qwen" in rid:
        return "qwen"
    if "phi" in rid:
        return "phi"
    if "gemma" in rid:
        return "gemma"
    if "mistral" in rid:
        return "mistral"
    if "smollm" in rid:
        return "smollm"
    return "other"


@tool(
    name="model.estimate_fit",
    description="Estimate VRAM fit and recommended training method (lora/qlora/full) for a single model.",
    input_schema={
        "type": "object",
        "properties": {
            "repo_id": {"type": "string"},
            "hardware": {"type": "object"},
            "max_seq_len": {"type": "integer", "default": 1024},
        },
        "required": ["repo_id", "hardware"],
    },
)
async def model_estimate_fit(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    repo_id = args["repo_id"]
    spec = next((m for m in CATALOG if m.repo_id == repo_id), None)
    if not spec:
        return {"error": "model not in curated catalog", "repo_id": repo_id}
    hw = args["hardware"] or {}
    method = _recommend_method(spec, hw)
    return {
        "repo_id": repo_id,
        "params_b": spec.params_b,
        "method": method,
        "vram_gb": hw.get("vram_gb"),
        "device": hw.get("device", "cpu"),
        "max_seq_len": int(args.get("max_seq_len", 1024)),
    }


# ── scoring internals ─────────────────────────────────────────────────────

def _score(m: BaseModelSpec, hw: dict[str, Any], pr: dict[str, Any], tk: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    score = 0.0
    fits = True

    # 1. Memory fit
    method = _recommend_method(m, hw)
    vram = hw.get("vram_gb") or 0
    device = hw.get("device", "cpu")
    if device == "cpu":
        # CPU only: punish big models hard.
        if m.params_b <= 0.5:
            score += 0.4
            reasons.append("tiny model — runnable on CPU")
        elif m.params_b <= 1.5:
            score += 0.1
            reasons.append("small model — slow but workable on CPU")
        else:
            score -= 0.5
            fits = False
            reasons.append("too large for CPU-only")
    else:
        budget = vram if vram else 8.0
        # Crude: 2 GB/B param at bf16 + LoRA overhead.
        needed = m.params_b * (2.2 if method == "lora" else 0.6 if method == "qlora" else 4.0)
        if needed <= budget * 0.85:
            score += 0.5
            reasons.append(f"fits in VRAM ({needed:.1f}GB <= {budget:.1f}GB) with {method}")
        elif needed <= budget:
            score += 0.2
            reasons.append(f"tight VRAM fit with {method}")
        else:
            score -= 0.4
            fits = False
            reasons.append("does not fit even with qlora")

    # 2. Sequence length fit
    p95 = pr.get("p95") or pr.get("token_p95") or 0
    if p95 and m.max_pos < p95:
        score -= 0.3
        reasons.append(f"max_pos {m.max_pos} < dataset p95 tokens {p95}")
    elif p95:
        score += 0.1
        reasons.append("seq-len OK")

    # 3. Task fit
    task_chosen = tk.get("chosen", "instruction")
    if task_chosen in ("chat", "instruction", "qa") and m.has_chat_template:
        score += 0.2
        reasons.append("chat template available")
    if task_chosen == "classification":
        # Smaller models often do classification just as well after fine-tune.
        if m.params_b <= 3.0:
            score += 0.1
            reasons.append("small model is fine for classification")

    # 4. Family preference (favour Qwen/Llama for breadth of community support).
    if m.family in ("qwen", "llama"):
        score += 0.05

    return {"score": score, "reasons": reasons, "fits": fits, "method": method}


def _recommend_method(m: BaseModelSpec, hw: dict[str, Any]) -> str:
    device = hw.get("device", "cpu")
    vram = hw.get("vram_gb") or 0
    if device == "cpu":
        return "lora"      # but only viable for tiny models; scoring punishes the rest
    if device == "mps":
        return "lora"
    if vram < 6:
        return "qlora"
    if vram < 24:
        return "lora"
    if m.params_b <= 7.0:
        return "lora"
    return "qlora"
