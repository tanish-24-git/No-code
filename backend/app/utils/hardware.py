"""Cheap, dependency-tolerant hardware detection. Used by the agent to suggest
configs and by the API for /health."""
from __future__ import annotations

import logging
import platform
import re
from typing import Any


log = logging.getLogger("finetune-studio.hardware")


def detect_hardware() -> dict[str, Any]:
    info: dict[str, Any] = {
        "device": "cpu",
        "gpu_name": None,
        "vram_gb": None,
        "cuda_version": None,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "recommended_trainer": "lora",
    }
    try:
        import torch  # type: ignore
    except Exception:
        info["recommended_trainer"] = "lora"
        return info

    try:
        if torch.cuda.is_available():
            idx = 0
            props = torch.cuda.get_device_properties(idx)
            vram_gb = round(props.total_memory / (1024**3), 2)
            info.update(
                device="cuda",
                gpu_name=props.name,
                vram_gb=vram_gb,
                cuda_version=getattr(torch.version, "cuda", None),
            )
            if vram_gb < 6:
                info["recommended_trainer"] = "qlora"
            elif vram_gb < 16:
                info["recommended_trainer"] = "lora"
            else:
                info["recommended_trainer"] = "lora"  # full requires careful setup; default to LoRA
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            info.update(device="mps", gpu_name="Apple Silicon")
            info["recommended_trainer"] = "lora"
    except Exception:
        # Torch present but unhappy — stay on CPU defaults rather than crash.
        pass
    return info


# ── VRAM budget math ──────────────────────────────────────────────────────
#
# Programmatic memory sizing so we don't OOM on a user's GPU when the LLM
# chose ambitious hyperparameters. Inputs are the LLM-proposed strategy +
# the detected hardware; output tells the caller whether it fits and, if
# not, how much room is missing.


# Curated fallback table used when transformers.AutoConfig can't be reached
# (offline / private repo / network blip). Best-effort — better than a wild
# guess. Keys match common substrings in the model id.
_MODEL_CARD_FALLBACKS: dict[str, dict[str, Any]] = {
    # name_substring (lowercased) -> {params_b, hidden_size, num_hidden_layers}
    "tinyllama": {"params_b": 1.1, "hidden_size": 2048, "num_hidden_layers": 22},
    "phi-1_5": {"params_b": 1.3, "hidden_size": 2048, "num_hidden_layers": 24},
    "phi-2": {"params_b": 2.7, "hidden_size": 2560, "num_hidden_layers": 32},
    "phi-3-mini": {"params_b": 3.8, "hidden_size": 3072, "num_hidden_layers": 32},
    "phi-3": {"params_b": 3.8, "hidden_size": 3072, "num_hidden_layers": 32},
    "gemma-2b": {"params_b": 2.0, "hidden_size": 2048, "num_hidden_layers": 18},
    "gemma-7b": {"params_b": 7.0, "hidden_size": 3072, "num_hidden_layers": 28},
    "llama-2-7b": {"params_b": 7.0, "hidden_size": 4096, "num_hidden_layers": 32},
    "llama-2-13b": {"params_b": 13.0, "hidden_size": 5120, "num_hidden_layers": 40},
    "llama-3-8b": {"params_b": 8.0, "hidden_size": 4096, "num_hidden_layers": 32},
    "llama-3.1-8b": {"params_b": 8.0, "hidden_size": 4096, "num_hidden_layers": 32},
    "llama-3.2-1b": {"params_b": 1.2, "hidden_size": 2048, "num_hidden_layers": 16},
    "llama-3.2-3b": {"params_b": 3.2, "hidden_size": 3072, "num_hidden_layers": 28},
    "qwen2-1.5b": {"params_b": 1.5, "hidden_size": 1536, "num_hidden_layers": 28},
    "qwen2-7b": {"params_b": 7.0, "hidden_size": 3584, "num_hidden_layers": 28},
    "mistral-7b": {"params_b": 7.0, "hidden_size": 4096, "num_hidden_layers": 32},
    "mistral-7b-instruct": {"params_b": 7.0, "hidden_size": 4096, "num_hidden_layers": 32},
    "mixtral-8x7b": {"params_b": 47.0, "hidden_size": 4096, "num_hidden_layers": 32},
    "deepseek-7b": {"params_b": 7.0, "hidden_size": 4096, "num_hidden_layers": 30},
    "gpt2": {"params_b": 0.124, "hidden_size": 768, "num_hidden_layers": 12},
}


def _params_b_from_name(model_id: str) -> float | None:
    """Last-ditch heuristic: pull the parameter count from the model name."""
    name = (model_id or "").lower()
    # Matches "7b", "7B", "7.5b", "1.1B" — common HF naming.
    m = re.search(r"(\d+(?:\.\d+)?)\s*b\b", name)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _resolve_model_card(model_id: str) -> dict[str, Any]:
    """Get (params_b, hidden_size, num_hidden_layers) for the model.

    Try `transformers.AutoConfig` first; fall back to the curated table; fall
    back to name-based heuristics. Always returns *some* answer so the
    downstream budget math never raises.
    """
    out: dict[str, Any] = {"params_b": None, "hidden_size": None, "num_hidden_layers": None}
    if not model_id:
        return out

    try:
        from transformers import AutoConfig  # type: ignore
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        hidden = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None)
        layers = (
            getattr(cfg, "num_hidden_layers", None)
            or getattr(cfg, "n_layer", None)
            or getattr(cfg, "num_layers", None)
        )
        params = getattr(cfg, "num_parameters", None)
        if hidden and layers:
            out["hidden_size"] = int(hidden)
            out["num_hidden_layers"] = int(layers)
            if params:
                out["params_b"] = round(float(params) / 1e9, 3)
            else:
                # Coarse estimate: 12 * hidden^2 * layers ≈ transformer params
                approx = 12 * (hidden ** 2) * layers
                out["params_b"] = round(approx / 1e9, 3)
            return out
    except Exception as e:
        log.info("AutoConfig lookup failed for %s (%s); falling back to table", model_id, e)

    name = (model_id or "").lower()
    for key, row in _MODEL_CARD_FALLBACKS.items():
        if key in name:
            out.update(row)
            return out

    # Last resort: just guess from the name suffix.
    pb = _params_b_from_name(model_id)
    if pb is not None:
        # Crude scaling: hidden ~ sqrt(params/12/layers); fix layers to a safe 32.
        layers = 32
        hidden = int((pb * 1e9 / (12 * layers)) ** 0.5)
        out.update({"params_b": pb, "hidden_size": max(hidden, 512), "num_hidden_layers": layers})
        return out

    # Truly unknown: assume a 7B-class transformer.
    out.update({"params_b": 7.0, "hidden_size": 4096, "num_hidden_layers": 32})
    return out


def _bytes_per_param(precision: str, quantization: str, method: str) -> float:
    q = (quantization or "none").lower()
    if q in ("int4", "4bit", "nf4") or (method or "").lower() == "qlora":
        return 0.5
    if q in ("int8", "8bit"):
        return 1.0
    p = (precision or "bf16").lower()
    if p in ("bf16", "bfloat16", "fp16", "float16"):
        return 2.0
    return 4.0  # fp32


def compute_vram_budget(
    *,
    model_id: str,
    method: str = "lora",
    precision: str = "bf16",
    quantization: str = "none",
    batch_size: int = 1,
    max_seq_len: int = 1024,
    gradient_accumulation: int = 1,  # unused for memory; kept for API symmetry
    use_gradient_checkpointing: bool = True,
    lora_rank: int = 16,
) -> dict[str, Any]:
    """Estimate VRAM bytes required for one training step and compare to the
    detected GPU. Pure function (no torch import needed)."""
    card = _resolve_model_card(model_id)
    params_b = card["params_b"] or 7.0
    hidden = int(card["hidden_size"] or 4096)
    layers = int(card["num_hidden_layers"] or 32)
    params = int(params_b * 1e9)

    bpp = _bytes_per_param(precision, quantization, method)
    model_bytes = int(params * bpp)

    is_full = (method or "").lower() == "full"
    # Trainable params for LoRA: roughly r * (rank-projected dims) * 2 matrices
    # per attention block. The exact figure depends on which modules are
    # targeted; we use a calibrated approximation matching peft defaults.
    if is_full:
        trainable_params = params
    else:
        # ~ 4 * rank * hidden * 2 matrices * layers (q,k,v,o + sometimes mlp)
        # Drop the 0.5 factor for QLoRA since trainable params are still fp32.
        trainable_params = int(min(params * 0.02, 6 * lora_rank * hidden * layers))

    # Adam state: 2 fp32 moments per trainable param.
    optimizer_bytes = trainable_params * 8
    # Gradients: fp32 per trainable param.
    gradient_bytes = trainable_params * 4
    # Adapter weights stored in fp32 even when base is quantized.
    adapter_bytes = trainable_params * 4 if not is_full else 0

    # Activations (forward pass). Calibrated against empirical 7B-LoRA runs:
    # ~ batch * seq * hidden * layers * ~4 dtype-bytes * (gc factor).
    gc_factor = 0.30 if use_gradient_checkpointing else 1.0
    activation_bpp = 2.0 if precision in ("bf16", "bfloat16", "fp16", "float16") else 4.0
    activation_bytes = int(
        batch_size * max_seq_len * hidden * layers * activation_bpp * 4.0 * gc_factor
    )

    # CUDA workspace + kernel scratch.
    overhead_bytes = int(1.0 * (1024 ** 3))  # 1 GiB

    total = (
        model_bytes
        + activation_bytes
        + adapter_bytes
        + optimizer_bytes
        + gradient_bytes
        + overhead_bytes
    )

    hw = detect_hardware()
    available_gb = float(hw.get("vram_gb") or 0.0)
    required_gb = round(total / (1024 ** 3), 2)
    utilization = round(required_gb / max(available_gb, 0.01), 3) if available_gb else float("inf")
    fits = available_gb >= required_gb if available_gb else False

    return {
        "required_gb": required_gb,
        "available_gb": round(available_gb, 2),
        "fits": bool(fits),
        "utilization": utilization,
        "model_params_b": params_b,
        "hidden_size": hidden,
        "num_layers": layers,
        "breakdown_gb": {
            "model": round(model_bytes / (1024 ** 3), 2),
            "activations": round(activation_bytes / (1024 ** 3), 2),
            "adapter": round(adapter_bytes / (1024 ** 3), 2),
            "optimizer": round(optimizer_bytes / (1024 ** 3), 2),
            "gradients": round(gradient_bytes / (1024 ** 3), 2),
            "overhead": round(overhead_bytes / (1024 ** 3), 2),
        },
        "device": hw.get("device", "cpu"),
    }


def clamp_strategy_to_vram(
    strategy: dict[str, Any],
    model_id: str,
    *,
    safety_margin: float = 0.85,
    max_halvings: int = 8,
) -> dict[str, Any]:
    """Halve batch_size / max_seq_len until the strategy fits within
    ``safety_margin`` of the GPU's VRAM. No-op when no GPU is detected
    (CPU Safe-Mode in trainer.py takes care of that path).

    Mutates ``strategy`` in place and returns it. Adds a private
    ``_vram_clamp`` field documenting what was changed so the agent can
    surface the diff to the user.
    """
    hw = detect_hardware()
    if not hw.get("vram_gb") or hw.get("device") != "cuda":
        return strategy

    bs = max(int(strategy.get("batch_size") or 1), 1)
    sl = max(int(strategy.get("max_seq_len") or 1024), 256)
    method = strategy.get("method", "lora")
    precision = strategy.get("precision", "bf16")
    quant = strategy.get("quantization", "none")
    lora_rank = int(strategy.get("lora_rank") or 16)

    original_bs, original_sl = bs, sl
    final_budget: dict[str, Any] = {}

    for _ in range(max_halvings):
        budget = compute_vram_budget(
            model_id=model_id,
            method=method,
            precision=precision,
            quantization=quant,
            batch_size=bs,
            max_seq_len=sl,
            lora_rank=lora_rank,
        )
        final_budget = budget
        if budget["fits"] and budget["utilization"] <= safety_margin:
            break
        # Halve the dimension that contributes most. Seq len enters as
        # ~ batch * seq * hidden * layers — so cutting whichever is bigger
        # in absolute terms gives the biggest single drop.
        if sl >= bs * 256 and sl > 256:
            sl = max(sl // 2, 256)
        elif bs > 1:
            bs = max(bs // 2, 1)
        elif sl > 256:
            sl = max(sl // 2, 256)
        else:
            break  # already at floor; can't shrink further

    if bs != original_bs or sl != original_sl:
        strategy["batch_size"] = bs
        strategy["max_seq_len"] = sl
        strategy["_vram_clamp"] = {
            "from": {"batch_size": original_bs, "max_seq_len": original_sl},
            "to": {"batch_size": bs, "max_seq_len": sl},
            "budget": final_budget,
            "safety_margin": safety_margin,
        }
        log.info(
            "VRAM clamp: bs %d->%d, seq %d->%d (need %.1fGB of %.1fGB)",
            original_bs, bs, original_sl, sl,
            final_budget.get("required_gb", 0), final_budget.get("available_gb", 0),
        )
    return strategy
