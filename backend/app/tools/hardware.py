"""Hardware detection. Wraps the existing detector and adds a coarse
throughput estimate so downstream agents can reason about runtime."""
from __future__ import annotations

from typing import Any

from app.tools.registry import ToolContext, tool
from app.utils.hardware import compute_vram_budget, detect_hardware


@tool(
    name="hardware.detect",
    description="Detect CPU/GPU/MPS, VRAM, CUDA, and a recommended training method.",
    input_schema={"type": "object", "properties": {}},
)
async def hardware_detect(_args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    info = detect_hardware()
    info["ram_gb"] = _detect_ram_gb()
    info["disk_free_gb"] = _detect_disk_free_gb()
    return info


@tool(
    name="hardware.estimate_throughput",
    description="Rough tokens/sec estimate for a given param count and precision on the detected hardware.",
    input_schema={
        "type": "object",
        "properties": {
            "params_billions": {"type": "number"},
            "precision": {"type": "string", "enum": ["bf16", "fp16", "float32", "int4", "int8"]},
        },
        "required": ["params_billions"],
    },
)
async def hardware_estimate_throughput(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    info = detect_hardware()
    params = float(args["params_billions"])
    precision = args.get("precision", "bf16")
    device = info.get("device", "cpu")
    vram = info.get("vram_gb") or 0
    # Very rough heuristic: tokens/sec falls with params and grows with VRAM.
    if device == "cuda":
        base = 8000.0 / max(params, 0.5)        # 1B ~ 8k tok/s baseline
        if precision in ("int4", "int8"):
            base *= 1.4
        if precision == "float32":
            base *= 0.5
        # Smaller VRAM means we can't keep many tokens in flight.
        if vram and vram < 8:
            base *= 0.6
        return {"tokens_per_sec": round(base, 1), "device": device, "params_b": params}
    if device == "mps":
        return {"tokens_per_sec": round(2000.0 / max(params, 0.5), 1), "device": device, "params_b": params}
    return {"tokens_per_sec": round(150.0 / max(params, 0.5), 1), "device": device, "params_b": params}


@tool(
    name="hardware.vram_budget",
    description=(
        "Estimate VRAM required to fine-tune a given model with the given "
        "hyperparameters, and report whether it fits on the detected GPU. "
        "Use this BEFORE finalizing a training strategy to avoid CUDA OOM."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "model_id": {"type": "string", "description": "HF repo id (e.g. meta-llama/Llama-3.1-8B)."},
            "method": {"type": "string", "description": "lora | qlora | dora | full"},
            "precision": {"type": "string", "description": "bf16 | fp16 | float32"},
            "quantization": {"type": "string", "description": "none | int4 | int8"},
            "batch_size": {"type": "integer", "minimum": 1},
            "max_seq_len": {"type": "integer", "minimum": 64},
            "gradient_accumulation": {"type": "integer", "minimum": 1, "default": 1},
            "lora_rank": {"type": "integer", "minimum": 1, "default": 16},
        },
        "required": ["model_id", "batch_size", "max_seq_len"],
    },
)
async def hardware_vram_budget(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    return compute_vram_budget(
        model_id=str(args.get("model_id") or ""),
        method=str(args.get("method") or "lora"),
        precision=str(args.get("precision") or "bf16"),
        quantization=str(args.get("quantization") or "none"),
        batch_size=int(args.get("batch_size") or 1),
        max_seq_len=int(args.get("max_seq_len") or 1024),
        gradient_accumulation=int(args.get("gradient_accumulation") or 1),
        lora_rank=int(args.get("lora_rank") or 16),
    )


def _detect_ram_gb() -> float | None:
    try:
        import psutil  # type: ignore
        return round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except Exception:
        return None


def _detect_disk_free_gb() -> float | None:
    try:
        import shutil
        usage = shutil.disk_usage(".")
        return round(usage.free / (1024 ** 3), 1)
    except Exception:
        return None
