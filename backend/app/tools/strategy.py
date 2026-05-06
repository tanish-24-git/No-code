"""Training strategy tools. Choose method/precision/batch based on
hardware + model + dataset profile."""
from __future__ import annotations

from typing import Any

from app.tools.registry import ToolContext, tool


@tool(
    name="strategy.choose",
    description="Pick training method, precision, batch size, gradient accumulation, lr, and epochs.",
    input_schema={
        "type": "object",
        "properties": {
            "model": {"type": "object"},
            "hardware": {"type": "object"},
            "profile": {"type": "object"},
            "task": {"type": "object"},
            "priority": {"type": "string", "enum": ["quality", "speed", "low_resource"], "default": "quality"},
        },
        "required": ["model", "hardware"],
    },
)
async def strategy_choose(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    model = args["model"]
    hw = args["hardware"]
    profile = args.get("profile") or {}
    task = args.get("task") or {}
    priority = args.get("priority", "quality")

    method = model.get("method") or "lora"
    device = hw.get("device", "cpu")
    vram = hw.get("vram_gb") or 0

    # Precision: bf16 if device supports, fp16 otherwise; fp32 on CPU.
    if device == "cuda":
        precision = "bf16" if vram >= 8 else "fp16"
    elif device == "mps":
        precision = "fp16"
    else:
        precision = "float32"

    # Batch size: pretty conservative defaults.
    if device == "cpu":
        batch_size, grad_accum = 1, 8
    elif vram and vram < 8:
        batch_size, grad_accum = 1, 8
    elif vram and vram < 16:
        batch_size, grad_accum = 2, 4
    elif vram and vram < 24:
        batch_size, grad_accum = 4, 4
    else:
        batch_size, grad_accum = 8, 2

    # Sequence length: clip to model.max_pos and a sane ceiling.
    p95 = int(profile.get("p95") or 0) or 512
    max_pos = int(model.get("max_pos") or 4096)
    max_seq_len = max(256, min(p95 + 32, max_pos, 4096))

    # Learning rate: standard LoRA default.
    learning_rate = 2e-4 if method in ("lora", "qlora") else 5e-5

    # LoRA rank: scale with priority.
    lora_rank = 8 if priority == "low_resource" else 16 if priority != "quality" else 32

    # Epochs: from priority + dataset size.
    rows = int(profile.get("row_count") or 0) or 1000
    if priority == "speed":
        epochs = 1
    elif priority == "low_resource":
        epochs = 2
    else:
        epochs = 3 if rows < 5000 else 2 if rows < 50000 else 1

    # LR schedule + early stopping.
    early_stopping = priority != "speed"

    return {
        "method": method,
        "precision": precision,
        "batch_size": batch_size,
        "gradient_accumulation": grad_accum,
        "max_seq_len": max_seq_len,
        "learning_rate": learning_rate,
        "lora_rank": lora_rank,
        "epochs": epochs,
        "early_stopping": early_stopping,
        "task_type": _map_task_label(task.get("chosen", "chat")),
    }


@tool(
    name="strategy.estimate_runtime",
    description="Rough wall-clock estimate (minutes) for a strategy on given hardware.",
    input_schema={
        "type": "object",
        "properties": {
            "strategy": {"type": "object"},
            "hardware": {"type": "object"},
            "profile":  {"type": "object"},
            "model":    {"type": "object"},
        },
        "required": ["strategy", "hardware", "profile", "model"],
    },
)
async def strategy_estimate_runtime(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    s = args["strategy"]
    hw = args["hardware"]
    p = args["profile"]
    m = args["model"]
    rows = int(p.get("row_count") or 0) or 1
    seq = int(s.get("max_seq_len") or 512)
    epochs = int(s.get("epochs") or 1)
    params_b = float(m.get("params_b") or 1.0)

    # tokens / sec heuristic — same shape as hardware.estimate_throughput.
    device = hw.get("device", "cpu")
    if device == "cuda":
        tps = 8000.0 / max(params_b, 0.5)
    elif device == "mps":
        tps = 2000.0 / max(params_b, 0.5)
    else:
        tps = 150.0 / max(params_b, 0.5)
    if s.get("method") == "qlora":
        tps *= 1.3
    if s.get("precision") in ("int4", "int8"):
        tps *= 1.4

    total_tokens = rows * seq * epochs
    minutes = (total_tokens / max(tps, 1.0)) / 60.0
    return {"estimated_minutes": round(minutes, 1), "tokens_per_sec": round(tps, 1)}


def _map_task_label(t: str) -> str:
    return {
        "chat": "Chat",
        "instruction": "Chat",
        "qa": "QA",
        "classification": "Classification",
        "extraction": "Extraction",
    }.get(t, "Chat")
