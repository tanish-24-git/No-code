"""Training strategy tools. Choose method/precision/batch based on
hardware + model + dataset profile.

Blueprint §5 SOTA-2026 selection:

    method            → lora | qlora | full
    adapter_variant   → none | dora | galore
    kernel_pack       → standard | unsloth
    alignment         → none | dpo | orpo

The Architectural Designer (TrainingStrategyAgent) reads ``rationale`` to
explain *why* each choice was made — a hard requirement of commandment §7
("Explain the Why").
"""
from __future__ import annotations

from typing import Any

from app.tools.registry import ToolContext, tool


@tool(
    name="strategy.choose",
    description="Pick training method, precision, batch size, gradient accumulation, lr, epochs, and SOTA-2026 PEFT variant.",
    input_schema={
        "type": "object",
        "properties": {
            "model": {"type": "object"},
            "hardware": {"type": "object"},
            "profile": {"type": "object"},
            "task": {"type": "object"},
            "priority": {"type": "string", "enum": ["quality", "speed", "low_resource"], "default": "quality"},
            "alignment": {"type": "string", "enum": ["none", "dpo", "orpo"], "default": "none"},
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
    alignment = args.get("alignment", "none")

    method = model.get("method") or "lora"
    device = hw.get("device", "cpu")
    vram = hw.get("vram_gb") or 0
    params_b = float(model.get("params_b") or 1.0)
    rationale: list[str] = []

    # Precision: bf16 if device supports, fp16 otherwise; fp32 on CPU.
    if device == "cuda":
        precision = "bf16" if vram >= 8 else "fp16"
    elif device == "mps":
        precision = "fp16"
    else:
        precision = "float32"
    rationale.append(f"precision={precision} based on device={device} vram={vram}GB")

    # Quantization downgrade on tight VRAM.
    quantization = "none"
    if device == "cuda" and vram and vram < 8 and params_b > 1:
        quantization = "int4"
        method = "qlora"
        rationale.append(
            f"int4 quantization + QLoRA — full-precision projection "
            f"of a {params_b:.1f}B model would exceed {vram}GB VRAM"
        )
    elif device == "cuda" and vram and vram < 12 and params_b > 3:
        quantization = "int8"
        rationale.append(f"int8 quantization to fit {params_b:.1f}B parameters in {vram}GB")

    # Adapter variant — DoRA when there's headroom and quality matters,
    # GaLore when the user wants full-parameter learning on consumer GPUs.
    adapter_variant = "none"
    if method in ("lora", "qlora") and priority == "quality" and params_b >= 1.0 and (device != "cpu"):
        adapter_variant = "dora"
        rationale.append(
            "DoRA decouples weight magnitude and direction for sharper "
            "updates without extra inference cost"
        )
    if method == "full" and device == "cuda" and vram and vram < 24:
        # GaLore lets full-parameter training fit on prosumer cards.
        method = "lora"
        adapter_variant = "galore"
        rationale.append(
            "GaLore projects gradients into a low-rank subspace — full-parameter "
            f"learning would exceed your {vram}GB VRAM budget"
        )

    # Unsloth fusions — only on CUDA with bf16/fp16. Worth ~70% VRAM cut.
    kernel_pack = "standard"
    if device == "cuda" and precision in ("bf16", "fp16") and method in ("lora", "qlora"):
        kernel_pack = "unsloth"
        rationale.append("Unsloth fused kernels enabled — up to 70% VRAM savings")

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
    if kernel_pack == "unsloth":
        # Unsloth typically lets us double the batch.
        batch_size = max(1, batch_size * 2)
        grad_accum = max(1, grad_accum // 2)
    rationale.append(f"micro_batch={batch_size}×{grad_accum} grad_accum")

    # Sequence length: clip to model.max_pos and a sane ceiling.
    p95 = int(profile.get("p95") or 0) or 512
    max_pos = int(model.get("max_pos") or 4096)
    max_seq_len = max(256, min(p95 + 32, max_pos, 4096))
    rationale.append(f"max_seq_len={max_seq_len} (dataset p95={p95}, model max_pos={max_pos})")

    # Learning rate: standard LoRA default.
    learning_rate = 2e-4 if method in ("lora", "qlora") else 5e-5
    if adapter_variant == "dora":
        learning_rate *= 0.7
        rationale.append(f"DoRA prefers a slightly lower LR ({learning_rate:.1e})")
    if adapter_variant == "galore":
        learning_rate = 1e-4

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
    rationale.append(f"{epochs} epoch(s) for ~{rows} rows × priority={priority}")

    if alignment in ("dpo", "orpo"):
        rationale.append(
            f"{alignment.upper()} alignment scheduled after SFT — preference "
            "optimization without a separate reward model"
        )

    # LR schedule + early stopping.
    early_stopping = priority != "speed"

    return {
        "method": method,
        "adapter_variant": adapter_variant,
        "kernel_pack": kernel_pack,
        "quantization": quantization,
        "alignment": alignment,
        "precision": precision,
        "batch_size": batch_size,
        "gradient_accumulation": grad_accum,
        "max_seq_len": max_seq_len,
        "learning_rate": learning_rate,
        "lora_rank": lora_rank,
        "epochs": epochs,
        "early_stopping": early_stopping,
        "task_type": _map_task_label(task.get("chosen", "chat")),
        "rationale": rationale,
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
