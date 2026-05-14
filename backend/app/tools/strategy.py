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
