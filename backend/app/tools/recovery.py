"""Recovery proposes a plan diff (ops, rationale, confidence) instead of
performing a blind retry. The agent emits the diff; only after user (or
policy) approval does the execution agent apply it."""
from __future__ import annotations

import uuid
from typing import Any

from app.tools.registry import ToolContext, tool


@tool(
    name="recovery.propose_plan",
    description="Given an anomaly + current config, propose a list of plan-diff operations.",
    input_schema={
        "type": "object",
        "properties": {
            "anomaly": {"type": "string"},
            "config": {"type": "object"},
            "extra_minutes": {"type": "number"},
        },
        "required": ["anomaly", "config"],
    },
)
async def recovery_propose_plan(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    anomaly = args["anomaly"]
    cfg = args.get("config") or {}
    extra_minutes = float(args.get("extra_minutes", 5.0))
    diff_id = "rec_" + uuid.uuid4().hex[:8]
    ops: list[dict[str, Any]] = []
    rationale: str = "no recovery available"
    confidence = 0.5

    if anomaly == "loss_nan_or_inf":
        new_lr = max((cfg.get("learning_rate") or 2e-4) * 0.25, 1e-5)
        ops = [
            {"op": "set", "path": "config.learning_rate", "old": cfg.get("learning_rate"), "new": new_lr},
            {"op": "set", "path": "config.precision", "old": cfg.get("precision"), "new": "fp16"},
        ]
        rationale = "non-finite loss — drop LR by 4x and force fp16 to stabilise gradients"
        confidence = 0.7
    elif anomaly == "loss_no_decrease":
        new_lr = (cfg.get("learning_rate") or 2e-4) * 0.5
        ops = [
            {"op": "set", "path": "config.learning_rate", "old": cfg.get("learning_rate"), "new": new_lr},
            {"op": "set", "path": "config.epochs", "old": cfg.get("epochs"), "new": (cfg.get("epochs", 1) or 1) + 1},
        ]
        rationale = "loss flat — halve LR and add an epoch to give gradient descent more room"
        confidence = 0.6
    elif anomaly == "loss_spike":
        old_bs = cfg.get("batch_size", 4) or 4
        ops = [
            {"op": "set", "path": "config.batch_size", "old": old_bs, "new": max(1, old_bs // 2)},
            {"op": "set", "path": "config.gradient_accumulation",
             "old": cfg.get("gradient_accumulation"),
             "new": (cfg.get("gradient_accumulation", 4) or 4) * 2},
        ]
        rationale = "sudden loss spike — halve batch and double grad-accum to reduce per-step variance"
        confidence = 0.55

    return {
        "diff_id": diff_id,
        "operations": ops,
        "rationale": rationale,
        "confidence": confidence,
        "estimated_extra_minutes": extra_minutes,
    }
