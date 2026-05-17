"""Recovery proposes a plan diff (ops, rationale, confidence) instead of
performing a blind retry. The agent emits the diff; only after user (or
policy) approval does the execution agent apply it.

Blueprint §9 — Three-level TAO ladder (Think → Act → Observe):

    Level 1 (retry)    transient I/O / API timeouts → resume from checkpoint
    Level 2 (adapt)    OOM / divergence              → mutate config and retry
    Level 3 (escalate) data corruption / hardware    → stop and produce a
                                                        diagnostic report

Every recovery attempt records its level so the UI can render the depth
of intervention.
"""
from __future__ import annotations

import uuid
from typing import Any

from app.tools.registry import ToolContext, tool


_LEVEL1 = {"network_timeout", "rate_limit", "checkpoint_unavailable"}
_LEVEL2 = {
    "loss_nan_or_inf", "loss_no_decrease", "loss_spike",
    "oom", "vram_exceeded",
    "gradient_explosion", "overfitting", "val_train_divergence",
}
_LEVEL3 = {"data_corruption", "hardware_failure", "cuda_unavailable"}


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
    level = "L0"

    if anomaly in _LEVEL1:
        # Transient — just retry with backoff.
        ops = [{"op": "noop", "path": "retry", "old": None, "new": "with_backoff"}]
        rationale = f"{anomaly} is transient; retrying with exponential backoff"
        confidence = 0.85
        level = "L1"

    elif anomaly == "loss_nan_or_inf":
        new_lr = max((cfg.get("learning_rate") or 2e-4) * 0.25, 1e-5)
        ops = [
            {"op": "set", "path": "config.learning_rate", "old": cfg.get("learning_rate"), "new": new_lr},
            {"op": "set", "path": "config.precision", "old": cfg.get("precision"), "new": "fp16"},
        ]
        rationale = "non-finite loss — drop LR by 4x and force fp16 to stabilise gradients"
        confidence = 0.7
        level = "L2"

    elif anomaly == "loss_no_decrease":
        new_lr = (cfg.get("learning_rate") or 2e-4) * 0.5
        ops = [
            {"op": "set", "path": "config.learning_rate", "old": cfg.get("learning_rate"), "new": new_lr},
            {"op": "set", "path": "config.epochs", "old": cfg.get("epochs"), "new": (cfg.get("epochs", 1) or 1) + 1},
        ]
        rationale = "loss flat — halve LR and add an epoch to give gradient descent more room"
        confidence = 0.6
        level = "L2"

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
        level = "L2"

    elif anomaly == "gradient_explosion":
        # Far more aggressive than loss_spike: enable gradient clipping,
        # cut LR by 10x, halve batch to dampen per-step variance.
        old_lr = cfg.get("learning_rate") or 2e-4
        old_bs = cfg.get("batch_size", 4) or 4
        ops = [
            {"op": "set", "path": "config.learning_rate",
             "old": old_lr, "new": max(old_lr * 0.1, 1e-6)},
            {"op": "set", "path": "config.batch_size",
             "old": old_bs, "new": max(1, old_bs // 2)},
            {"op": "set", "path": "config.max_grad_norm",
             "old": cfg.get("max_grad_norm"), "new": 1.0},
        ]
        rationale = (
            "gradient explosion — cut LR 10x, halve batch, clip gradients "
            "at norm=1.0"
        )
        confidence = 0.75
        level = "L2"

    elif anomaly == "overfitting":
        # Eval loss drifting up while train drops: the model is memorising.
        # Bump LoRA dropout, cap epochs at the current value, drop LR.
        cur_epoch = cfg.get("epochs", 3) or 3
        old_lr = cfg.get("learning_rate") or 2e-4
        ops = [
            {"op": "set", "path": "config.lora_dropout",
             "old": cfg.get("lora_dropout"), "new": 0.1},
            {"op": "set", "path": "config.epochs",
             "old": cur_epoch, "new": max(1, cur_epoch - 1)},
            {"op": "set", "path": "config.learning_rate",
             "old": old_lr, "new": old_lr * 0.5},
            {"op": "set", "path": "config.early_stopping",
             "old": cfg.get("early_stopping"), "new": True},
        ]
        rationale = (
            "overfitting (val rising while train falls) — increase LoRA "
            "dropout to 0.1, shorten training by 1 epoch, halve LR, enable "
            "early stopping"
        )
        confidence = 0.7
        level = "L2"

    elif anomaly == "val_train_divergence":
        # Catastrophic-forgetting-style drift: gap widening as the model
        # specialises away from the base distribution. Lower LR + add
        # regularisation; do NOT halve epochs (the user wants more domain
        # specialisation, just gentler).
        old_lr = cfg.get("learning_rate") or 2e-4
        ops = [
            {"op": "set", "path": "config.learning_rate",
             "old": old_lr, "new": old_lr * 0.3},
            {"op": "set", "path": "config.lora_dropout",
             "old": cfg.get("lora_dropout"), "new": 0.08},
            {"op": "set", "path": "config.weight_decay",
             "old": cfg.get("weight_decay"), "new": 0.01},
        ]
        rationale = (
            "val/train divergence — drop LR to 30%, add LoRA dropout 0.08, "
            "add weight decay 0.01 to slow specialisation"
        )
        confidence = 0.65
        level = "L2"

    elif anomaly in {"oom", "vram_exceeded"}:
        # Aggressive memory recovery — gradient checkpointing, smaller seq.
        old_seq = int(cfg.get("max_seq_len") or 512)
        old_bs = cfg.get("batch_size", 4) or 4
        ops = [
            {"op": "set", "path": "config.batch_size", "old": old_bs, "new": max(1, old_bs // 2)},
            {"op": "set", "path": "config.gradient_accumulation",
             "old": cfg.get("gradient_accumulation"),
             "new": (cfg.get("gradient_accumulation", 4) or 4) * 2},
            {"op": "set", "path": "config.max_seq_len", "old": old_seq, "new": max(256, old_seq // 2)},
            {"op": "set", "path": "config.gradient_checkpointing", "old": cfg.get("gradient_checkpointing"), "new": True},
        ]
        rationale = "OOM — halve batch, halve sequence length, enable gradient checkpointing"
        confidence = 0.75
        level = "L2"

    elif anomaly in _LEVEL3:
        # Cannot adapt — produce a diagnostic and stop.
        ops = [{"op": "stop", "path": "session", "old": None, "new": "escalate_to_user"}]
        rationale = (
            f"{anomaly} is irrecoverable from inside the agent; "
            "producing a diagnostic report for the user"
        )
        confidence = 0.95
        level = "L3"

    return {
        "diff_id": diff_id,
        "level": level,
        "operations": ops,
        "rationale": rationale,
        "confidence": confidence,
        "estimated_extra_minutes": extra_minutes,
    }
