"""Metrics + anomaly detection tools."""
from __future__ import annotations

import math
import re
from typing import Any

from app.services import job_service
from app.tools.registry import ToolContext, tool


# Patterns scanned in the trailing log buffer when the caller passes a
# job_id. These map directly to recovery.propose_plan reasons so the
# RecoveryAgent can act without a translation step.
_OOM_PATTERNS = re.compile(
    r"(cuda out of memory|cuda oom|outofmemoryerror|"
    r"runtimeerror.*out of memory|cublas_status_alloc_failed|"
    r"torch\.cuda\.outofmemoryerror)",
    re.IGNORECASE,
)
_DATA_CORRUPTION_PATTERNS = re.compile(
    r"(unicodedecodeerror|jsondecodeerror|expected.*column|"
    r"datasetschemaerror|corrupted|truncated stream)",
    re.IGNORECASE,
)
_CUDA_UNAVAILABLE_PATTERNS = re.compile(
    r"(cuda(?: is)? not available|cuda driver not found|"
    r"no nvidia driver|libcudart.*cannot open)",
    re.IGNORECASE,
)


@tool(
    name="metrics.read",
    description="Read all recorded metrics for a job.",
    input_schema={
        "type": "object",
        "properties": {"job_id": {"type": "string"}},
        "required": ["job_id"],
    },
)
async def metrics_read(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    job = job_service.get(args["job_id"])
    if not job:
        return {"error": "job not found"}
    return {
        "job_id": job.id,
        "status": job.status,
        "current_loss": job.current_loss,
        "current_step": job.current_step,
        "current_epoch": job.current_epoch,
        "metrics": [m.model_dump(mode="json") for m in job.metrics],
    }


@tool(
    name="metrics.detect_anomaly",
    description=(
        "Detect training anomalies: NaN/Inf loss, no decrease over a "
        "window, sudden spike, gradient explosion, overfitting "
        "(train/val divergence), OOM / data-corruption / CUDA-unavailable "
        "(via log scan when job_id is provided). Reasons map directly to "
        "recovery.propose_plan handlers."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "metrics": {"type": "array"},
            "window": {"type": "integer", "default": 5},
            "job_id": {
                "type": "string",
                "description": (
                    "Optional: when provided, recent log lines are scanned "
                    "for OOM / data corruption / CUDA driver errors."
                ),
            },
        },
        "required": ["metrics"],
    },
)
async def metrics_detect_anomaly(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    metrics = args.get("metrics") or []
    window = max(int(args.get("window", 5)), 2)
    job_id = args.get("job_id")

    # 1. Log-buffer scan — catches OOM / data corruption / CUDA errors
    #    that never surface as a loss reading because training crashed
    #    before the next on_log callback fired.
    if job_id:
        try:
            recent_logs = "\n".join(job_service.get_buffered_logs(job_id)[-80:])
        except Exception:
            recent_logs = ""
        if recent_logs:
            if _OOM_PATTERNS.search(recent_logs):
                return {
                    "anomaly": "oom",
                    "reason": "CUDA out-of-memory observed in recent logs",
                    "source": "log_scan",
                }
            if _CUDA_UNAVAILABLE_PATTERNS.search(recent_logs):
                return {
                    "anomaly": "cuda_unavailable",
                    "reason": "CUDA driver/runtime unavailable",
                    "source": "log_scan",
                }
            if _DATA_CORRUPTION_PATTERNS.search(recent_logs):
                return {
                    "anomaly": "data_corruption",
                    "reason": "dataset decode / schema error observed",
                    "source": "log_scan",
                }

    losses = [m.get("loss") for m in metrics if m.get("loss") is not None]
    if not losses:
        return {"anomaly": None, "reason": "no losses yet"}

    # 2. Non-finite loss — fastest possible signal.
    last = losses[-1]
    try:
        last_f = float(last)
    except (TypeError, ValueError):
        last_f = 0.0
    if math.isnan(last_f) or math.isinf(last_f):
        return {"anomaly": "loss_nan_or_inf", "reason": "non-finite loss", "value": last_f}

    # 3. Gradient explosion — last loss >= 10x recent window mean. Stronger
    #    signal than the 3x spike check below, so we evaluate it first.
    if len(losses) >= window:
        win = [float(x) for x in losses[-window:]]
        win_mean = sum(win) / len(win)
        if win_mean > 0 and last_f >= win_mean * 10:
            return {
                "anomaly": "gradient_explosion",
                "reason": f"loss exploded to {last_f:.2f} (window mean {win_mean:.2f})",
                "value": last_f,
                "window_mean": round(win_mean, 4),
            }

    # 4. Overfitting / val-train divergence — val_loss diverging upward
    #    while train_loss continues to drop. Indicates the model is
    #    memorising the train set; recovery is dropout + early stop.
    val_pairs = [
        (m.get("loss"), m.get("val_loss"))
        for m in metrics
        if m.get("loss") is not None and m.get("val_loss") is not None
    ]
    if len(val_pairs) >= 3:
        recent_vals = [float(v) for _, v in val_pairs[-3:]]
        recent_trains = [float(t) for t, _ in val_pairs[-3:]]
        val_trend = recent_vals[-1] - recent_vals[0]
        train_trend = recent_trains[-1] - recent_trains[0]
        # Val going up by ≥10% while train going down: classic overfitting.
        if val_trend > 0 and recent_vals[0] > 0 and (val_trend / recent_vals[0]) >= 0.10 and train_trend < 0:
            return {
                "anomaly": "overfitting",
                "reason": (
                    f"val_loss rose {val_trend:+.3f} ({(val_trend / recent_vals[0]) * 100:.1f}%) "
                    f"while train_loss fell {train_trend:+.3f}"
                ),
                "val_trend": round(val_trend, 4),
                "train_trend": round(train_trend, 4),
            }
        # Val/train gap widening even when both decrease — early sign of
        # catastrophic forgetting on a domain-shift dataset.
        gap_now = recent_vals[-1] - recent_trains[-1]
        gap_then = recent_vals[0] - recent_trains[0]
        if gap_then > 0 and gap_now > gap_then * 1.5 and gap_now > 0.5:
            return {
                "anomaly": "val_train_divergence",
                "reason": f"val/train gap widened from {gap_then:.3f} to {gap_now:.3f}",
                "gap_now": round(gap_now, 4),
                "gap_then": round(gap_then, 4),
            }

    # 5. Plateau — no improvement over 2*window steps.
    if len(losses) >= window * 2:
        recent = [float(x) for x in losses[-window:]]
        prior = [float(x) for x in losses[-2 * window:-window]]
        if min(recent) >= min(prior) and (sum(recent) / window) >= (sum(prior) / window) * 0.99:
            return {
                "anomaly": "loss_no_decrease",
                "reason": f"no decrease over {window} steps",
                "recent_mean": sum(recent) / window,
                "prior_mean": sum(prior) / window,
            }

    # 6. Spike — fallback, weaker than gradient_explosion.
    if len(losses) >= 3:
        try:
            prev = float(losses[-2])
        except (TypeError, ValueError):
            prev = 0.0
        if prev > 0 and last_f > prev * 3:
            return {"anomaly": "loss_spike", "reason": "loss tripled", "value": last_f}

    return {"anomaly": None, "reason": "ok"}
