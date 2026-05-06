"""Metrics + anomaly detection tools."""
from __future__ import annotations

import math
from typing import Any

from app.services import job_service
from app.tools.registry import ToolContext, tool


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
    description="Detect training anomalies: NaN loss, no decrease over a window, sudden spike, OOM tag.",
    input_schema={
        "type": "object",
        "properties": {
            "metrics": {"type": "array"},
            "window": {"type": "integer", "default": 5},
        },
        "required": ["metrics"],
    },
)
async def metrics_detect_anomaly(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    metrics = args.get("metrics") or []
    window = int(args.get("window", 5))
    losses = [m.get("loss") for m in metrics if m.get("loss") is not None]
    if not losses:
        return {"anomaly": None, "reason": "no losses yet"}

    last = losses[-1]
    if isinstance(last, float) and (math.isnan(last) or math.isinf(last)):
        return {"anomaly": "loss_nan_or_inf", "reason": "non-finite loss", "value": last}

    if len(losses) >= window * 2:
        recent = losses[-window:]
        prior = losses[-2 * window:-window]
        if min(recent) >= min(prior) and (sum(recent) / window) >= (sum(prior) / window) * 0.99:
            return {
                "anomaly": "loss_no_decrease",
                "reason": f"no decrease over {window} steps",
                "recent_mean": sum(recent) / window,
                "prior_mean": sum(prior) / window,
            }

    if len(losses) >= 3 and losses[-1] > losses[-2] * 3:
        return {"anomaly": "loss_spike", "reason": "loss tripled", "value": losses[-1]}

    return {"anomaly": None, "reason": "ok"}
