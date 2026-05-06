"""Evaluation tools. The current implementation reports placeholders pulled
from the job's last metrics; swap this for a real eval suite when you wire
one in (e.g. lm-eval-harness, custom holdout)."""
from __future__ import annotations

from typing import Any

from app.services import job_service
from app.tools.registry import ToolContext, tool


@tool(
    name="eval.run_suite",
    description="Run an evaluation suite against the trained artifact. Returns aggregated metrics.",
    input_schema={
        "type": "object",
        "properties": {
            "job_id": {"type": "string"},
            "suite":  {"type": "string", "default": "basic"},
        },
        "required": ["job_id"],
    },
)
async def eval_run_suite(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    job = job_service.get(args["job_id"])
    if not job:
        return {"error": "job not found"}
    last = job.metrics[-1] if job.metrics else None
    return {
        "suite": args.get("suite", "basic"),
        "final_loss": last.loss if last else None,
        "training_steps": job.current_step,
        "epochs_completed": job.current_epoch,
        "score": _score_from_loss(last.loss if last else None),
        "model_output_path": job.model_output_path,
    }


@tool(
    name="eval.compare_baseline",
    description="Compare the trained model to a base-model baseline. Returns delta scores.",
    input_schema={
        "type": "object",
        "properties": {
            "trained_score": {"type": "number"},
            "baseline_score": {"type": "number"},
        },
        "required": ["trained_score"],
    },
)
async def eval_compare_baseline(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    trained = float(args["trained_score"])
    baseline = float(args.get("baseline_score") or 0.5)
    return {
        "trained": trained,
        "baseline": baseline,
        "delta": round(trained - baseline, 4),
        "improved": trained > baseline,
    }


def _score_from_loss(loss: float | None) -> float | None:
    if loss is None:
        return None
    # Bounded transform so a UI gauge has something to show.
    import math
    return round(1.0 / (1.0 + math.exp(loss - 1.0)), 4)
