"""Export tools. Local save = move/symlink under models/; HF push wraps the
existing hf_service. The push tool is requires_approval — agents emit a
PushToHFRequested event; only after a SaveLocalRequested/PushToHFRequested
that originates from a user choice do we actually call these."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.services import hf_service
from app.storage import store
from app.tools.registry import ToolContext, tool
from app.utils.config import settings


@tool(
    name="export.save_local",
    description="Register a trained artifact in the local model registry under models/.",
    input_schema={
        "type": "object",
        "properties": {
            "job_id": {"type": "string", "description": "The ID of the training job to export."},
            "name": {"type": "string", "description": "Optional display name for the model."},
        },
        "required": ["job_id"],
    },
    side_effect="write_resource",
)
async def export_save_local(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    job_id = args.get("job_id")
    if not job_id:
        return {"error": "Missing 'job_id'. You can only export a model after run_training has successfully completed."}
    
    raw_job = store.read("jobs", job_id)
    if not raw_job:
        return {"error": f"Job {job_id} not found. Ensure the training job actually started."}
    
    out = raw_job.get("model_output_path")
    if not out or not Path(out).exists():
        return {"error": "No model artifact found for this job. Did training finish successfully?"}

    model_id = f"trained_{job_id}"
    name = args.get("name") or f"trained-{job_id[:8]}"
    rec = {
        "id": model_id,
        "kind": "trained",
        "name": name,
        "job_id": job_id,
        "local_path": str(out),
        "is_pushed_to_hub": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    store.write("models", model_id, rec)
    return {"model_id": model_id, "local_path": str(out)}


@tool(
    name="export.push_hf",
    description="Push a trained model to Hugging Face Hub. Requires HF token.",
    input_schema={
        "type": "object",
        "properties": {
            "model_id": {"type": "string"},
            "repo_id":  {"type": "string"},
        },
        "required": ["model_id", "repo_id"],
    },
    side_effect="external",
    requires_approval=True,
    cost_class="medium",
)
async def export_push_hf(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    model_id = args["model_id"]
    repo_id = args["repo_id"]
    try:
        result = hf_service.start_push(model_id, repo_id)
        return {"status": result.get("status", "pushing"), "repo_id": repo_id, "model_id": model_id}
    except ValueError as e:
        return {"error": str(e)}
