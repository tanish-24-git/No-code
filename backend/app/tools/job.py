"""Job control tools. The execution agent never starts jobs without an
explicit PipelineApproved event upstream — this is enforced in the agent,
not here, but `start` is marked requires_approval for documentation."""
from __future__ import annotations

from typing import Any

from app.services import job_service
from app.tools.registry import ToolContext, tool


@tool(
    name="job.start",
    description="Kick off a training job for an approved pipeline.",
    input_schema={
        "type": "object",
        "properties": {"pipeline_id": {"type": "string"}},
        "required": ["pipeline_id"],
    },
    side_effect="write_resource",
    requires_approval=True,
    cost_class="expensive",
)
async def job_start(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    try:
        job = job_service.create(args["pipeline_id"])
        job_service.run_in_background(job.id)
        return {"job_id": job.id, "status": "queued"}
    except ValueError as e:
        return {"error": str(e)}


@tool(
    name="job.stop",
    description="Request a soft stop on a running job.",
    input_schema={
        "type": "object",
        "properties": {"job_id": {"type": "string"}},
        "required": ["job_id"],
    },
    side_effect="write_resource",
)
async def job_stop(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    ok = job_service.request_stop(args["job_id"])
    return {"stopped_requested": ok}
