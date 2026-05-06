"""Audit tool: agents call this to durably record a decision."""
from __future__ import annotations

from typing import Any

from app.services import decision_log
from app.tools.registry import ToolContext, tool


@tool(
    name="audit.write",
    description="Record a structured decision for the audit trail.",
    input_schema={
        "type": "object",
        "properties": {
            "agent": {"type": "string"},
            "kind":  {"type": "string"},
            "inputs": {"type": "object"},
            "candidates": {"type": "array"},
            "chosen": {},
            "confidence": {"type": "number"},
            "rationale": {"type": "string"},
            "tool_calls": {"type": "array"},
        },
        "required": ["agent", "kind"],
    },
    side_effect="write_resource",
)
async def audit_write(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    d = decision_log.record(
        session_id=ctx.session_id,
        agent=args["agent"],
        kind=args["kind"],
        inputs=args.get("inputs") or {},
        candidates=args.get("candidates") or [],
        chosen=args.get("chosen"),
        confidence=float(args.get("confidence") or 0.0),
        rationale=args.get("rationale") or "",
        tool_calls=args.get("tool_calls") or [],
    )
    return {"decision_id": d.id}


@tool(
    name="audit.read",
    description="Read all decisions recorded for the current session.",
    input_schema={"type": "object", "properties": {}},
)
async def audit_read(_args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    decisions = decision_log.list_for_session(ctx.session_id)
    return {"decisions": [d.model_dump(mode="json") for d in decisions]}
