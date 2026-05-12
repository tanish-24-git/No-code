"""Clarification tools. The agent now designs free-form questions;
replies are parsed and validated via these tools."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.api.schemas.session import ClarificationAnswer, ClarificationQuestion
from app.tools.registry import ToolContext, tool


@tool(
    name="clarify.parse_reply",
    description="Validate a user's reply against a question's expected kind and option set.",
    input_schema={
        "type": "object",
        "properties": {
            "question": {"type": "object"},
            "reply": {},
        },
        "required": ["question", "reply"],
    },
)
async def clarify_parse_reply(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    q_raw = args["question"]
    reply = args["reply"]
    try:
        q = ClarificationQuestion(**q_raw)
    except Exception as e:
        return {"error": f"invalid question payload: {e}"}

    if q.kind == "single_choice":
        if not isinstance(reply, str):
            return {"valid": False, "error": "expected string"}
        if q.options and reply not in q.options:
            return {"valid": False, "error": f"not in options: {q.options}"}
    elif q.kind == "multi_choice":
        if not isinstance(reply, list) or not all(isinstance(x, str) for x in reply):
            return {"valid": False, "error": "expected list of strings"}
        if q.options:
            bad = [x for x in reply if x not in q.options]
            if bad:
                return {"valid": False, "error": f"unknown options: {bad}"}
    elif q.kind == "yes_no":
        if reply not in ("yes", "no", True, False):
            return {"valid": False, "error": "expected yes|no"}
    elif q.kind == "text":
        if not isinstance(reply, str) or not reply.strip():
            return {"valid": False, "error": "expected non-empty text"}
    answer = ClarificationAnswer(
        question_id=q.question_id,
        value=reply,
        answered_at=datetime.now(timezone.utc),
    )
    return {"valid": True, "answer": answer.model_dump(mode="json")}
