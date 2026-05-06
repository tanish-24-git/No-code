"""Task-type inference tools. The classifier here is deterministic; the
TaskInferenceAgent may wrap an LLM around these signals to produce a final
choice with rationale."""
from __future__ import annotations

from typing import Any

from app.tools.registry import ToolContext, tool


_TASK_TYPES = ("chat", "instruction", "classification", "extraction", "qa")


@tool(
    name="task.classify",
    description=(
        "Heuristic classifier over dataset field shape: returns a probability "
        "distribution over task types {chat, instruction, classification, "
        "extraction, qa}. Pure function over field metadata."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "field_buckets": {"type": "object"},
            "column_types": {"type": "object"},
            "row_count": {"type": "integer"},
            "imbalance": {"type": "object"},
        },
        "required": ["field_buckets"],
    },
)
async def task_classify(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    buckets = args.get("field_buckets") or {}
    imbalance = args.get("imbalance") or {}
    has_instr = bool(buckets.get("instruction_like"))
    has_input = bool(buckets.get("input_like"))
    has_output = bool(buckets.get("output_like"))
    has_label = bool(buckets.get("label_like")) or bool(imbalance.get("label_field"))

    scores = {k: 0.1 for k in _TASK_TYPES}
    if has_instr and has_output:
        scores["instruction"] += 0.5
    if has_input and has_output and has_instr:
        scores["instruction"] += 0.2
    if has_input and has_output and not has_instr:
        scores["chat"] += 0.4
    if has_label and not has_output:
        scores["classification"] += 0.6
    if has_input and has_output and "answer" in [c.lower() for c in buckets.get("output_like", [])]:
        scores["qa"] += 0.4
        scores["instruction"] -= 0.1
    if has_input and has_label and "extract" in str(buckets).lower():
        scores["extraction"] += 0.3

    # Normalise.
    total = sum(scores.values()) or 1.0
    probs = {k: round(v / total, 4) for k, v in scores.items()}
    chosen = max(probs, key=lambda k: probs[k])
    return {
        "scores": probs,
        "chosen": chosen,
        "confidence": probs[chosen],
        "missing_signals": _missing_signals(buckets, imbalance),
    }


@tool(
    name="task.score_candidates",
    description="Score candidate task types against given field assignments. Lighter than task.classify; takes explicit assignments.",
    input_schema={
        "type": "object",
        "properties": {
            "input_field": {"type": "string"},
            "output_field": {"type": "string"},
            "label_field": {"type": "string"},
            "instruction_field": {"type": "string"},
        },
    },
)
async def task_score_candidates(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    scores = {k: 0.0 for k in _TASK_TYPES}
    if args.get("instruction_field") and args.get("output_field"):
        scores["instruction"] += 0.7
    if args.get("input_field") and args.get("output_field") and not args.get("instruction_field"):
        scores["chat"] += 0.6
    if args.get("label_field") and not args.get("output_field"):
        scores["classification"] += 0.7
    total = sum(scores.values()) or 1.0
    probs = {k: round(v / total, 4) for k, v in scores.items()}
    chosen = max(probs, key=lambda k: probs[k])
    return {"scores": probs, "chosen": chosen, "confidence": probs[chosen]}


def _missing_signals(buckets: dict[str, list[str]], imbalance: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    if not buckets.get("output_like") and not imbalance.get("label_field"):
        missing.append("target_field")
    if not buckets.get("instruction_like") and not buckets.get("input_like"):
        missing.append("input_field")
    if not buckets.get("output_like") and not buckets.get("label_like"):
        missing.append("user_goal")
    return missing
