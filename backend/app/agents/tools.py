"""Tools the agent can call. Each tool has a JSON schema (provider-agnostic)
and a Python callable that runs server-side. Both Anthropic and OpenAI use
the same schema shape with minor naming differences, normalised here."""
from __future__ import annotations

import json
from typing import Any, Callable

from app.services import dataset_service, pipeline_service
from app.storage import store
from app.utils.hardware import detect_hardware


# ── Tool implementations ───────────────────────────────────────────────────

def _get_hardware(_: dict[str, Any]) -> Any:
    return detect_hardware()


def _get_dataset(args: dict[str, Any]) -> Any:
    d = dataset_service.get_dataset(args["dataset_id"])
    return d.model_dump(mode="json") if d else {"error": "not found"}


def _list_models(_: dict[str, Any]) -> Any:
    return list(store.list_all("models"))


def _suggest_pipeline_config(args: dict[str, Any]) -> Any:
    rec = pipeline_service.apply_agent_config(
        args["pipeline_id"],
        args.get("config", {}),
        args.get("reasoning", {}),
    )
    return rec.model_dump(mode="json") if rec else {"error": "pipeline not found"}


# ── Tool registry (canonical schema) ───────────────────────────────────────

ToolFn = Callable[[dict[str, Any]], Any]


class Tool:
    __slots__ = ("name", "description", "input_schema", "fn")

    def __init__(self, name: str, description: str, input_schema: dict[str, Any], fn: ToolFn) -> None:
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.fn = fn


TOOLS: list[Tool] = [
    Tool(
        "get_hardware",
        "Detect local hardware (CPU/GPU, VRAM, CUDA). Use before recommending precision, batch size, or quantization.",
        {"type": "object", "properties": {}},
        _get_hardware,
    ),
    Tool(
        "get_dataset",
        "Inspect a dataset's schema, sample rows, and stats. Use before recommending training config.",
        {
            "type": "object",
            "properties": {"dataset_id": {"type": "string"}},
            "required": ["dataset_id"],
        },
        _get_dataset,
    ),
    Tool(
        "list_models",
        "List models in the local registry (pulled base models and trained outputs).",
        {"type": "object", "properties": {}},
        _list_models,
    ),
    Tool(
        "suggest_pipeline_config",
        "Apply a recommended pipeline config to a pipeline. The config object contains any fields you want to set; the server merges them into the existing pipeline config. The reasoning object maps each field name to a short justification. Returns the updated pipeline.",
        {
            "type": "object",
            "properties": {
                "pipeline_id": {"type": "string"},
                "config": {"type": "object"},
                "reasoning": {"type": "object", "additionalProperties": {"type": "string"}},
            },
            "required": ["pipeline_id", "config"],
        },
        _suggest_pipeline_config,
    ),
]


_BY_NAME: dict[str, Tool] = {t.name: t for t in TOOLS}


def run_tool(name: str, args: dict[str, Any]) -> tuple[str, bool]:
    """Execute a tool by name. Returns (json_string, is_error)."""
    tool = _BY_NAME.get(name)
    if not tool:
        return json.dumps({"error": f"unknown tool {name}"}), True
    try:
        result = tool.fn(args or {})
        return json.dumps(result, default=str), False
    except Exception as e:
        return json.dumps({"error": str(e)}), True


SYSTEM_PROMPT = """You are FineTune Studio's pipeline copilot.

You help the user:
- choose base models and hyperparameters for fine-tuning,
- map dataset structure to a sensible training task,
- pick precision, batch size, sequence length, and quantization that fit
  the user's hardware.

Use the available tools liberally. Always inspect what the user actually has
(get_hardware, get_dataset) before giving recommendations. When you suggest
config, justify each non-default value briefly. If the user has an active
pipeline, you may call suggest_pipeline_config to write your recommendation
back to that pipeline.

Keep responses concise; prefer short tables to walls of prose.

CRITICAL: Use ONLY the tools listed above. If you need to return JSON, return it as a raw string in your message. NEVER try to call a tool named 'json' or 'format'.
"""
