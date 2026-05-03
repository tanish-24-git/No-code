"""Tools the agent can call. Each tool has a JSON schema (provider-agnostic)
and a Python callable that runs server-side. Both Anthropic and OpenAI use
the same schema shape with minor naming differences, normalised here."""
from __future__ import annotations

import json
from typing import Any, Callable

from app.services import dataset_service, inference_service, pipeline_service
from app.storage import store
from app.utils.hardware import detect_hardware


# ── Tool implementations ───────────────────────────────────────────────────

def _list_inferences(_: dict[str, Any]) -> Any:
    return [r.model_dump(mode="json") for r in inference_service.list_all()]


def _get_inference(args: dict[str, Any]) -> Any:
    rec = inference_service.get(args["inference_id"])
    return rec.model_dump(mode="json") if rec else {"error": "not found"}


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


def _suggest_inference_metrics(args: dict[str, Any]) -> Any:
    record_id = args["inference_id"]
    raw = store.read("inferences", record_id)
    if not raw:
        return {"error": "inference not found"}
    raw["suggested_metrics"] = {**raw.get("suggested_metrics", {}), **args.get("metrics", {})}
    if args.get("reasoning"):
        raw["metrics_reasoning"] = {**raw.get("metrics_reasoning", {}), **args.get("reasoning", {})}
    store.write("inferences", record_id, raw)
    return {"ok": True, "suggested_metrics": raw["suggested_metrics"]}


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
        "list_inferences",
        "List all inference endpoints the user has registered (Ollama, OpenAI-compatible, HF, Anthropic). Returns name, kind, base_url, default model, and last reachability probe.",
        {"type": "object", "properties": {}},
        _list_inferences,
    ),
    Tool(
        "get_inference",
        "Get full details of a single registered inference endpoint by id.",
        {
            "type": "object",
            "properties": {"inference_id": {"type": "string"}},
            "required": ["inference_id"],
        },
        _get_inference,
    ),
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
    Tool(
        "suggest_inference_metrics",
        "Recommend generation metrics for an inference endpoint. The metrics dict is saved on the endpoint record. Use after get_inference and get_hardware. Typical keys: max_tokens, temperature, top_p, top_k, num_ctx, num_thread, stop, frequency_penalty, presence_penalty.",
        {
            "type": "object",
            "properties": {
                "inference_id": {"type": "string"},
                "metrics": {"type": "object"},
                "reasoning": {"type": "object", "additionalProperties": {"type": "string"}},
            },
            "required": ["inference_id", "metrics"],
        },
        _suggest_inference_metrics,
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


SYSTEM_PROMPT = """You are FineTune Studio's pipeline + inference copilot.

You help the user:
- choose base models and hyperparameters for fine-tuning,
- understand and tune their LOCAL inference endpoints (Ollama, OpenAI-
  compatible servers, HF Inference, etc.),
- pick generation parameters (max_tokens, temperature, top_p, num_ctx,
  num_thread, stop sequences) appropriate to the endpoint and use case,
- diagnose mismatches between training data and inference settings.

Use the available tools liberally. Always inspect what the user actually has
(list_inferences, get_hardware, get_dataset) before giving recommendations.
When you suggest config, justify each non-default value briefly. If the user
has an active pipeline, you may call suggest_pipeline_config to write your
recommendation back to that pipeline.

Keep responses concise; prefer short tables to walls of prose.
"""
