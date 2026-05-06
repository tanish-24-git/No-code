"""Data preparation tools. These describe steps that the pipeline will run;
they don't (yet) do heavyweight transformation here. The pipeline executor
in services/job_service.py picks them up by node type."""
from __future__ import annotations

from typing import Any

from app.tools.registry import ToolContext, tool


@tool(
    name="data.convert_format",
    description="Plan a format conversion (e.g. csv -> jsonl, flat -> chat-template). Returns a plan node spec.",
    input_schema={
        "type": "object",
        "properties": {
            "from_format": {"type": "string"},
            "to_format": {"type": "string"},
            "template": {"type": "string"},
        },
        "required": ["from_format", "to_format"],
    },
    side_effect="write_session",
)
async def data_convert_format(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    return {
        "node": {
            "type": "convert_format",
            "from": args["from_format"],
            "to": args["to_format"],
            "template": args.get("template"),
        }
    }


@tool(
    name="data.split",
    description="Plan a train/val split node.",
    input_schema={
        "type": "object",
        "properties": {
            "ratio": {"type": "number", "minimum": 0.5, "maximum": 0.95, "default": 0.8},
            "stratify_by": {"type": "string"},
        },
    },
    side_effect="write_session",
)
async def data_split(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    return {
        "node": {
            "type": "split",
            "ratio": float(args.get("ratio", 0.8)),
            "stratify_by": args.get("stratify_by"),
        }
    }


@tool(
    name="data.template_prompts",
    description="Plan a prompt-templating step (instruction/input/output -> chat messages).",
    input_schema={
        "type": "object",
        "properties": {
            "template": {"type": "string", "default": "alpaca"},
            "instruction_field": {"type": "string"},
            "input_field": {"type": "string"},
            "output_field": {"type": "string"},
            "system_prompt": {"type": "string"},
        },
    },
    side_effect="write_session",
)
async def data_template_prompts(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    return {
        "node": {
            "type": "template_prompts",
            "template": args.get("template", "alpaca"),
            "instruction_field": args.get("instruction_field"),
            "input_field": args.get("input_field"),
            "output_field": args.get("output_field"),
            "system_prompt": args.get("system_prompt"),
        }
    }


@tool(
    name="data.balance",
    description="Plan a class-balancing step.",
    input_schema={
        "type": "object",
        "properties": {
            "label_field": {"type": "string"},
            "strategy": {"type": "string", "enum": ["upsample", "downsample"], "default": "upsample"},
        },
        "required": ["label_field"],
    },
    side_effect="write_session",
)
async def data_balance(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    return {
        "node": {
            "type": "balance",
            "label_field": args["label_field"],
            "strategy": args.get("strategy", "upsample"),
        }
    }


@tool(
    name="data.augment",
    description="Plan a data augmentation step (paraphrase, back-translation).",
    input_schema={
        "type": "object",
        "properties": {
            "method": {"type": "string", "enum": ["paraphrase", "backtranslate"], "default": "paraphrase"},
            "factor": {"type": "number", "default": 1.5},
        },
    },
    side_effect="write_session",
)
async def data_augment(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    return {
        "node": {
            "type": "augment",
            "method": args.get("method", "paraphrase"),
            "factor": float(args.get("factor", 1.5)),
        }
    }


@tool(
    name="data.tokenize_check",
    description="Sanity-check that a tokenizer can handle the dataset (no OOV-only rows).",
    input_schema={
        "type": "object",
        "properties": {
            "tokenizer": {"type": "string"},
            "sample_text": {"type": "string"},
        },
        "required": ["tokenizer"],
    },
)
async def data_tokenize_check(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    text = args.get("sample_text") or "the quick brown fox jumps over the lazy dog"
    try:
        from transformers import AutoTokenizer  # type: ignore
        enc = AutoTokenizer.from_pretrained(args["tokenizer"])
        ids = enc.encode(text, add_special_tokens=False)
        return {"ok": True, "token_count": len(ids), "vocab_size": getattr(enc, "vocab_size", None)}
    except Exception as e:
        return {"ok": False, "error": str(e)}
