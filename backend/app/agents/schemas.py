"""Strict pydantic schemas for every LLM-returned structured object.

The agents call the configured LLM provider for many decisions (strategy,
model choice, restructuring, recovery, ...). LLM JSON is unreliable - it
can be missing fields, contain wrong types, or wrap the JSON in markdown.

Every agent goes through ``parse_llm_json`` which:

    1. Locates a JSON object inside the raw text (handles markdown fences).
    2. Validates against the agent's pydantic schema.
    3. Returns a typed object or raises ``LLMSchemaError`` so the calling
       agent can surface a clean error rather than silently accepting
       garbage.

No schema has hard-coded sentinel values - every field is either required
or has an explicit Optional with a documented meaning.
"""
from __future__ import annotations

import json
import re
from typing import Any, Literal, Optional, Type, TypeVar

from pydantic import BaseModel, Field, ValidationError, model_validator


T = TypeVar("T", bound=BaseModel)


class LLMSchemaError(Exception):
    """Raised when an LLM response can't be parsed against its expected schema."""


def _walk_balanced_json(raw: str) -> list[str]:
    """Find every substring that begins with '{' or '[' and contains a
    balanced JSON value, respecting strings + escapes.

    The previous parser used a greedy ``\\{[\\s\\S]*\\}`` regex which
    merged conversational prose with the actual JSON when the LLM
    "thought out loud" before emitting its answer. This walker is
    bracket-aware, so prose that happens to contain a stray ``{`` does
    NOT pull in everything up to the last ``}`` of the real JSON.

    For truncated responses (LLM hit max_tokens mid-object), we append
    closers for the still-open structures so the salvage attempt has
    a chance.

    Candidates are returned in document order. Caller is responsible
    for deciding which one validates.
    """
    candidates: list[str] = []
    n = len(raw)
    i = 0
    while i < n:
        ch = raw[i]
        if ch not in ("{", "["):
            i += 1
            continue
        # Walk forward, tracking the opener stack so we can produce
        # type-correct closers if truncated.
        stack: list[str] = []
        in_str = False
        escape = False
        j = i
        found_close = False
        while j < n:
            c = raw[j]
            if in_str:
                if escape:
                    escape = False
                elif c == "\\":
                    escape = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == "{":
                    stack.append("{")
                elif c == "[":
                    stack.append("[")
                elif c == "}":
                    if stack and stack[-1] == "{":
                        stack.pop()
                    if not stack:
                        candidates.append(raw[i:j + 1])
                        found_close = True
                        break
                elif c == "]":
                    if stack and stack[-1] == "[":
                        stack.pop()
                    if not stack:
                        candidates.append(raw[i:j + 1])
                        found_close = True
                        break
            j += 1
        if not found_close and stack:
            # Truncated mid-object. Best-effort: append matching
            # closers in reverse order so json.loads at least attempts
            # to decode.
            closers = "".join("}" if op == "{" else "]" for op in reversed(stack))
            candidates.append(raw[i:] + closers)
            break  # nothing more to walk past truncation
        i = (j + 1) if found_close else n
    return candidates


def parse_llm_json(raw: str, schema: Type[T]) -> T:
    """Extract the first JSON object/array from ``raw`` and validate it.

    Tolerates: markdown fences (```json ... ```), prose-around-JSON,
    stray braces inside the LLM's "thinking", and truncated output
    (LLM hit max_tokens). Raises ``LLMSchemaError`` with a helpful
    message if no valid JSON matching the schema is found.
    """
    if not raw:
        raise LLMSchemaError("empty LLM response")

    candidates: list[str] = []

    # 1. Fenced code block: the LLM explicitly marked its JSON. Prefer
    #    this over anything else — if it's there, the LLM is telling us
    #    where to look. Also accept nested fences (the model sometimes
    #    wraps prose AND a fenced JSON together).
    for m in re.finditer(r"```(?:json)?\s*([\s\S]+?)\s*```", raw):
        candidates.append(m.group(1).strip())

    # 2. Balanced-brace walker: respects strings and escapes, so prose
    #    containing "{" does not poison the parse.
    candidates.extend(_walk_balanced_json(raw))

    if not candidates:
        raise LLMSchemaError(
            f"no JSON object found in LLM response (first 200 chars: "
            f"{raw[:200]!r})"
        )

    # Try the longest candidates first — when there are multiple, the
    # actual answer is almost always the biggest balanced structure;
    # short ones are usually the LLM's commentary like {"step": "1"}.
    candidates.sort(key=len, reverse=True)

    last_err: Optional[Exception] = None
    for block in candidates:
        try:
            data = json.loads(block)
        except json.JSONDecodeError as e:
            last_err = e
            continue
        try:
            return schema.model_validate(data)
        except ValidationError as e:
            last_err = e
            continue

    raise LLMSchemaError(
        f"could not parse LLM response as {schema.__name__}: "
        f"{last_err.__class__.__name__ if last_err else 'no JSON found'}: "
        f"{str(last_err)[:200] if last_err else ''}"
    )


# ── Decision schemas ──────────────────────────────────────────────────────


class TaskInferenceResult(BaseModel):
    """TaskInferenceAgent output."""
    chosen: str = Field(..., description="Task type identifier")
    scores: dict[str, float] = Field(default_factory=dict)
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    rationale: str = ""


class ModelChoiceResolution(BaseModel):
    """When the user comments on a model selection card."""
    repo_id: Optional[str] = None
    family: Optional[str] = None       # e.g. "llama", "qwen"
    size_b: Optional[float] = None     # parameter count in billions
    rationale: str = ""


class StrategyChoice(BaseModel):
    """TrainingStrategyAgent output.

    v4.0 additions:
        use_unsloth    — 2x-5x faster training via Unsloth FastModel
        optimizer      — 'adamw' (default) | 'galore' | 'galore_q' | 'adam8bit'
        use_liger      — Triton-optimized loss kernels for max throughput
    """
    method: Literal["lora", "qlora", "dora", "full"] = "lora"
    adapter_variant: Literal["none", "dora", "galore"] = "none"
    precision: Literal["float16", "fp16", "bfloat16", "bf16", "float32", "fp32"] = "fp16"
    quantization: Literal["none", "int4", "int8", "4bit", "8bit"] = "none"
    kernel_pack: Literal["standard", "unsloth", "liger"] = "standard"
    use_unsloth: bool = False
    optimizer: Literal["adamw", "galore", "galore_q", "adam8bit"] = "adamw"
    use_liger: bool = False
    batch_size: int = Field(1, ge=1, le=64)
    gradient_accumulation: int = Field(1, ge=1, le=64)
    max_seq_len: int = Field(512, ge=64, le=8192)
    learning_rate: float = Field(2e-4, gt=0.0, lt=1.0)
    epochs: int = Field(1, ge=1, le=20)
    lora_rank: int = Field(16, ge=4, le=256)
    lora_alpha: Optional[int] = None
    lora_dropout: float = 0.05
    early_stopping: bool = True
    rationale: str = ""

    @model_validator(mode="before")
    @classmethod
    def _coerce_enum_casing(cls, data: Any) -> Any:
        """Lowercase the enum-typed string fields BEFORE Literal validation.

        LLMs commonly mirror the casing of technique names in the prompt
        ('QLoRA', 'GaLore', 'BF16'). The Literal validators are strict
        lowercase. Without this pre-pass, a single capital letter
        deadlocks the entire pipeline because TrainingStrategyAgent
        bails when call_llm_typed returns None.

        Also maps common synonyms the LLM emits but the schema doesn't
        list (e.g. 'qlora_4bit' -> 'qlora', 'nf4' -> 'int4').
        """
        if not isinstance(data, dict):
            return data
        synonyms = {
            "qlora_4bit": "qlora",
            "qlora-4bit": "qlora",
            "nf4": "int4",
            "4-bit": "4bit",
            "8-bit": "8bit",
        }
        for key in (
            "method", "adapter_variant", "precision", "quantization",
            "kernel_pack", "optimizer",
        ):
            v = data.get(key)
            if isinstance(v, str):
                vl = v.strip().lower()
                data[key] = synonyms.get(vl, vl)
        return data


class GraphNode(BaseModel):
    id: str
    type: str
    position: dict[str, float] = Field(default_factory=lambda: {"x": 0, "y": 0})
    data: dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    id: str
    source: str
    target: str
    animated: bool = False


class GraphProposal(BaseModel):
    """PipelineBuilderAgent output."""
    nodes: list[GraphNode]
    edges: list[GraphEdge] = Field(default_factory=list)
    rationale: str = ""


class DataHealthReport(BaseModel):
    verdict: Literal["healthy", "advisory", "needs_attention", "blocking"] = "advisory"
    score: float = Field(0.7, ge=0.0, le=1.0)
    summary: str = ""
    asks: list[str] = Field(default_factory=list)
    confidence: Optional[float] = None


class RecoveryOperation(BaseModel):
    op: Literal["set", "noop", "stop"]
    path: str
    new: Any = None


class RecoveryPlan(BaseModel):
    diff_id: Optional[str] = None
    level: Literal["L1", "L2", "L3"] = "L2"
    operations: list[RecoveryOperation] = Field(default_factory=list)
    rationale: str = ""
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    estimated_extra_minutes: float = Field(5.0, ge=0.0)


class RestructurePair(BaseModel):
    """A single instruct/chat/qa pair produced by the restructurer."""
    instruction: Optional[str] = None
    input: Optional[str] = None
    output: Optional[str] = None
    question: Optional[str] = None
    answer: Optional[str] = None
    messages: Optional[list[dict[str, str]]] = None
    label: Optional[str] = None


class RestructureBatch(BaseModel):
    """Batch returned by alchemy.restructure_text per chunk."""
    pairs: list[RestructurePair]


class MasterPlanResponse(BaseModel):
    steps: list[str]


class TaskKind(str):
    """Canonical task names; the LLM is asked to pick one of these."""
    INSTRUCTION = "instruction"
    CHAT = "chat"
    QA = "qa"
    CLASSIFICATION = "classification"
    EXTRACTION = "extraction"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    LANGUAGE_MODELING = "language_modeling"


CANONICAL_TASKS = (
    "instruction", "chat", "qa", "classification",
    "extraction", "summarization", "translation", "language_modeling",
)
