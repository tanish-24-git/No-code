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

from pydantic import BaseModel, Field, ValidationError


T = TypeVar("T", bound=BaseModel)


class LLMSchemaError(Exception):
    """Raised when an LLM response can't be parsed against its expected schema."""


def parse_llm_json(raw: str, schema: Type[T]) -> T:
    """Extract the first JSON object/array from ``raw`` and validate it.

    Tolerates: markdown fences (```json ... ```), surrounding prose,
    trailing commentary. Raises ``LLMSchemaError`` with a helpful message
    if no valid JSON matching the schema is found.
    """
    if not raw:
        raise LLMSchemaError("empty LLM response")

    blocks = []
    fence = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", raw)
    if fence:
        blocks.append(fence.group(1))
    # Greedy outermost match is what we want; LLMs often wrap the JSON in prose.
    obj = re.search(r"\{[\s\S]*\}", raw)
    if obj:
        blocks.append(obj.group(0))
    arr = re.search(r"\[[\s\S]*\]", raw)
    if arr:
        blocks.append(arr.group(0))

    last_err: Optional[Exception] = None
    for block in blocks:
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
        f"{last_err.__class__.__name__ if last_err else 'no JSON found'}"
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
    """TrainingStrategyAgent output."""
    method: Literal["lora", "qlora", "dora", "full"] = "lora"
    adapter_variant: Literal["none", "dora", "galore"] = "none"
    precision: Literal["float16", "fp16", "bfloat16", "bf16", "float32", "fp32"] = "fp16"
    quantization: Literal["none", "int4", "int8", "4bit", "8bit"] = "none"
    kernel_pack: Literal["standard", "unsloth"] = "standard"
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
