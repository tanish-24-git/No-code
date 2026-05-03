"""User-registered inference endpoints. The agent sees these and can suggest
generation metrics (max_tokens, temperature, top_p, etc.) tuned to each one."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


InferenceKind = Literal["ollama", "openai_compat", "huggingface_inference", "anthropic"]


class InferenceCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)
    kind: InferenceKind = "openai_compat"
    base_url: str
    api_key: Optional[str] = None
    default_model: Optional[str] = None
    notes: Optional[str] = None


class InferenceUpdate(BaseModel):
    name: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    default_model: Optional[str] = None
    notes: Optional[str] = None


class InferenceProbe(BaseModel):
    reachable: bool
    latency_ms: Optional[float] = None
    models: list[str] = Field(default_factory=list)
    detail: Optional[str] = None


class InferenceRecord(BaseModel):
    id: str
    name: str
    kind: InferenceKind
    base_url: str
    api_key_masked: Optional[str] = None
    default_model: Optional[str] = None
    notes: Optional[str] = None
    last_probe: Optional[InferenceProbe] = None
    suggested_metrics: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class GenerateRequest(BaseModel):
    """Generic test-generate against a registered inference endpoint."""
    inference_id: str
    prompt: str
    model: Optional[str] = None
    max_tokens: int = 256
    temperature: float = 0.7


class GenerateResponse(BaseModel):
    text: str
    model: Optional[str] = None
    latency_ms: float
