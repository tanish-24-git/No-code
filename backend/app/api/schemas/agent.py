"""Agent chat schemas."""
from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    pipeline_id: Optional[str] = None
    dataset_id: Optional[str] = None


class ApplyConfigRequest(BaseModel):
    pipeline_id: str
    config: dict[str, Any]
    reasoning: dict[str, str] = Field(default_factory=dict)
