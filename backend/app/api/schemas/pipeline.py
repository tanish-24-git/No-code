"""Pipeline schemas. A pipeline is a node graph (visual) + a flat 22-field
config (training). Both are user-editable; the agent fills in the config."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


class NodeGraph(BaseModel):
    nodes: list[dict[str, Any]] = Field(default_factory=list)
    edges: list[dict[str, Any]] = Field(default_factory=list)
    viewport: dict[str, float] = Field(default_factory=lambda: {"x": 0, "y": 0, "zoom": 1})


class PipelineConfig(BaseModel):
    project_name: str = "untitled"
    description: Optional[str] = None
    tags: list[str] = Field(default_factory=list)

    dataset_id: Optional[str] = None
    target_column: Optional[str] = None
    input_columns: list[str] = Field(default_factory=list)
    split_ratio: float = Field(0.8, ge=0.5, le=0.95)

    task_type: Literal["Classification", "Chat", "QA", "Extraction"] = "Chat"
    output_type: Literal["JSON", "text", "label", "multi-label"] = "text"
    domain: Literal["General", "Finance", "Medical", "Legal", "Code"] = "General"
    language: str = "en"

    training_mode: Literal["fast", "balanced", "high_quality"] = "balanced"
    training_method: Literal["lora", "qlora", "full"] = "lora"
    base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    epochs: int = Field(3, ge=1, le=100)
    batch_size: int = Field(4, ge=1, le=128)
    learning_rate: float = Field(2e-4, ge=1e-6, le=1e-1)
    max_seq_len: int = Field(512, ge=64, le=8192)
    lora_rank: Literal[8, 16, 32, 64] = 16
    gradient_accumulation: int = Field(4, ge=1, le=64)
    precision: Literal["bf16", "fp16", "float32"] = "fp16"

    early_stopping: bool = True
    class_balancing: bool = False
    data_augmentation: bool = False
    resume_checkpoint: Optional[str] = None


class PipelineRecord(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    node_graph: NodeGraph = Field(default_factory=NodeGraph)
    config: PipelineConfig = Field(default_factory=PipelineConfig)
    is_agent_configured: bool = False
    reasoning: dict[str, str] = Field(default_factory=dict)  # field_name -> agent's why
    created_at: datetime
    updated_at: datetime


class PipelineCreate(BaseModel):
    name: str
    description: Optional[str] = None
    dataset_id: Optional[str] = None


class PipelineUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    node_graph: Optional[NodeGraph] = None
    config: Optional[PipelineConfig] = None
