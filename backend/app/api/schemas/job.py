from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel, Field


JobStatus = Literal["queued", "running", "completed", "failed", "stopped"]


class JobMetric(BaseModel):
    step: int
    epoch: float
    loss: Optional[float] = None
    val_loss: Optional[float] = None
    learning_rate: Optional[float] = None
    timestamp: datetime


class JobRecord(BaseModel):
    model_config = {"protected_namespaces": ()}
    id: str
    pipeline_id: str
    status: JobStatus = "queued"
    current_epoch: int = 0
    total_epochs: int = 0
    current_step: int = 0
    total_steps: int = 0
    progress_pct: float = 0.0
    current_loss: Optional[float] = None
    val_loss: Optional[float] = None
    error_message: Optional[str] = None
    model_output_path: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime
    metrics: list[JobMetric] = Field(default_factory=list)


class JobStartRequest(BaseModel):
    pipeline_id: str


class JobStartResponse(BaseModel):
    job_id: str
    status: JobStatus
