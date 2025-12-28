"""
Pydantic schemas for all 22 UI fields matching master prompt specification.
Provides type-safe validation for complete training pipeline configuration.
"""
from pydantic import BaseModel, Field, validator
from typing import Literal, Optional, List, Dict, Any
from datetime import datetime


# ============================================================================
# SCREEN 1: PROJECT SETUP (3 fields)
# ============================================================================

class ProjectCreate(BaseModel):
    """POST /api/v1/projects - Create new project."""
    project_name: str = Field(..., min_length=1, max_length=100, description="Project name (required)")
    description: Optional[str] = Field(None, max_length=500, description="Project description (optional)")
    tags: List[str] = Field(default_factory=list, description="Tags for organization (optional)")


class ProjectResponse(BaseModel):
    """Project response with metadata."""
    project_id: str
    project_name: str
    description: Optional[str]
    tags: List[str]
    created_at: str
    updated_at: str
    job_count: int = 0
    dataset_count: int = 0


# ============================================================================
# SCREEN 2: DATASET CONFIG (5 fields)
# ============================================================================

class DatasetUploadRequest(BaseModel):
    """POST /api/v1/datasets/upload - Dataset upload configuration."""
    dataset_name: str = Field(..., min_length=1, max_length=100, description="Dataset name")
    target_column: str = Field(..., description="Target column (response|label|summary)")
    input_columns: List[str] = Field(..., min_items=1, description="Input columns")
    split_ratio: float = Field(0.8, ge=0.5, le=0.95, description="Train:Test split ratio (0.8 = 80/15/5)")


class DatasetStats(BaseModel):
    """Dataset statistics from DatasetAgent."""
    rows: int
    columns: int
    text_columns: List[str]
    avg_text_length: float
    token_count_estimate: int
    has_missing: bool
    duplicate_ratio: float


# ============================================================================
# SCREEN 3: TASK DEFINITION (4 fields)
# ============================================================================

class TaskDefinition(BaseModel):
    """POST /api/v1/tasks/suggest - Task type configuration."""
    task_type: Literal["classification", "regression", "chat", "summarization", "qa", "extraction"] = Field(
        ..., description="ML task type"
    )
    output_type: Literal["label", "text", "json", "multi-label"] = Field(
        "text", description="Output format"
    )
    domain: Literal["general", "finance", "medical", "legal", "code", "custom"] = Field(
        "general", description="Domain for specialized behavior"
    )
    language: str = Field("en", description="Language code (en|es|fr|de|multi)")


class TaskDetectionResponse(BaseModel):
    """AI task detection result."""
    task_type: str
    confidence: float = Field(..., ge=0, le=1)
    target_column: str
    reasoning: str
    suggested_config: Dict[str, Any]


# ============================================================================
# SCREEN 4: TRAINING CONFIG (7 fields - AI PRE-FILLED)
# ============================================================================

class TrainingConfig(BaseModel):
    """POST /api/v1/jobs - Main training configuration (7 fields)."""
    
    # Training Mode Preset
    training_mode: Literal["fast", "balanced", "high_quality"] = Field(
        "balanced",
        description="Preset: fast(1 epoch), balanced(3 epochs), high_quality(5+ epochs)"
    )
    
    # Model Selection
    base_model: str = Field(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        description="HuggingFace model ID"
    )
    
    # Basic Hyperparameters (AI suggests these)
    epochs: int = Field(3, ge=1, le=10, description="Number of training epochs")
    batch_size: int = Field(4, ge=1, le=16, description="Batch size (auto-scaled by GPU memory)")
    learning_rate: float = Field(2e-4, ge=1e-5, le=5e-4, description="Learning rate")
    
    # Sequence Length
    max_seq_len: int = Field(2048, ge=512, le=4096, description="Maximum sequence length")
    
    # LoRA Configuration
    lora_rank: int = Field(16, ge=8, le=64, description="LoRA rank (r)")
    
    @validator('lora_rank')
    def validate_lora_rank(cls, v):
        """Ensure LoRA rank is power of 2 or common value."""
        valid_ranks = [8, 16, 32, 64]
        if v not in valid_ranks:
            return min(valid_ranks, key=lambda x: abs(x - v))
        return v


class AIConfigSuggestion(BaseModel):
    """Response from POST /ai/suggest-config."""
    batch_size: int
    epochs: int
    lora_rank: int
    lora_alpha: int
    learning_rate: float
    precision: str
    reasoning: str


# ============================================================================
# SCREEN 5: ADVANCED CONFIG (6 optional fields)
# ============================================================================

class AdvancedConfig(BaseModel):
    """Optional advanced training settings."""
    
    gradient_accumulation: int = Field(1, ge=1, le=8, description="Gradient accumulation steps")
    precision: Literal["fp16", "bf16", "float32"] = Field("bf16", description="Training precision")
    early_stopping: bool = Field(True, description="Enable early stopping")
    class_balancing: bool = Field(False, description="Balance classes (for classification)")
    data_augmentation: bool = Field(False, description="Enable data augmentation")
    resume_checkpoint: Optional[str] = Field(None, description="S3 checkpoint path to resume from")


# ============================================================================
# COMPLETE JOB SUBMISSION (ALL 22 FIELDS COMBINED)
# ============================================================================

class JobSubmissionRequest(BaseModel):
    """Complete job submission with all 22 fields."""
    
    # Project (1 field - if not using separate endpoint)
    project_id: Optional[str] = Field(None, description="Link to existing project")
    
    # Dataset (already uploaded, reference by ID)
    dataset_id: str = Field(..., description="Uploaded dataset ID")
    
    # Task (4 fields)
    task: TaskDefinition
    
    # Training (7 fields)
    training: TrainingConfig
    
    # Advanced (6 optional fields)
    advanced: AdvancedConfig = Field(default_factory=AdvancedConfig)
    
    # Auto-filled by AI
    ai_auto_config: bool = Field(True, description="Let AI suggest optimal config")


class JobResponse(BaseModel):
    """Job submission response."""
    job_id: str
    status: Literal["pending", "running", "completed", "failed", "cancelled"]
    created_at: datetime
    pipeline_config: Dict[str, Any]


class JobStatusResponse(BaseModel):
    """GET /jobs/{job_id} - Job status."""
    job_id: str
    status: Literal["pending", "running", "completed", "failed", "cancelled"]
    progress: float = Field(0, ge=0, le=1, description="Progress 0-1")
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    loss: Optional[float] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    artifacts: List[str] = Field(default_factory=list)
    error: Optional[str] = None


# ============================================================================
# TRAINING PRESETS (fast/balanced/high_quality)
# ============================================================================

class TrainingPresetRequest(BaseModel):
    """POST /api/v1/tasks/preset - Get training preset."""
    training_mode: Literal["fast", "balanced", "high_quality"]
    dataset_stats: Dict[str, Any]
    gpu_count: int = Field(0, ge=0, le=8)
    task_type: Optional[str] = None


class TrainingPresetResponse(BaseModel):
    """Training preset response."""
    preset_name: str
    epochs: int
    batch_size: int
    learning_rate: float
    lora_rank: int
    lora_alpha: int
    precision: str
    gradient_accumulation: int
    reasoning: str


# ============================================================================
# EXECUTION CONTROLS
# ============================================================================

class JobActionRequest(BaseModel):
    """POST /jobs/{id}/action - Control job execution."""
    action: Literal["start", "pause", "stop", "resume"]
    save_checkpoint: bool = Field(False, description="Save checkpoint before stopping")


# ============================================================================
# MODEL EXPORT & DOWNLOAD
# ============================================================================

class ModelArtifact(BaseModel):
    """Model artifact metadata."""
    artifact_type: Literal["adapter", "merged", "gguf", "tokenizer", "model_card"]
    filename: str
    size_bytes: int
    s3_path: str
    download_url: Optional[str] = None  # Signed URL (2hr expiry)


class ModelListResponse(BaseModel):
    """GET /models - List all models."""
    models: List[Dict[str, Any]]
    total: int


# ============================================================================
# LIVE METRICS (SSE /logs/stream)
# ============================================================================

class TrainingMetric(BaseModel):
    """Real-time training metric event."""
    job_id: str
    timestamp: datetime
    event_type: Literal["train", "eval", "checkpoint", "error", "complete"]
    step: Optional[int] = None
    epoch: Optional[int] = None
    loss: Optional[float] = None
    learning_rate: Optional[float] = None
    gpu_utilization: Optional[float] = None
    samples_per_sec: Optional[float] = None
    message: Optional[str] = None
