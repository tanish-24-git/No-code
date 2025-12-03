# models/requests.py
from pydantic import BaseModel, validator
from typing import Optional, List, Dict, Any
from enum import Enum
import json
class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
class MissingStrategy(str, Enum):
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    DROP = "drop"
class EncodingMethod(str, Enum):
    ONEHOT = "onehot"
    LABEL = "label"
    TARGET = "target"
    KFOLD = "kfold"
    PASSTHROUGH = "passthrough"
class ColumnAction(BaseModel):
    action: Optional[str] = "keep" # keep | drop | custom
    missing_strategy: Optional[MissingStrategy] = None
    scaling: Optional[bool] = None
    encoding: Optional[EncodingMethod] = None
    custom_transform: Optional[str] = None # name of allowed custom transform
class PreprocessRequest(BaseModel):
    # global defaults
    global_defaults: Optional[Dict[str, Any]] = {}
    # per-column overrides
    columns: Optional[Dict[str, ColumnAction]] = {}
    # fallback target column if any
    target_column: Optional[str] = None
class TrainRequest(BaseModel):
    task_type: TaskType
    model_type: Optional[str] = None
    target_columns: Optional[Dict[str, str]] = None
    @validator('target_columns', pre=True)
    def parse_target_columns(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except:
                return {}
        return v or {}