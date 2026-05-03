from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel


class DatasetSchema(BaseModel):
    id: str
    name: str
    file_path: str
    file_type: str  # csv | json | jsonl
    row_count: int
    column_names: list[str]
    column_types: dict[str, str]
    sample_rows: list[dict[str, Any]] = []
    size_bytes: int
    is_analyzed: bool = False
    analysis: Optional[dict[str, Any]] = None
    created_at: datetime
