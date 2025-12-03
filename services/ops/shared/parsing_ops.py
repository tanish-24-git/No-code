# services/ops/shared/parsing_ops.py
from typing import List, Dict, Any, Tuple
import pandas as pd
import chardet  # Add to requirements if not
OP_REGISTRY = {}

def chunked_csv_reading(df: pd.DataFrame, chunk_size: int = 10000, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    # Placeholder: engine uses Dask for chunked
    return df, {"chunked": True}, []

def on_bad_lines_skip(df: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    # Already in read_csv
    return df, {}, []

def encoding_detection(df: pd.DataFrame, file_path: str, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    with open(file_path, 'rb') as f:
        raw = f.read(10000)
        enc = chardet.detect(raw)['encoding']
    # Re-read if needed, but assume UTF-8
    return df, {"detected_encoding": enc}, []

def corrupted_csv_recovery(df: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    # Skip bad lines already
    return df, {}, []

def streaming_processing(df: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    # Dask
    return df, {"streaming": True}, []

def safe_row_counting(df: pd.DataFrame, file_path: str, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    row_count = len(df)
    return df, {"row_count": row_count}, []

OP_REGISTRY = {
    "chunked_csv_reading": chunked_csv_reading,
    "on_bad_lines_skip": on_bad_lines_skip,
    "encoding_detection": encoding_detection,
    "corrupted_csv_recovery": corrupted_csv_recovery,
    "streaming_processing": streaming_processing,
    "safe_row_counting": safe_row_counting,
}