# services/ops/nlp/jsonl_ops.py
from typing import List, Dict, Any, Tuple
import pandas as pd
import json
import base64
OP_REGISTRY = {}

def csv_to_jsonl(df: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    records = df.to_dict(orient="records")
    return pd.DataFrame({"jsonl": [json.dumps(r) for r in records]}), {"converted": "csv_to_jsonl"}, []

def json_to_jsonl(df: pd.DataFrame, field: str = "json", **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    df = df.copy()
    df['jsonl'] = df[field].apply(json.dumps)
    return df, {"converted": "json_to_jsonl"}, []

def multi_column_to_prompt_completion(df: pd.DataFrame, prompt_cols: List[str] = None, completion_cols: List[str] = None, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    df = df.copy()
    if prompt_cols:
        df['prompt'] = df[prompt_cols].astype(str).agg(' '.join, axis=1)
    if completion_cols:
        df['completion'] = df[completion_cols].astype(str).agg(' '.join, axis=1)
    return df, {"prompt_completion_cols": {"prompt": prompt_cols, "completion": completion_cols}}, []

def escape_unsafe_characters(df: pd.DataFrame, field: str = "text", **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    df = df.copy()
    df[field] = df[field].astype(str).replace(['\\', '"', "'"], ['\\\\', '\\"', "\\'"])
    return df, {"escaped": field}, []

def validate_each_jsonl_line(df: pd.DataFrame, field: str = "jsonl", **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    warns = []
    df = df.copy()
    def validate(line):
        try:
            json.loads(line)
            return line
        except:
            warns.append("Invalid JSON line")
            return '{}'
    df[field] = df[field].apply(validate)
    return df, {"validated_lines": len(df) - len(warns)}, warns

def base64_safe_encoding(df: pd.DataFrame, field: str = "text", **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    df = df.copy()
    df[field] = df[field].astype(str).apply(lambda x: base64.b64encode(x.encode()).decode())
    return df, {"base64_encoded": field}, []

def cleaning_invalid_utf8(df: pd.DataFrame, field: str = "text", **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    df = df.copy()
    df[field] = df[field].astype(str).encode('utf-8', errors='replace').decode('utf-8')
    return df, {"utf8_cleaned": field}, []

OP_REGISTRY = {
    "csv_to_jsonl": csv_to_jsonl,
    "json_to_jsonl": json_to_jsonl,
    "multi_column_to_prompt_completion": multi_column_to_prompt_completion,
    "escape_unsafe_characters": escape_unsafe_characters,
    "validate_each_jsonl_line": validate_each_jsonl_line,
    "base64_safe_encoding": base64_safe_encoding,
    "cleaning_invalid_utf8": cleaning_invalid_utf8,
}