# services/ops/nlp/metadata_ops.py
from typing import List, Dict, Any, Tuple
import pandas as pd
import tiktoken
OP_REGISTRY = {}

def tokenizer_name(df: pd.DataFrame, tokenizer: str = "cl100k_base", **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    return df, {"tokenizer_name": tokenizer}, []

def base_model_name(df: pd.DataFrame, model: str = "gpt-3.5-turbo", **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    return df, {"base_model": model}, []

def training_sample_count(df: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    return df, {"training_samples": len(df)}, []

def average_tokens_per_example(df: pd.DataFrame, field: str = "text", **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    encoding = tiktoken.get_encoding("cl100k_base")
    avg_tokens = df[field].astype(str).apply(lambda x: len(encoding.encode(x))).mean()
    return df, {"avg_tokens": avg_tokens}, []

def safety_warnings(df: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    warns = ["Check for PII in production"]
    return df, {"safety_warnings": warns}, warns

def recommended_training_lr_batch_size(df: pd.DataFrame, lr: float = 1e-5, batch_size: int = 8, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    return df, {"recommended_lr": lr, "recommended_batch": batch_size}, []

OP_REGISTRY = {
    "tokenizer_name": tokenizer_name,
    "base_model_name": base_model_name,
    "training_sample_count": training_sample_count,
    "average_tokens_per_example": average_tokens_per_example,
    "safety_warnings": safety_warnings,
    "recommended_training_lr_batch_size": recommended_training_lr_batch_size,
}