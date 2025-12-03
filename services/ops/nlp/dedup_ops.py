# services/ops/nlp/dedup_ops.py
from typing import List, Dict, Any, Tuple
import pandas as pd
from fuzzywuzzy import fuzz
OP_REGISTRY = {}

def exact_duplicate_removal(df: pd.DataFrame, field: str = "text", **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    initial_len = len(df)
    df = df.drop_duplicates(subset=[field])
    dropped = initial_len - len(df)
    return df, {"exact_dedup_dropped": dropped}, []

def near_duplicate_removal(df: pd.DataFrame, field: str = "text", threshold: int = 90, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    initial_len = len(df)
    to_drop = set()
    for i in range(len(df)):
        for j in range(i+1, len(df)):
            if fuzz.ratio(df.iloc[i][field], df.iloc[j][field]) > threshold:
                to_drop.add(j)
    df = df.drop(index=list(to_drop)).reset_index(drop=True)
    dropped = initial_len - len(df)
    return df, {"near_dedup_dropped": dropped, "threshold": threshold}, []

def deduplicate_by_target(df: pd.DataFrame, target_col: str = "label", field: str = "text", **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    initial_len = len(df)
    df = df.drop_duplicates(subset=[target_col, field])
    dropped = initial_len - len(df)
    return df, {"target_dedup_dropped": dropped}, []

def deduplicate_qa_templates(df: pd.DataFrame, q_col: str = "question", a_col: str = "answer", **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    initial_len = len(df)
    df['qa_pair'] = df[q_col].astype(str) + '|' + df[a_col].astype(str)
    df = df.drop_duplicates(subset=['qa_pair']).drop('qa_pair', axis=1)
    dropped = initial_len - len(df)
    return df, {"qa_dedup_dropped": dropped}, []

OP_REGISTRY = {
    "exact_duplicate_removal": exact_duplicate_removal,
    "near_duplicate_removal": near_duplicate_removal,
    "deduplicate_by_target": deduplicate_by_target,
    "deduplicate_qa_templates": deduplicate_qa_templates,
}