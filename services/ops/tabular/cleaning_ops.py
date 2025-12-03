# services/ops/tabular/cleaning_ops.py
from typing import List, Dict, Any, Tuple
import pandas as pd
import re
import ftfy  # For encoding fix
OP_REGISTRY = {}

def trim_whitespace(df: pd.DataFrame, columns: List[str] = None, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    cols = columns or df.select_dtypes(include=['object']).columns.tolist()
    for col in cols:
        df[col] = df[col].astype(str).str.strip()
    return df, {"trimmed": cols}, []

def lowercase_normalization(df: pd.DataFrame, columns: List[str] = None, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    cols = columns or df.select_dtypes(include=['object']).columns.tolist()
    for col in cols:
        df[col] = df[col].astype(str).str.lower()
    return df, {"lowercased": cols}, []

def uppercase_normalization(df: pd.DataFrame, columns: List[str] = None, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    cols = columns or df.select_dtypes(include=['object']).columns.tolist()
    for col in cols:
        df[col] = df[col].astype(str).str.upper()
    return df, {"uppercased": cols}, []

def remove_emojis(df: pd.DataFrame, columns: List[str] = None, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    cols = columns or df.select_dtypes(include=['object']).columns.tolist()
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        "]+", flags=re.UNICODE)
    for col in cols:
        df[col] = df[col].astype(str).apply(lambda x: emoji_pattern.sub(r'', x))
    return df, {"emojis_removed": cols}, []

def remove_special_characters(df: pd.DataFrame, columns: List[str] = None, keep: str = " ", **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    cols = columns or df.select_dtypes(include=['object']).columns.tolist()
    pattern = rf"[^0-9a-zA-Z{re.escape(keep)}]"
    for col in cols:
        df[col] = df[col].astype(str).str.replace(pattern, "", regex=True)
    return df, {"special_chars_removed": cols}, []

def regex_pattern_replacements(df: pd.DataFrame, columns: List[str] = None, pattern: str = None, repl: str = "", **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    if not pattern:
        return df, {}, ["Pattern required for regex replace"]
    cols = columns or df.select_dtypes(include=['object']).columns.tolist()
    for col in cols:
        df[col] = df[col].astype(str).str.replace(pattern, repl, regex=True)
    return df, {"regex_replaced": {"pattern": pattern, "repl": repl}}, []

def remove_html_tags(df: pd.DataFrame, columns: List[str] = None, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    cols = columns or df.select_dtypes(include=['object']).columns.tolist()
    pattern = re.compile(r'<.*?>')
    for col in cols:
        df[col] = df[col].astype(str).apply(lambda x: pattern.sub('', x))
    return df, {"html_removed": cols}, []

def unicode_normalization(df: pd.DataFrame, columns: List[str] = None, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    cols = columns or df.select_dtypes(include=['object']).columns.tolist()
    for col in cols:
        df[col] = df[col].astype(str).str.normalize('NFKD')
    return df, {"unicode_normalized": cols}, []

def fix_encoding_errors(df: pd.DataFrame, columns: List[str] = None, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    cols = columns or df.select_dtypes(include=['object']).columns.tolist()
    for col in cols:
        df[col] = df[col].apply(lambda x: ftfy.fix_text(str(x)) if pd.notna(x) else x)
    return df, {"encoding_fixed": cols}, []

OP_REGISTRY = {
    "trim_whitespace": trim_whitespace,
    "lowercase_normalization": lowercase_normalization,
    "uppercase_normalization": uppercase_normalization,
    "remove_emojis": remove_emojis,
    "remove_special_characters": remove_special_characters,
    "regex_pattern_replacements": regex_pattern_replacements,
    "remove_html_tags": remove_html_tags,
    "unicode_normalization": unicode_normalization,
    "fix_encoding_errors": fix_encoding_errors,
}