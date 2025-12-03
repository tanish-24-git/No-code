# services/ops/tabular/type_fix_ops.py
from typing import List, Dict, Any, Tuple
import pandas as pd
from datetime import datetime
import re
OP_REGISTRY = {}

def cast_to_int(df: pd.DataFrame, columns: List[str] = None, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    warns = []
    cols = columns or df.select_dtypes(include=['object', 'float']).columns.tolist()
    for col in cols:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
        except Exception as e:
            warns.append(f"Cast to int failed for {col}: {e}")
    return df, {"cast_to_int": cols}, warns

def cast_to_float(df: pd.DataFrame, columns: List[str] = None, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    warns = []
    cols = columns or df.select_dtypes(include=['object', 'int']).columns.tolist()
    for col in cols:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except Exception as e:
            warns.append(f"Cast to float failed for {col}: {e}")
    return df, {"cast_to_float": cols}, warns

def cast_to_categorical(df: pd.DataFrame, columns: List[str] = None, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    cols = columns or df.select_dtypes(include=['object']).columns.tolist()
    for col in cols:
        df[col] = df[col].astype('category')
    return df, {"cast_to_categorical": cols}, []

def parse_to_datetime(df: pd.DataFrame, columns: List[str] = None, format: str = None, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    warns = []
    cols = columns or df.select_dtypes(include=['object']).columns.tolist()
    for col in cols:
        try:
            df[col] = pd.to_datetime(df[col], format=format, errors='coerce')
        except Exception as e:
            warns.append(f"Parse datetime failed for {col}: {e}")
    return df, {"parse_to_datetime": cols}, warns

def convert_numeric_strings(df: pd.DataFrame, columns: List[str] = None, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    warns = []
    cols = columns or df.select_dtypes(include=['object']).columns.tolist()
    for col in cols:
        def clean_num(s):
            if pd.isna(s):
                return s
            s = str(s).replace(',', '').replace('k', '000').replace('m', '000000')
            try:
                return pd.to_numeric(s)
            except:
                return s
        df[col] = df[col].apply(clean_num)
    return df, {"numeric_string_convert": cols}, warns

def auto_detect_binary(df: pd.DataFrame, columns: List[str] = None, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    cols = columns or df.select_dtypes(include=['object']).columns.tolist()
    binary_cols = []
    for col in cols:
        unique = df[col].nunique()
        if unique == 2:
            df[col] = df[col].astype('bool')
            binary_cols.append(col)
    return df, {"binary_detected": binary_cols}, []

OP_REGISTRY = {
    "cast_to_int": cast_to_int,
    "cast_to_float": cast_to_float,
    "cast_to_categorical": cast_to_categorical,
    "parse_to_datetime": parse_to_datetime,
    "convert_numeric_strings": convert_numeric_strings,
    "auto_detect_binary": auto_detect_binary,
}