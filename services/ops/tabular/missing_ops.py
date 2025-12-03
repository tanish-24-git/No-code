# services/ops/tabular/missing_ops.py
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.pipeline import Pipeline
from scipy.stats import mode
OP_REGISTRY = {}

def mean_impute(df: pd.DataFrame, columns: List[str] = None, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    warns = []
    cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
    imputer = SimpleImputer(strategy='mean')
    df[cols] = pd.DataFrame(imputer.fit_transform(df[cols]), columns=cols, index=df.index)
    return df, {"mean_imputer": imputer}, warns

def median_impute(df: pd.DataFrame, columns: List[str] = None, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    warns = []
    cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
    imputer = SimpleImputer(strategy='median')
    df[cols] = pd.DataFrame(imputer.fit_transform(df[cols]), columns=cols, index=df.index)
    return df, {"median_imputer": imputer}, warns

def mode_impute(df: pd.DataFrame, columns: List[str] = None, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    warns = []
    cols = columns or df.columns.tolist()
    imputer = SimpleImputer(strategy='most_frequent')
    df[cols] = pd.DataFrame(imputer.fit_transform(df[cols]), columns=cols, index=df.index)
    return df, {"mode_imputer": imputer}, warns

def constant_fill(df: pd.DataFrame, columns: List[str] = None, value: Any = 0, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    cols = columns or df.columns
    df[cols] = df[cols].fillna(value)
    return df, {"constant_value": value}, []

def drop_rows_missing(df: pd.DataFrame, columns: List[str] = None, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    initial_rows = len(df)
    if columns:
        df = df.dropna(subset=columns)
    else:
        df = df.dropna()
    dropped = initial_rows - len(df)
    return df, {"dropped_rows": dropped}, []

def drop_cols_missing(df: pd.DataFrame, threshold: float = 0.5, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    miss_pct = df.isnull().mean()
    cols_to_drop = miss_pct[miss_pct > threshold].index.tolist()
    df = df.drop(columns=cols_to_drop)
    return df, {"dropped_cols": cols_to_drop}, []

def forward_fill(df: pd.DataFrame, columns: List[str] = None, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    cols = columns or df.columns
    df[cols] = df[cols].fillna(method='ffill')
    return df, {}, []

def backward_fill(df: pd.DataFrame, columns: List[str] = None, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    cols = columns or df.columns
    df[cols] = df[cols].fillna(method='bfill')
    return df, {}, []

def missing_indicator(df: pd.DataFrame, columns: List[str] = None, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    from sklearn.impute import MissingIndicator
    ind = MissingIndicator()
    cols = columns or df.columns
    indicators = ind.fit_transform(df[cols])
    ind_cols = [f"{c}_missing" for c in cols]
    df[ind_cols] = pd.DataFrame(indicators, columns=ind_cols, index=df.index)
    return df, {"indicator_features": ind_cols}, []

def model_based_impute(df: pd.DataFrame, columns: List[str] = None, n_neighbors: int = 5, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df[cols] = pd.DataFrame(imputer.fit_transform(df[cols]), columns=cols, index=df.index)
    return df, {"knn_imputer": imputer}, []

OP_REGISTRY = {
    "mean_impute": mean_impute,
    "median_impute": median_impute,
    "mode_impute": mode_impute,
    "constant_fill": constant_fill,
    "drop_rows_missing": drop_rows_missing,
    "drop_cols_missing": drop_cols_missing,
    "forward_fill": forward_fill,
    "backward_fill": backward_fill,
    "missing_indicator": missing_indicator,
    "model_based_impute": model_based_impute,
}