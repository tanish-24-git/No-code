# services/ops/tabular/scaling_ops.py
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer, QuantileTransformer
OP_REGISTRY = {}

def standard_scaler(df: pd.DataFrame, columns: List[str] = None, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    warns = []
    cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df, {"standard_scaler": scaler}, warns

def min_max_scaler(df: pd.DataFrame, columns: List[str] = None, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    warns = []
    cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
    scaler = MinMaxScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df, {"minmax_scaler": scaler}, warns

def robust_scaler(df: pd.DataFrame, columns: List[str] = None, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    warns = []
    cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
    scaler = RobustScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df, {"robust_scaler": scaler}, warns

def normalizer_l1(df: pd.DataFrame, columns: List[str] = None, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    warns = []
    cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
    normalizer = Normalizer(norm='l1')
    df[cols] = normalizer.fit_transform(df[cols])
    return df, {"l1_normalizer": normalizer}, warns

def normalizer_l2(df: pd.DataFrame, columns: List[str] = None, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    warns = []
    cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
    normalizer = Normalizer(norm='l2')
    df[cols] = normalizer.fit_transform(df[cols])
    return df, {"l2_normalizer": normalizer}, warns

def log_scaling(df: pd.DataFrame, columns: List[str] = None, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    warns = []
    cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
    for col in cols:
        df[col] = np.log1p(df[col].clip(lower=0))
    return df, {"log_scaled": cols}, warns

def outlier_clipping(df: pd.DataFrame, columns: List[str] = None, lower: float = 0.01, upper: float = 0.99, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    warns = []
    cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
    for col in cols:
        q_low = df[col].quantile(lower)
        q_high = df[col].quantile(upper)
        df[col] = df[col].clip(q_low, q_high)
    return df, {"clipped_quantiles": {"lower": lower, "upper": upper}}, warns

def quantile_transformer(df: pd.DataFrame, columns: List[str] = None, n_quantiles: int = 1000, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    warns = []
    cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
    qt = QuantileTransformer(n_quantiles=n_quantiles)
    df[cols] = qt.fit_transform(df[cols])
    return df, {"quantile_transformer": qt}, warns

OP_REGISTRY = {
    "standard_scaler": standard_scaler,
    "min_max_scaler": min_max_scaler,
    "robust_scaler": robust_scaler,
    "normalizer_l1": normalizer_l1,
    "normalizer_l2": normalizer_l2,
    "log_scaling": log_scaling,
    "outlier_clipping": outlier_clipping,
    "quantile_transformer": quantile_transformer,
}