# services/ops/tabular/fe_ops.py
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
OP_REGISTRY = {}

def polynomial_features(df: pd.DataFrame, columns: List[str] = None, degree: int = 2, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    warns = []
    cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()[:2]  # Limit to avoid explosion
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    poly_features = poly.fit_transform(df[cols])
    poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(cols), index=df.index)
    df = pd.concat([df, poly_df], axis=1)
    return df, {"poly_features_degree": degree}, warns

def interaction_features(df: pd.DataFrame, columns: List[str] = None, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    warns = []
    cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            new_col = f"{cols[i]}_x_{cols[j]}"
            df[new_col] = df[cols[i]] * df[cols[j]]
    return df, {"interaction_cols": [f"{c1}_x_{c2}" for c1, c2 in combinations]}, warns

def bucketization_binning(df: pd.DataFrame, columns: List[str] = None, bins: int = 5, method: str = 'quantile', **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    warns = []
    cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
    for col in cols:
        new_col = f"{col}_bin"
        if method == 'quantile':
            df[new_col] = pd.qcut(df[col], q=bins, labels=False, duplicates='drop')
        else:
            df[new_col] = pd.cut(df[col], bins=bins, labels=False)
    return df, {"binned_cols": [f"{c}_bin" for c in cols]}, warns

def date_feature_splits(df: pd.DataFrame, columns: List[str] = None, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    warns = []
    cols = columns or df.select_dtypes(include=['datetime']).columns.tolist()
    for col in cols:
        df[f"{col}_year"] = df[col].dt.year
        df[f"{col}_month"] = df[col].dt.month
        df[f"{col}_day"] = df[col].dt.day
        df[f"{col}_week"] = df[col].dt.isocalendar().week
        df[f"{col}_hour"] = df[col].dt.hour
    return df, {"date_features": [f"{c}_{unit}" for c in cols for unit in ['year', 'month', 'day', 'week', 'hour']]}, warns

def text_length_features(df: pd.DataFrame, columns: List[str] = None, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    warns = []
    cols = columns or df.select_dtypes(include=['object']).columns.tolist()
    for col in cols:
        df[f"{col}_num_chars"] = df[col].astype(str).str.len()
        df[f"{col}_num_words"] = df[col].astype(str).str.split().str.len()
    return df, {"text_features": [f"{c}_{metric}" for c in cols for metric in ['num_chars', 'num_words']]}, warns

def frequency_encoding(df: pd.DataFrame, columns: List[str] = None, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    warns = []
    cols = columns or df.select_dtypes(include=['object']).columns.tolist()
    for col in cols:
        freq = df[col].value_counts()
        df[f"{col}_freq"] = df[col].map(freq)
    return df, {"freq_encoded": cols}, warns

def groupby_aggregate_features(df: pd.DataFrame, group_col: str, agg_col: str, agg_func: str = 'mean', **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    agg = df.groupby(group_col)[agg_col].agg(agg_func).rename(f"{agg_col}_{agg_func}_by_{group_col}")
    df = df.join(agg, on=group_col)
    return df, {"agg_feature": agg.name}, []

def time_series_rolling_aggregates(df: pd.DataFrame, columns: List[str] = None, window: int = 3, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    warns = []
    cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
    for col in cols:
        df[f"{col}_rolling_mean"] = df[col].rolling(window=window).mean()
    return df, {"rolling_features": [f"{c}_rolling_mean" for c in cols]}, warns

def lag_features(df: pd.DataFrame, columns: List[str] = None, lags: List[int] = [1, 2], **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    warns = []
    cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
    for col in cols:
        for lag in lags:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)
    return df, {"lag_features": [f"{c}_lag_{l}" for c in cols for l in lags]}, warns

OP_REGISTRY = {
    "polynomial_features": polynomial_features,
    "interaction_features": interaction_features,
    "bucketization_binning": bucketization_binning,
    "date_feature_splits": date_feature_splits,
    "text_length_features": text_length_features,
    "frequency_encoding": frequency_encoding,
    "groupby_aggregate_features": groupby_aggregate_features,
    "time_series_rolling_aggregates": time_series_rolling_aggregates,
    "lag_features": lag_features,
}