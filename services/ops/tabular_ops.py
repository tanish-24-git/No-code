# services/ops/tabular_ops.py
from typing import List, Dict, Any, Optional, Tuple, Callable
import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    LabelEncoder,
    OneHotEncoder,
)
from sklearn.feature_selection import VarianceThreshold
import logging

logger = logging.getLogger(__name__)

# Type alias for any tabular op function
TabularOpFunc = Callable[..., pd.DataFrame]


# --------------------------------
# Missing value handling
# --------------------------------

def impute_mean(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
    df = df.copy()
    for c in cols:
        try:
            df[c] = df[c].fillna(df[c].mean())
        except Exception as e:
            logger.warning("impute_mean failed on %s: %s", c, e)
    return df


def impute_median(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
    df = df.copy()
    for c in cols:
        try:
            df[c] = df[c].fillna(df[c].median())
        except Exception as e:
            logger.warning("impute_median failed on %s: %s", c, e)
    return df


def impute_mode(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    cols = columns or df.columns.tolist()
    df = df.copy()
    for c in cols:
        try:
            modes = df[c].mode()
            if len(modes) > 0:
                df[c] = df[c].fillna(modes.iloc[0])
        except Exception as e:
            logger.warning("impute_mode failed on %s: %s", c, e)
    return df


def drop_missing(df: pd.DataFrame, axis: int = 0) -> pd.DataFrame:
    """axis=0 drop rows, axis=1 drop columns"""
    return df.dropna(axis=axis)


def apply_missing_strategy(
    df: pd.DataFrame,
    strategy: str,
    columns: Optional[List[str]],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    High-level dispatcher: mean / median / mode / drop
    Returns (new_df, warnings).
    """
    warnings: List[str] = []
    strat = (strategy or "mean").lower()

    if strat == "mean":
        return impute_mean(df, columns), warnings
    elif strat == "median":
        return impute_median(df, columns), warnings
    elif strat == "mode":
        return impute_mode(df, columns), warnings
    elif strat == "drop":
        if columns:
            # drop rows where any of these columns are NA
            return df.dropna(subset=columns), warnings
        else:
            # drop any row with NA anywhere
            return df.dropna(), warnings
    elif strat == "constant":
        # placeholder; user should supply default value per-column in future
        warnings.append("constant fill not configured; no-op")
        return df, warnings
    elif strat == "model_based":
        warnings.append("model-based imputation not implemented yet; no-op")
        return df, warnings
    else:
        warnings.append(f"Unknown missing strategy '{strategy}'")
        return df, warnings


# --------------------------------
# Cleaning utilities
# --------------------------------

def strip_whitespace(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    df = df.copy()
    cols = columns or df.select_dtypes(include=["object", "category"]).columns.tolist()
    for c in cols:
        try:
            df[c] = df[c].astype(str).str.strip()
        except Exception as e:
            logger.warning("strip_whitespace failed on %s: %s", c, e)
    return df


def lower_case(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    df = df.copy()
    cols = columns or df.select_dtypes(include=["object", "category"]).columns.tolist()
    for c in cols:
        try:
            df[c] = df[c].astype(str).str.lower()
        except Exception as e:
            logger.warning("lower_case failed on %s: %s", c, e)
    return df


def upper_case(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    df = df.copy()
    cols = columns or df.select_dtypes(include=["object", "category"]).columns.tolist()
    for c in cols:
        try:
            df[c] = df[c].astype(str).str.upper()
        except Exception as e:
            logger.warning("upper_case failed on %s: %s", c, e)
    return df


def remove_emojis(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Basic emoji removal using unicode ranges.
    """
    df = df.copy()
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "]+",
        flags=re.UNICODE,
    )
    cols = columns or df.select_dtypes(include=["object", "category"]).columns.tolist()
    for c in cols:
        try:
            df[c] = df[c].astype(str).apply(lambda x: emoji_pattern.sub("", x))
        except Exception as e:
            logger.warning("remove_emojis failed on %s: %s", c, e)
    return df


def remove_special_chars(df: pd.DataFrame, columns: Optional[List[str]] = None, keep: str = " ") -> pd.DataFrame:
    """
    Remove non-alphanumeric characters (except allowed 'keep' chars).
    """
    df = df.copy()
    pattern = rf"[^0-9a-zA-Z{re.escape(keep)}]"
    cols = columns or df.select_dtypes(include=["object", "category"]).columns.tolist()
    for c in cols:
        try:
            df[c] = df[c].astype(str).str.replace(pattern, "", regex=True)
        except Exception as e:
            logger.warning("remove_special_chars failed on %s: %s", c, e)
    return df


def regex_replace(df: pd.DataFrame, columns: List[str], pattern: str, repl: str = "") -> pd.DataFrame:
    df = df.copy()
    for c in columns:
        try:
            df[c] = df[c].astype(str).str.replace(pattern, repl, regex=True)
        except Exception as e:
            logger.warning("regex_replace failed on %s: %s", c, e)
    return df


# --------------------------------
# Encoding & scaling
# --------------------------------

def label_encode_columns(
    df: pd.DataFrame,
    columns: List[str],
    save_meta: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Applies LabelEncoder to each column listed and stores classes in save_meta.
    Returns transformed df and meta updates.
    """
    df = df.copy()
    meta = {}
    for c in columns:
        try:
            le = LabelEncoder()
            df[c] = le.fit_transform(df[c].astype(str).fillna("##MISSING##"))
            meta[c] = {"type": "label", "classes": le.classes_.tolist()}
        except Exception as e:
            logger.warning("label_encode_columns failed on %s: %s", c, e)
    save_meta.update(meta)
    return df, save_meta


def onehot_encode_columns(
    df: pd.DataFrame,
    columns: List[str],
    save_meta: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    OneHotEncode specified columns (drop='first' to avoid collinearity).
    Returns df with new columns and meta listing the produced feature names.
    """
    df = df.copy()
    try:
        ohe = OneHotEncoder(drop="first", sparse=False, handle_unknown="ignore")
        arr = ohe.fit_transform(df[columns].astype(str).fillna("##MISSING##"))
        out_cols = ohe.get_feature_names_out(columns).tolist()
        df = df.drop(columns=columns)
        df_out = pd.DataFrame(arr, columns=out_cols, index=df.index)
        df = pd.concat([df, df_out], axis=1)
        save_meta.setdefault("onehot", {})
        save_meta["onehot"] = {"columns": columns, "produced": out_cols}
    except Exception as e:
        logger.warning("onehot_encode_columns failed for %s: %s", columns, e)
        save_meta.setdefault("warnings", []).append(f"onehot failed for {columns}: {str(e)}")
    return df, save_meta


def apply_scaler(
    df: pd.DataFrame,
    columns: List[str],
    method: str = "standard",
    save_meta: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    method: 'standard' | 'minmax' | 'passthrough'
    Stores scaler params in save_meta if provided.
    """
    df = df.copy()
    save_meta = save_meta or {}

    if method == "passthrough":
        return df, save_meta

    try:
        if method == "standard":
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()
        arr = scaler.fit_transform(df[columns].astype(float))
        df[columns] = arr
        save_meta.setdefault("scaler", {})
        save_meta["scaler"][",".join(columns)] = {"method": method}
    except Exception as e:
        logger.warning("apply_scaler failed for %s: %s", columns, e)
        save_meta.setdefault("warnings", []).append(f"scaler failed for {columns}: {str(e)}")

    return df, save_meta


def standard_scale(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    arr = StandardScaler().fit_transform(df[columns].astype(float))
    df = df.copy()
    df[columns] = arr
    return df


def minmax_scale(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    arr = MinMaxScaler().fit_transform(df[columns].astype(float))
    df = df.copy()
    df[columns] = arr
    return df


# --------------------------------
# Feature selection & simple FE
# --------------------------------

def variance_threshold(df: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    """
    Remove columns with variance <= threshold.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        return df
    selector = VarianceThreshold(threshold=threshold)
    arr = selector.fit_transform(df[num_cols])
    kept_cols = [c for c, keep in zip(num_cols, selector.get_support()) if keep]
    df = df.copy()
    df = df.drop(columns=num_cols)
    df[kept_cols] = arr
    return df


def bucketize_numeric(df: pd.DataFrame, column: str, bins: int = 5, strategy: str = "quantile") -> pd.DataFrame:
    """
    Simple binning: equal-width or quantile.
    """
    df = df.copy()
    try:
        if strategy == "quantile":
            df[column + "_bin"] = pd.qcut(df[column], q=bins, duplicates="drop")
        else:
            df[column + "_bin"] = pd.cut(df[column], bins=bins)
    except Exception as e:
        logger.warning("bucketize_numeric failed on %s: %s", column, e)
    return df


def deduplicate_rows(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    return df.drop_duplicates(subset=subset)


def winsorize_outliers(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> pd.DataFrame:
    """
    Clamps numeric values to [q_lower, q_upper] per column.
    """
    df = df.copy()
    cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
    for c in cols:
        try:
            q_low = df[c].quantile(lower_quantile)
            q_high = df[c].quantile(upper_quantile)
            df[c] = df[c].clip(q_low, q_high)
        except Exception as e:
            logger.warning("winsorize_outliers failed on %s: %s", c, e)
    return df


# --------------------------------
# Generic op registry + dispatcher
# --------------------------------

OP_REGISTRY: Dict[str, TabularOpFunc] = {
    # Missing
    "mean_impute": impute_mean,
    "median_impute": impute_median,
    "mode_impute": impute_mode,
    "drop_missing": drop_missing,
    # Cleaning
    "trim_whitespace": strip_whitespace,
    "lowercase": lower_case,
    "uppercase": upper_case,
    "remove_emojis": remove_emojis,
    "remove_special_chars": remove_special_chars,
    "regex_replace": regex_replace,
    # Scaling / encoding (note: label/onehot/scale need special handler)
    "standard_scale": standard_scale,
    "minmax_scale": minmax_scale,
    # Feature selection / FE
    "variance_threshold": variance_threshold,
    "bucketize": bucketize_numeric,
    "deduplicate": deduplicate_rows,
    "winsorize": winsorize_outliers,
    # Placeholders for advanced stuff (safe no-op)
    # You can implement real logic later.
    "model_based_imputation": lambda df, **_: df,
    "frequency_encoding": lambda df, **_: df,
    "groupby_aggregate": lambda df, **_: df,
    "rolling_aggregate": lambda df, **_: df,
    "lag_features": lambda df, **_: df,
    "select_k_best": lambda df, **_: df,
    "drop_correlated": lambda df, **_: df,
    "remove_multicollinearity": lambda df, **_: df,
}

def list_ops() -> List[str]:
    return sorted(list(OP_REGISTRY.keys()))


def apply_op(
    df: pd.DataFrame,
    op_name: str,
    func: TabularOpFunc,
    step_config: Dict[str, Any],
    meta_store: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any], List[str]]:
    """
    Generic dispatcher for tabular ops used by the engine.
    It standardizes how we pull arguments from step_config and how we emit warnings.

    Returns: (new_df, updated_meta_store, warnings)
    """
    warnings: List[str] = []

    # Many ops accept "columns"; some accept extra params; we standardize a bit.
    columns = step_config.get("columns")
    kwargs = {k: v for k, v in step_config.items() if k not in ("op", "columns")}

    # Special handling for ops that know about meta_store (encoding/scaling), but
    # we keep them simple here to avoid the engine having to know all details.
    if op_name in ("label_encode", "label_encoding"):
        if not columns:
            warnings.append("label_encode requires 'columns'; step skipped")
            return df, meta_store, warnings
        new_df, meta_store = label_encode_columns(df, columns, meta_store)
        return new_df, meta_store, warnings

    if op_name in ("onehot_encode", "one_hot_encoding"):
        if not columns:
            warnings.append("onehot_encode requires 'columns'; step skipped")
            return df, meta_store, warnings
        new_df, meta_store = onehot_encode_columns(df, columns, meta_store)
        return new_df, meta_store, warnings

    if op_name in ("scale", "scale_numeric"):
        if not columns:
            warnings.append("scale requires 'columns'; step skipped")
            return df, meta_store, warnings
        method = kwargs.get("method", "standard")
        new_df, meta_store = apply_scaler(df, columns, method=method, save_meta=meta_store)
        return new_df, meta_store, warnings

    # Generic path: function returns DataFrame only
    if columns is not None:
        new_df = func(df, columns=columns, **kwargs)
    else:
        new_df = func(df, **kwargs)

    return new_df, meta_store, warnings
