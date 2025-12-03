# services/ops/tabular/selection_ops.py
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor
OP_REGISTRY = {}

def select_k_best(df: pd.DataFrame, k: int = 10, target_col: str = None, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    warns = []
    if not target_col:
        warns.append("Target column required for SelectKBest")
        return df, {}, warns
    X = df.drop(columns=[target_col])
    y = df[target_col]
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    selected_cols = X.columns[selector.get_support()].tolist()
    df = df[selected_cols + [target_col]]
    return df, {"selected_kbest": selected_cols}, warns

def variance_threshold(df: pd.DataFrame, threshold: float = 0.0, columns: List[str] = None, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(df[cols])
    selected = df[cols].columns[selector.get_support()].tolist()
    df = df.drop(columns=[c for c in cols if c not in selected])
    return df, {"variance_selected": selected}, []

def correlation_based_dropping(df: pd.DataFrame, threshold: float = 0.9, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    df = df.drop(columns=to_drop)
    return df, {"corr_dropped": to_drop}, []

def remove_multicollinearity_vif(df: pd.DataFrame, threshold: float = 5.0, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    warns = []
    numeric_df = df.select_dtypes(include=[np.number])
    vif_data = pd.DataFrame()
    vif_data["feature"] = numeric_df.columns
    vif_data["VIF"] = [variance_inflation_factor(numeric_df.values, i) for i in range(len(numeric_df.columns))]
    to_drop = vif_data[vif_data['VIF'] > threshold]['feature'].tolist()
    df = df.drop(columns=to_drop)
    return df, {"vif_dropped": to_drop}, warns

def feature_importance_selection(df: pd.DataFrame, k: int = 10, target_col: str = None, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    warns = []
    if not target_col:
        warns.append("Target column required for feature importance")
        return df, {}, warns
    X = df.drop(columns=[target_col])
    y = df[target_col]
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    top_k = importances.head(k).index.tolist()
    df = df[top_k + [target_col]]
    return df, {"importance_selected": top_k}, warns

def manual_column_selection(df: pd.DataFrame, columns: List[str] = None, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    if columns:
        df = df[columns]
    return df, {"manual_selected": columns or []}, []

def auto_detect_redundant_fields(df: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    warns = []
    # Detect IDs/UUIDs via regex
    id_pattern = r'^[0-9a-f]{8}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{12}$'
    redundant = [col for col in df.columns if df[col].dtype == 'object' and df[col].str.match(id_pattern, na=False).any()]
    df = df.drop(columns=redundant)
    return df, {"redundant_detected": redundant}, warns

OP_REGISTRY = {
    "select_k_best": select_k_best,
    "variance_threshold": variance_threshold,
    "correlation_based_dropping": correlation_based_dropping,
    "remove_multicollinearity_vif": remove_multicollinearity_vif,
    "feature_importance_selection": feature_importance_selection,
    "manual_column_selection": manual_column_selection,
    "auto_detect_redundant_fields": auto_detect_redundant_fields,
}