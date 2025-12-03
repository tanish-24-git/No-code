# services/ops/tabular/row_ops.py
from typing import List, Dict, Any, Tuple
import pandas as pd
OP_REGISTRY = {}

def deduplication(df: pd.DataFrame, subset: List[str] = None, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    initial_len = len(df)
    df = df.drop_duplicates(subset=subset)
    dropped = initial_len - len(df)
    return df, {"dedup_dropped": dropped}, []

def conditional_filtering(df: pd.DataFrame, condition: str = None, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    if condition:
        df = df.query(condition)
    return df, {"filtered_condition": condition}, []

def outlier_removal_winsorization(df: pd.DataFrame, columns: List[str] = None, method: str = 'iqr', **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    warns = []
    cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df, {"outliers_removed_method": method}, warns

def aggregation_based_reduction(df: pd.DataFrame, group_col: str, agg_dict: Dict = None, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    if agg_dict is None:
        agg_dict = {'value': 'mean'}  # Default
    df_agg = df.groupby(group_col).agg(agg_dict).reset_index()
    return df_agg, {"agg_reduction": agg_dict}, []

OP_REGISTRY = {
    "deduplication": deduplication,
    "conditional_filtering": conditional_filtering,
    "outlier_removal_winsorization": outlier_removal_winsorization,
    "aggregation_based_reduction": aggregation_based_reduction,
}