# services/ops/tabular/encoding_ops.py
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from category_encoders import TargetEncoder, LeaveOneOutEncoder, HashingEncoder, BinaryEncoder
from sklearn.model_selection import KFold
OP_REGISTRY = {}

def one_hot_encoding(df: pd.DataFrame, columns: List[str] = None, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    warns = []
    cols = columns or df.select_dtypes(include=['object', 'category']).columns.tolist()
    try:
        ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        encoded = ohe.fit_transform(df[cols])
        encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(cols), index=df.index)
        df = df.drop(columns=cols)
        df = pd.concat([df, encoded_df], axis=1)
    except Exception as e:
        warns.append(f"One-hot encoding failed: {e}")
    return df, {"onehot_encoder": ohe, "encoded_cols": ohe.get_feature_names_out().tolist()}, warns

def label_encoding(df: pd.DataFrame, columns: List[str] = None, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    warns = []
    cols = columns or df.select_dtypes(include=['object', 'category']).columns.tolist()
    encoders = {}
    for col in cols:
        try:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str).fillna('missing'))
            encoders[col] = le
        except Exception as e:
            warns.append(f"Label encoding failed for {col}: {e}")
    return df, {"label_encoders": encoders}, warns

def ordinal_encoding(df: pd.DataFrame, columns: List[str] = None, mapping: Dict = None, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    warns = []
    cols = columns or df.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in cols:
        try:
            oe = OrdinalEncoder(categories=[mapping.get(col, sorted(df[col].unique()))] if mapping else 'auto')
            df[col] = oe.fit_transform(df[[col]]).flatten()
        except Exception as e:
            warns.append(f"Ordinal encoding failed for {col}: {e}")
    return df, {"ordinal_encoders": {col: oe for col in cols}}, warns

def target_encoding(df: pd.DataFrame, columns: List[str] = None, target_col: str = None, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    warns = []
    if not target_col:
        warns.append("Target column required for target encoding")
        return df, {}, warns
    cols = columns or df.select_dtypes(include=['object', 'category']).columns.tolist()
    try:
        te = TargetEncoder(cols=cols)
        df[cols] = te.fit_transform(df[cols], df[target_col])
    except Exception as e:
        warns.append(f"Target encoding failed: {e}")
    return df, {"target_encoder": te}, warns

def kfold_target_encoding(df: pd.DataFrame, columns: List[str] = None, target_col: str = None, n_splits: int = 5, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    warns = []
    if not target_col:
        warns.append("Target column required for k-fold target encoding")
        return df, {}, warns
    cols = columns or df.select_dtypes(include=['object', 'category']).columns.tolist()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    encoded_cols = {}
    for col in cols:
        encoded = np.full(len(df), np.nan)
        for train_idx, val_idx in kf.split(df):
            te_fold = TargetEncoder(cols=[col])
            train_enc = te_fold.fit_transform(df.iloc[train_idx][[col]], df.iloc[train_idx][target_col])
            val_enc = te_fold.transform(df.iloc[val_idx][[col]])
            encoded[val_idx] = val_enc[col].values
        df[col + '_kfold_target'] = encoded
        encoded_cols[col] = True
    return df, {"kfold_target_encoded": encoded_cols}, warns

def hashing_encoding(df: pd.DataFrame, columns: List[str] = None, n_components: int = 8, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    warns = []
    cols = columns or df.select_dtypes(include=['object', 'category']).columns.tolist()
    try:
        he = HashingEncoder(cols=cols, n_components=n_components)
        encoded = he.fit_transform(df[cols])
        df = df.drop(columns=cols)
        df = pd.concat([df, encoded], axis=1)
    except Exception as e:
        warns.append(f"Hashing encoding failed: {e}")
    return df, {"hashing_encoder": he}, warns

def binary_encoding(df: pd.DataFrame, columns: List[str] = None, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    warns = []
    cols = columns or df.select_dtypes(include=['object', 'category']).columns.tolist()
    try:
        be = BinaryEncoder(cols=cols)
        encoded = be.fit_transform(df[cols])
        df = df.drop(columns=cols)
        df = pd.concat([df, encoded], axis=1)
    except Exception as e:
        warns.append(f"Binary encoding failed: {e}")
    return df, {"binary_encoder": be}, warns

def leave_one_out_encoding(df: pd.DataFrame, columns: List[str] = None, target_col: str = None, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    warns = []
    if not target_col:
        warns.append("Target column required for leave-one-out encoding")
        return df, {}, warns
    cols = columns or df.select_dtypes(include=['object', 'category']).columns.tolist()
    try:
        looe = LeaveOneOutEncoder(cols=cols)
        df[cols] = looe.fit_transform(df[cols], df[target_col])
    except Exception as e:
        warns.append(f"Leave-one-out encoding failed: {e}")
    return df, {"loo_encoder": looe}, warns

OP_REGISTRY = {
    "one_hot_encoding": one_hot_encoding,
    "label_encoding": label_encoding,
    "ordinal_encoding": ordinal_encoding,
    "target_encoding": target_encoding,
    "kfold_target_encoding": kfold_target_encoding,
    "hashing_encoding": hashing_encoding,
    "binary_encoding": binary_encoding,
    "leave_one_out_encoding": leave_one_out_encoding,
}