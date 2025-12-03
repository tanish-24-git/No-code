# services/ops/nlp/splitting_ops.py
from typing import List, Dict, Any, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
OP_REGISTRY = {}

def train_validation_test_split(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.2, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    train_val, test = train_test_split(df, test_size=test_size, random_state=42)
    train, val = train_test_split(train_val, test_size=val_size / (1 - test_size), random_state=42)
    splits = {"train": train, "val": val, "test": test}
    return pd.concat(splits.values()), {"splits": {"train": len(train), "val": len(val), "test": len(test)}}, []

def stratified_split(df: pd.DataFrame, target_col: str = "label", test_size: float = 0.2, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    train, test = train_test_split(df, test_size=test_size, stratify=df[target_col], random_state=42)
    return pd.concat([train, test]), {"stratified_split": {"train": len(train), "test": len(test)}}, []

def seed_based_deterministic_splits(df: pd.DataFrame, seed: int = 42, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    np.random.seed(seed)
    idx = np.random.permutation(len(df))
    split_point = int(len(df) * 0.8)
    train = df.iloc[idx[:split_point]]
    test = df.iloc[idx[split_point:]]
    return pd.concat([train, test]), {"seed_split": seed}, []

def shuffle_no_shuffle_modes(df: pd.DataFrame, shuffle: bool = True, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    if shuffle:
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df, {"shuffled": shuffle}, []

OP_REGISTRY = {
    "train_validation_test_split": train_validation_test_split,
    "stratified_split": stratified_split,
    "seed_based_deterministic_splits": seed_based_deterministic_splits,
    "shuffle_no_shuffle_modes": shuffle_no_shuffle_modes,
}