"""
Deduplication utilities for text data.
Supports exact and near-duplicate detection.
"""
import hashlib
from typing import List, Set, Tuple
import pandas as pd


def exact_dedup(texts: List[str]) -> Tuple[List[str], List[int]]:
    """
    Remove exact duplicates from text list.
    
    Args:
        texts: List of texts
    
    Returns:
        Tuple of (deduplicated texts, indices of kept items)
    """
    seen = set()
    deduplicated = []
    kept_indices = []
    
    for i, text in enumerate(texts):
        if text not in seen:
            seen.add(text)
            deduplicated.append(text)
            kept_indices.append(i)
    
    return deduplicated, kept_indices


def hash_text(text: str) -> str:
    """Generate hash for text."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def exact_dedup_dataframe(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """
    Remove exact duplicates from DataFrame based on text column.
    
    Args:
        df: Input DataFrame
        text_column: Name of text column
    
    Returns:
        Deduplicated DataFrame
    """
    return df.drop_duplicates(subset=[text_column], keep='first')


def near_dedup_simple(texts: List[str], similarity_threshold: float = 0.9) -> Tuple[List[str], List[int]]:
    """
    Simple near-duplicate detection using character n-grams.
    
    Note: This is a simplified version. For production, consider using
    MinHash LSH for better performance on large datasets.
    
    Args:
        texts: List of texts
        similarity_threshold: Similarity threshold (0-1)
    
    Returns:
        Tuple of (deduplicated texts, indices of kept items)
    """
    # For now, just do exact dedup
    # TODO: Implement proper MinHash LSH for near-duplicate detection
    return exact_dedup(texts)


def get_duplicate_stats(df: pd.DataFrame, text_column: str) -> dict:
    """
    Get statistics about duplicates in dataset.
    
    Args:
        df: Input DataFrame
        text_column: Name of text column
    
    Returns:
        Dictionary with duplicate statistics
    """
    total_rows = len(df)
    unique_rows = df[text_column].nunique()
    duplicate_count = total_rows - unique_rows
    
    return {
        "total_rows": total_rows,
        "unique_rows": unique_rows,
        "duplicate_count": duplicate_count,
        "duplicate_percentage": (duplicate_count / total_rows * 100) if total_rows > 0 else 0
    }
