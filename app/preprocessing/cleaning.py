"""
LLM-native text cleaning utilities.
NO sklearn preprocessing - only text normalization for LLMs.
"""
import re
import unicodedata
from typing import List


def normalize_unicode(text: str) -> str:
    """
    Normalize unicode characters to NFKC form.
    
    Args:
        text: Input text
    
    Returns:
        Normalized text
    """
    return unicodedata.normalize('NFKC', text)


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace (collapse multiple spaces, remove leading/trailing).
    
    Args:
        text: Input text
    
    Returns:
        Normalized text
    """
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    return text


def remove_control_characters(text: str) -> str:
    """
    Remove control characters (except newlines and tabs).
    
    Args:
        text: Input text
    
    Returns:
        Cleaned text
    """
    # Keep newlines and tabs, remove other control characters
    return re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)


def fix_encoding_issues(text: str) -> str:
    """
    Fix common encoding issues.
    
    Args:
        text: Input text
    
    Returns:
        Fixed text
    """
    # Try to encode/decode to fix encoding issues
    try:
        # Encode to bytes and decode back
        text = text.encode('utf-8', errors='ignore').decode('utf-8')
    except Exception:
        pass
    
    return text


def clean_text(text: str, aggressive: bool = False) -> str:
    """
    Clean text for LLM training.
    
    Args:
        text: Input text
        aggressive: If True, apply more aggressive cleaning
    
    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Basic cleaning
    text = fix_encoding_issues(text)
    text = normalize_unicode(text)
    text = remove_control_characters(text)
    text = normalize_whitespace(text)
    
    if aggressive:
        # Remove URLs
        text = re.sub(r'http[s]?://\S+', '', text)
        # Remove email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        # Normalize whitespace again after removals
        text = normalize_whitespace(text)
    
    return text


def clean_batch(texts: List[str], aggressive: bool = False) -> List[str]:
    """
    Clean a batch of texts.
    
    Args:
        texts: List of input texts
        aggressive: If True, apply aggressive cleaning
    
    Returns:
        List of cleaned texts
    """
    return [clean_text(text, aggressive=aggressive) for text in texts]
