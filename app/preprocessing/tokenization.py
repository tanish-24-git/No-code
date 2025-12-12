"""
Tokenization utilities for LLM training.
"""
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer


class TokenizerWrapper:
    """
    Wrapper for HuggingFace tokenizers with caching.
    """
    
    _tokenizer_cache: Dict[str, Any] = {}
    
    @classmethod
    def get_tokenizer(cls, model_name: str) -> Any:
        """
        Get tokenizer for a model (with caching).
        
        Args:
            model_name: HuggingFace model name
        
        Returns:
            Tokenizer instance
        """
        if model_name not in cls._tokenizer_cache:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            
            # Ensure pad token is set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            cls._tokenizer_cache[model_name] = tokenizer
        
        return cls._tokenizer_cache[model_name]


def tokenize_text(
    text: str,
    model_name: str,
    max_length: Optional[int] = None,
    truncation: bool = True,
    padding: bool = False
) -> Dict[str, Any]:
    """
    Tokenize a single text.
    
    Args:
        text: Input text
        model_name: Model name for tokenizer
        max_length: Maximum sequence length
        truncation: Whether to truncate
        padding: Whether to pad
    
    Returns:
        Dictionary with input_ids, attention_mask, and token_count
    """
    tokenizer = TokenizerWrapper.get_tokenizer(model_name)
    
    encoded = tokenizer(
        text,
        max_length=max_length,
        truncation=truncation,
        padding='max_length' if padding else False,
        return_tensors=None
    )
    
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded.get("attention_mask", []),
        "token_count": len(encoded["input_ids"])
    }


def tokenize_batch(
    texts: List[str],
    model_name: str,
    max_length: Optional[int] = None,
    truncation: bool = True,
    padding: bool = True
) -> Dict[str, Any]:
    """
    Tokenize a batch of texts.
    
    Args:
        texts: List of input texts
        model_name: Model name for tokenizer
        max_length: Maximum sequence length
        truncation: Whether to truncate
        padding: Whether to pad
    
    Returns:
        Dictionary with batched input_ids and attention_mask
    """
    tokenizer = TokenizerWrapper.get_tokenizer(model_name)
    
    encoded = tokenizer(
        texts,
        max_length=max_length,
        truncation=truncation,
        padding=padding,
        return_tensors=None
    )
    
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "token_counts": [len(ids) for ids in encoded["input_ids"]]
    }


def count_tokens(text: str, model_name: str) -> int:
    """
    Count tokens in text.
    
    Args:
        text: Input text
        model_name: Model name for tokenizer
    
    Returns:
        Token count
    """
    tokenizer = TokenizerWrapper.get_tokenizer(model_name)
    return len(tokenizer.encode(text))


def decode_tokens(token_ids: List[int], model_name: str) -> str:
    """
    Decode token IDs back to text.
    
    Args:
        token_ids: List of token IDs
        model_name: Model name for tokenizer
    
    Returns:
        Decoded text
    """
    tokenizer = TokenizerWrapper.get_tokenizer(model_name)
    return tokenizer.decode(token_ids, skip_special_tokens=True)
