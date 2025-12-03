# services/ops/nlp/tokenization_ops.py
from typing import List, Dict, Any, Tuple
import pandas as pd
import tiktoken
from spacy.lang.en import English
import warnings
OP_REGISTRY = {}

def token_count_estimation(df: pd.DataFrame, field: str = "text", out_field: str = "token_count", **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    df = df.copy()
    if field in df.columns:
        encoding = tiktoken.get_encoding("cl100k_base")
        df[out_field] = df[field].astype(str).apply(lambda x: len(encoding.encode(x)))
    return df, {"token_estimator": "tiktoken"}, []

def max_token_truncation(df: pd.DataFrame, field: str = "text", max_tokens: int = 512, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    df = df.copy()
    if field in df.columns:
        encoding = tiktoken.get_encoding("cl100k_base")
        def trunc(text):
            tokens = encoding.encode(text)
            if len(tokens) > max_tokens:
                tokens = tokens[:max_tokens]
                text = encoding.decode(tokens)
            return text
        df[field] = df[field].astype(str).apply(trunc)
    return df, {"truncated_max": max_tokens}, []

def token_overflow_warnings(df: pd.DataFrame, field: str = "text", max_tokens: int = 512, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    warns = []
    if field in df.columns:
        encoding = tiktoken.get_encoding("cl100k_base")
        overflows = (df[field].astype(str).apply(lambda x: len(encoding.encode(x))) > max_tokens).sum()
        if overflows > 0:
            warns.append(f"{overflows} rows exceed {max_tokens} tokens")
    return df, {"overflow_count": overflows if 'overflows' in locals() else 0}, warns

def word_piece_bpe_simulation(df: pd.DataFrame, field: str = "text", **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    df = df.copy()
    if field in df.columns:
        encoding = tiktoken.get_encoding("cl100k_base")
        df[f"{field}_bpe"] = df[field].astype(str).apply(lambda x: encoding.encode(x))
    return df, {"bpe_simulated": True}, []

def sentencepiece_based_splitting(df: pd.DataFrame, field: str = "text", **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    # Placeholder: Use subword-nmt or sentencepiece lib if added
    nlp = English()
    nlp.add_pipe("sentencizer")
    df = df.copy()
    if field in df.columns:
        def split_sent(text):
            doc = nlp(text)
            return [sent.text.strip() for sent in doc.sents]
        df[f"{field}_sentences"] = df[field].astype(str).apply(split_sent)
    return df, {"sentencepiece_split": True}, []

def padding_required(df: pd.DataFrame, field: str = "text", max_length: int = 512, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    df = df.copy()
    if field in df.columns:
        encoding = tiktoken.get_encoding("cl100k_base")
        def pad(text):
            tokens = encoding.encode(text)
            if len(tokens) < max_length:
                tokens += [0] * (max_length - len(tokens))  # Pad with 0
            return encoding.decode(tokens)
        df[field] = df[field].astype(str).apply(pad)
    return df, {"padded_to": max_length}, []

def token_normalization(df: pd.DataFrame, field: str = "text", **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    df = df.copy()
    if field in df.columns:
        # Lowercase and remove extra spaces
        df[field] = df[field].astype(str).str.lower().str.replace(r'\s+', ' ', regex=True).str.strip()
    return df, {"tokens_normalized": True}, []

OP_REGISTRY = {
    "token_count_estimation": token_count_estimation,
    "max_token_truncation": max_token_truncation,
    "token_overflow_warnings": token_overflow_warnings,
    "word_piece_bpe_simulation": word_piece_bpe_simulation,
    "sentencepiece_based_splitting": sentencepiece_based_splitting,
    "padding_required": padding_required,
    "token_normalization": token_normalization,
}