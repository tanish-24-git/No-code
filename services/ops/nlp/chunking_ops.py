# services/ops/nlp/chunking_ops.py
from typing import List, Dict, Any, Tuple
import pandas as pd
import tiktoken
import spacy
spacy_nlp = spacy.load("en_core_web_sm")
OP_REGISTRY = {}

def chunk_by_token_count(df: pd.DataFrame, field: str = "text", max_tokens: int = 256, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    df = df.copy()
    encoding = tiktoken.get_encoding("cl100k_base")
    new_rows = []
    for _, row in df.iterrows():
        text = str(row[field])
        tokens = encoding.encode(text)
        i = 0
        while i < len(tokens):
            chunk_tokens = tokens[i:i + max_tokens]
            new_row = row.to_dict()
            new_row[field] = encoding.decode(chunk_tokens)
            new_rows.append(new_row)
            i += max_tokens
    return pd.DataFrame(new_rows), {"chunked_by_tokens": max_tokens}, []

def sliding_window_chunking(df: pd.DataFrame, field: str = "text", max_tokens: int = 256, stride: int = 128, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    df = df.copy()
    encoding = tiktoken.get_encoding("cl100k_base")
    new_rows = []
    for _, row in df.iterrows():
        text = str(row[field])
        tokens = encoding.encode(text)
        i = 0
        while i < len(tokens):
            chunk_tokens = tokens[i:i + max_tokens]
            new_row = row.to_dict()
            new_row[field] = encoding.decode(chunk_tokens)
            new_rows.append(new_row)
            i += stride
    return pd.DataFrame(new_rows), {"sliding_window": {"size": max_tokens, "stride": stride}}, []

def sentence_boundary_chunking(df: pd.DataFrame, field: str = "text", max_sentences: int = 5, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    df = df.copy()
    new_rows = []
    for _, row in df.iterrows():
        text = str(row[field])
        doc = spacy_nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        i = 0
        while i < len(sentences):
            chunk_sents = sentences[i:i + max_sentences]
            new_row = row.to_dict()
            new_row[field] = " ".join(chunk_sents)
            new_rows.append(new_row)
            i += max_sentences
    return pd.DataFrame(new_rows), {"chunked_by_sentences": max_sentences}, []

def paragraph_level_chunking(df: pd.DataFrame, field: str = "text", max_paras: int = 3, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    df = df.copy()
    # Assume paras split by \n\n
    new_rows = []
    for _, row in df.iterrows():
        text = str(row[field])
        paras = text.split('\n\n')
        i = 0
        while i < len(paras):
            chunk_paras = paras[i:i + max_paras]
            new_row = row.to_dict()
            new_row[field] = "\n\n".join(chunk_paras)
            new_rows.append(new_row)
            i += max_paras
    return pd.DataFrame(new_rows), {"chunked_by_paragraphs": max_paras}, []

def hard_truncation(df: pd.DataFrame, field: str = "text", max_length: int = 1000, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    df = df.copy()
    df[field] = df[field].astype(str).str[:max_length]
    return df, {"hard_truncated": max_length}, []

def smart_truncation(df: pd.DataFrame, field: str = "text", max_length: int = 1000, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    df = df.copy()
    def smart_trunc(text):
        doc = spacy_nlp(text)
        truncated = " ".join([sent.text for sent in list(doc.sents)[:-1]])  # Keep all but last sentence
        if len(truncated) > max_length:
            truncated = truncated[:max_length]
        return truncated
    df[field] = df[field].astype(str).apply(smart_trunc)
    return df, {"smart_truncated": max_length}, []

OP_REGISTRY = {
    "chunk_by_token_count": chunk_by_token_count,
    "sliding_window_chunking": sliding_window_chunking,
    "sentence_boundary_chunking": sentence_boundary_chunking,
    "paragraph_level_chunking": paragraph_level_chunking,
    "hard_truncation": hard_truncation,
    "smart_truncation": smart_truncation,
}