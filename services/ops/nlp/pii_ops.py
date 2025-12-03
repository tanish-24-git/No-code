# services/ops/nlp/pii_ops.py
from typing import List, Dict, Any, Tuple
import pandas as pd
import re
from better_profanity import profanity  # Add to requirements
OP_REGISTRY = {}

def email_redaction(df: pd.DataFrame, field: str = "text", mask: str = "<EMAIL>", **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    df = df.copy()
    pattern = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
    df[field] = df[field].astype(str).apply(lambda x: pattern.sub(mask, x))
    return df, {"email_redacted": True}, []

def phone_number_redaction(df: pd.DataFrame, field: str = "text", mask: str = "<PHONE>", **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    df = df.copy()
    pattern = re.compile(r"\+?1?[-.\s]?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})")
    df[field] = df[field].astype(str).apply(lambda x: pattern.sub(mask, x))
    return df, {"phone_redacted": True}, []

def address_redaction(df: pd.DataFrame, field: str = "text", mask: str = "<ADDRESS>", **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    df = df.copy()
    pattern = re.compile(r'\d{1,5}\s\w+\s(?:St|Rd|Ave|Dr|Ln|Blvd|Way|Ct)\.?\s\w+')
    df[field] = df[field].astype(str).apply(lambda x: pattern.sub(mask, x))
    return df, {"address_redacted": True}, []

def id_masking(df: pd.DataFrame, field: str = "text", mask: str = "<ID>", **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    df = df.copy()
    # UUID and simple ID patterns
    uuid_pattern = re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}')
    id_pattern = re.compile(r'\b\d{10,12}\b')  # e.g., Aadhar-like
    def mask_ids(text):
        text = uuid_pattern.sub(mask, text)
        text = id_pattern.sub(mask, text)
        return text
    df[field] = df[field].astype(str).apply(mask_ids)
    return df, {"id_masked": True}, []

def profanity_removal(df: pd.DataFrame, field: str = "text", mask: str = "***", **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    df = df.copy()
    df[field] = df[field].astype(str).apply(lambda x: profanity.censor(x, mask))
    return df, {"profanity_censored": True}, []

def regex_custom_pii_removal(df: pd.DataFrame, field: str = "text", pattern: str = None, mask: str = "<PII>", **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    if not pattern:
        return df, {}, ["Pattern required for custom PII"]
    df = df.copy()
    compiled = re.compile(pattern)
    df[field] = df[field].astype(str).apply(lambda x: compiled.sub(mask, x))
    return df, {"custom_pii_pattern": pattern}, []

def strict_pii_mode(df: pd.DataFrame, field: str = "text", **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    # Chain common PII
    df, _, _ = email_redaction(df, field)
    df, _, _ = phone_number_redaction(df, field)
    df, _, _ = address_redaction(df, field)
    df, _, _ = id_masking(df, field)
    return df, {"strict_pii_applied": True}, []

def ip_address_masking(df: pd.DataFrame, field: str = "text", mask: str = "<IP>", **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    df = df.copy()
    pattern = re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b')
    df[field] = df[field].astype(str).apply(lambda x: pattern.sub(mask, x))
    return df, {"ip_masked": True}, []

OP_REGISTRY = {
    "email_redaction": email_redaction,
    "phone_number_redaction": phone_number_redaction,
    "address_redaction": address_redaction,
    "id_masking": id_masking,
    "profanity_removal": profanity_removal,
    "regex_custom_pii_removal": regex_custom_pii_removal,
    "strict_pii_mode": strict_pii_mode,
    "ip_address_masking": ip_address_masking,
}