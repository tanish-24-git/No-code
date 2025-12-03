# services/ops/nlp/cleaning_ops.py
from typing import List, Dict, Any, Tuple
import re
import pandas as pd
import spacy
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
spacy_nlp = spacy.load("en_core_web_sm")
OP_REGISTRY = {}

def lowercase_text(df: pd.DataFrame, field: str = "text", **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    df = df.copy()
    if field in df.columns:
        df[field] = df[field].astype(str).str.lower()
    return df, {}, []

def uppercase_text(df: pd.DataFrame, field: str = "text", **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    df = df.copy()
    if field in df.columns:
        df[field] = df[field].astype(str).str.upper()
    return df, {}, []

def remove_punctuation(df: pd.DataFrame, field: str = "text", **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    df = df.copy()
    if field in df.columns:
        df[field] = df[field].astype(str).str.replace(r"[^\w\s]", "", regex=True)
    return df, {}, []

def remove_stopwords(df: pd.DataFrame, field: str = "text", **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    df = df.copy()
    if field not in df.columns:
        return df, {}, []
    stop_words = set(stopwords.words('english'))
    def drop_sw(text: str) -> str:
        tokens = str(text).split()
        return " ".join([t for t in tokens if t.lower() not in stop_words])
    df[field] = df[field].apply(drop_sw)
    return df, {"stopwords_removed": True}, []

def lemmatization(df: pd.DataFrame, field: str = "text", **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    df = df.copy()
    if field not in df.columns:
        return df, {}, []
    lemmatizer = WordNetLemmatizer()
    def lemm(text: str) -> str:
        doc = spacy_nlp(text)
        return " ".join([token.lemma_ for token in doc])
    df[field] = df[field].apply(lemm)
    return df, {"lemmatized": True}, []

def stemming(df: pd.DataFrame, field: str = "text", **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    df = df.copy()
    if field not in df.columns:
        return df, {}, []
    stemmer = PorterStemmer()
    def stem(text: str) -> str:
        tokens = str(text).split()
        return " ".join([stemmer.stem(t) for t in tokens])
    df[field] = df[field].apply(stem)
    return df, {"stemmed": True}, []

def remove_emojis(df: pd.DataFrame, field: str = "text", **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    df = df.copy()
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags
        "]+", flags=re.UNICODE)
    if field in df.columns:
        df[field] = df[field].astype(str).apply(lambda x: emoji_pattern.sub(r'', x))
    return df, {}, []

def remove_urls(df: pd.DataFrame, field: str = "text", **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    df = df.copy()
    pattern = re.compile(r"http[s]?://\S+")
    if field in df.columns:
        df[field] = df[field].astype(str).apply(lambda x: pattern.sub("", x))
    return df, {}, []

def strip_html(df: pd.DataFrame, field: str = "text", **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    df = df.copy()
    pattern = re.compile(r"<.*?>")
    if field in df.columns:
        df[field] = df[field].astype(str).apply(lambda x: pattern.sub("", x))
    return df, {}, []

OP_REGISTRY = {
    "lowercase_text": lowercase_text,
    "uppercase_text": uppercase_text,
    "remove_punctuation": remove_punctuation,
    "remove_stopwords": remove_stopwords,
    "lemmatization": lemmatization,
    "stemming": stemming,
    "remove_emojis": remove_emojis,
    "remove_urls": remove_urls,
    "strip_html": strip_html,
}