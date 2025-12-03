# services/ops/nlp/prompt_ops.py
from typing import List, Dict, Any, Tuple
import pandas as pd
OP_REGISTRY = {}

def template_based_prompt_generation(df: pd.DataFrame, template: str = "Prompt: {text}", field: str = "prompt", **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    df = df.copy()
    def render(row):
        return template.format(**row.to_dict())
    df[field] = df.apply(render, axis=1)
    return df, {"prompt_template": template}, []

def template_based_completion_generation(df: pd.DataFrame, template: str = "Completion: {label}", field: str = "completion", **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    df = df.copy()
    def render(row):
        return template.format(**row.to_dict())
    df[field] = df.apply(render, axis=1)
    return df, {"completion_template": template}, []

def multi_column_prompt_merge(df: pd.DataFrame, columns: List[str] = None, separator: str = " ", field: str = "prompt", **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    df = df.copy()
    if columns:
        df[field] = df[columns].astype(str).agg(separator.join, axis=1)
    return df, {"merged_columns": columns}, []

def system_prefix_injection(df: pd.DataFrame, prefix: str = "System: ", field: str = "prompt", **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    df = df.copy()
    df[field] = prefix + df[field].astype(str)
    return df, {"system_prefix": prefix}, []

def role_based_formatting(df: pd.DataFrame, roles: Dict[str, str] = None, field: str = "messages", **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    df = df.copy()
    if roles:
        def format_roles(row):
            messages = []
            for col, role in roles.items():
                if col in row:
                    messages.append({"role": role, "content": str(row[col])})
            return messages
        df[field] = df.apply(format_roles, axis=1)
    return df, {"role_format": roles}, []

def chat_style_conversation_formatting(df: pd.DataFrame, columns: List[str] = ['user', 'assistant'], **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    df = df.copy()
    def chat_format(row):
        messages = []
        for i, col in enumerate(columns):
            role = 'user' if i % 2 == 0 else 'assistant'
            messages.append({"role": role, "content": str(row[col])})
        return messages
    df['chat_messages'] = df.apply(chat_format, axis=1)
    return df, {"chat_columns": columns}, []

def reinforcement_prompt_formatting(df: pd.DataFrame, template: str = "RL Prompt: {state} Action: {action}", **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    df = df.copy()
    def render(row):
        return template.format(**row.to_dict())
    df['rl_prompt'] = df.apply(render, axis=1)
    return df, {"rl_template": template}, []

OP_REGISTRY = {
    "template_based_prompt_generation": template_based_prompt_generation,
    "template_based_completion_generation": template_based_completion_generation,
    "multi_column_prompt_merge": multi_column_prompt_merge,
    "system_prefix_injection": system_prefix_injection,
    "role_based_formatting": role_based_formatting,
    "chat_style_conversation_formatting": chat_style_conversation_formatting,
    "reinforcement_prompt_formatting": reinforcement_prompt_formatting,
}