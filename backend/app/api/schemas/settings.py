"""Settings schemas. Sensitive values (LLM API key, HF token) are stored
encrypted; the API only returns masked previews. Provider names are
validated at runtime against the registry, not in the type system, so the
list stays editable from one place."""
from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


# Provider is just `str` here; the route validates it against the registry.
Provider = str


class SettingsRead(BaseModel):
    llm_provider: Optional[Provider] = None
    llm_model: str = ""
    llm_base_url: str = ""
    llm_api_key_masked: Optional[str] = None
    llm_api_key_set: bool = False
    llm_source: Literal["env", "ui", "unset"] = "unset"

    hf_token_masked: Optional[str] = None
    hf_token_set: bool = False
    hf_username: Optional[str] = None
    hf_source: Literal["env", "ui", "unset"] = "unset"

    auto_config_on_upload: bool = True
    show_agent_reasoning: bool = True

    is_configured: bool = False


class LLMConfigUpdate(BaseModel):
    provider: Provider
    api_key: Optional[str] = None  # omit to keep the existing key
    model: str = Field(..., min_length=1)
    base_url: Optional[str] = None


class HFTokenUpdate(BaseModel):
    token: str = Field(..., min_length=10)


class SettingsUpdate(BaseModel):
    auto_config_on_upload: Optional[bool] = None
    show_agent_reasoning: Optional[bool] = None


class VerifyResult(BaseModel):
    valid: bool
    detail: Optional[str] = None
    username: Optional[str] = None
    models: Optional[list[str]] = None


class ProviderInfo(BaseModel):
    name: str
    label: str
    engine: Literal["anthropic", "openai"]
    base_url: str
    needs_key: bool
    sample_models: list[str]
    notes: str = ""
