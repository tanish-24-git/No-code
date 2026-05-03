"""Settings: LLM provider config, HF token, behaviour flags.

Resolution order for LLM and HF credentials:
    1. Values saved in data/settings.json (set via the UI).
    2. Environment variables / .env (LLM_PROVIDER, LLM_API_KEY, LLM_MODEL,
       LLM_BASE_URL, HF_TOKEN).

Sensitive values are encrypted at rest using a Fernet key under
data/.encryption_key (auto-generated).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

import httpx
from fastapi import APIRouter, HTTPException

from app.agents.registry import PROVIDERS, get_spec, resolve_base_url
from app.api.schemas.settings import (
    HFTokenUpdate,
    LLMConfigUpdate,
    ProviderInfo,
    SettingsRead,
    SettingsUpdate,
    VerifyResult,
)
from app.storage import store
from app.utils import crypto
from app.utils.config import settings as env_settings


router = APIRouter(prefix="/api/settings", tags=["settings"])


_DEFAULTS: dict[str, Any] = {
    "llm_provider": None,
    "llm_model": "",
    "llm_base_url": "",
    "llm_key_enc": "",
    "hf_token_enc": "",
    "hf_username": None,
    "auto_config_on_upload": True,
    "show_agent_reasoning": True,
    "updated_at": None,
}


def _load_raw() -> dict[str, Any]:
    s = store.read_singleton("settings") or {}
    return {**_DEFAULTS, **s}


def _save_raw(s: dict[str, Any]) -> None:
    s["updated_at"] = datetime.now(timezone.utc).isoformat()
    store.write_singleton("settings", s)


# ── LLM config resolution (UI overrides env) ───────────────────────────────

@dataclass
class LLMConfig:
    provider: str | None
    api_key: str
    model: str
    base_url: str
    source: Literal["env", "ui", "unset"]


def _has_required_key(provider: str | None, api_key: str) -> bool:
    """Local providers (Ollama, LM Studio, vLLM, custom) do not need a key.
    Everything cloud-y does."""
    spec = get_spec(provider)
    if not spec:
        return bool(api_key)
    return bool(api_key) if spec.needs_key else True


def get_llm_config() -> LLMConfig:
    s = _load_raw()
    ui_provider = s.get("llm_provider")
    ui_key = crypto.decrypt(s.get("llm_key_enc", "")) if s.get("llm_key_enc") else ""
    ui_model = s.get("llm_model") or ""
    ui_base = s.get("llm_base_url") or ""
    if ui_provider and ui_model and _has_required_key(ui_provider, ui_key):
        return LLMConfig(
            provider=ui_provider, api_key=ui_key, model=ui_model, base_url=ui_base, source="ui",
        )
    if env_settings.llm_provider and env_settings.llm_model and _has_required_key(
        env_settings.llm_provider, env_settings.llm_api_key,
    ):
        return LLMConfig(
            provider=env_settings.llm_provider,
            api_key=env_settings.llm_api_key,
            model=env_settings.llm_model,
            base_url=env_settings.llm_base_url,
            source="env",
        )
    return LLMConfig(provider=None, api_key="", model="", base_url="", source="unset")


def get_hf_token_resolved() -> tuple[str, Literal["env", "ui", "unset"]]:
    s = _load_raw()
    if s.get("hf_token_enc"):
        return crypto.decrypt(s["hf_token_enc"]), "ui"
    if env_settings.hf_token:
        return env_settings.hf_token, "env"
    return "", "unset"


def get_hf_token() -> str:
    return get_hf_token_resolved()[0]


# ── Public read / write ────────────────────────────────────────────────────

def _redact() -> SettingsRead:
    s = _load_raw()
    llm = get_llm_config()
    hf_token, hf_source = get_hf_token_resolved()
    is_configured = bool(llm.provider and llm.model and _has_required_key(llm.provider, llm.api_key))
    return SettingsRead(
        llm_provider=llm.provider,
        llm_model=llm.model,
        llm_base_url=llm.base_url,
        llm_api_key_masked=crypto.mask(llm.api_key) or None,
        llm_api_key_set=bool(llm.api_key),
        llm_source=llm.source,
        hf_token_masked=crypto.mask(hf_token) or None,
        hf_token_set=bool(hf_token),
        hf_username=s.get("hf_username"),
        hf_source=hf_source,
        auto_config_on_upload=bool(s.get("auto_config_on_upload", True)),
        show_agent_reasoning=bool(s.get("show_agent_reasoning", True)),
        is_configured=is_configured,
    )


@router.get("", response_model=SettingsRead)
def read_settings() -> SettingsRead:
    return _redact()


@router.put("", response_model=SettingsRead)
def update_settings(payload: SettingsUpdate) -> SettingsRead:
    s = _load_raw()
    for k, v in payload.model_dump(exclude_none=True).items():
        s[k] = v
    _save_raw(s)
    return _redact()


@router.get("/providers", response_model=list[ProviderInfo])
def list_providers() -> list[ProviderInfo]:
    """The full provider catalogue. The UI reads this to render the provider
    dropdown and to auto-fill the default base URL."""
    return [
        ProviderInfo(
            name=p.name,
            label=p.label,
            engine=p.engine,
            base_url=p.base_url,
            needs_key=p.needs_key,
            sample_models=list(p.sample_models),
            notes=p.notes,
        )
        for p in PROVIDERS.values()
    ]


@router.post("/llm", response_model=SettingsRead)
def set_llm(payload: LLMConfigUpdate) -> SettingsRead:
    if payload.provider not in PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown provider {payload.provider!r}. "
            f"Valid options: {', '.join(sorted(PROVIDERS.keys()))}",
        )
    s = _load_raw()
    s["llm_provider"] = payload.provider
    s["llm_model"] = payload.model
    s["llm_base_url"] = payload.base_url or ""
    if payload.api_key is not None:
        s["llm_key_enc"] = crypto.encrypt(payload.api_key) if payload.api_key else ""
    _save_raw(s)
    return _redact()


@router.delete("/llm", response_model=SettingsRead)
def clear_llm() -> SettingsRead:
    s = _load_raw()
    s["llm_provider"] = None
    s["llm_model"] = ""
    s["llm_base_url"] = ""
    s["llm_key_enc"] = ""
    _save_raw(s)
    return _redact()


@router.post("/hf-token", response_model=SettingsRead)
def set_hf_token(payload: HFTokenUpdate) -> SettingsRead:
    s = _load_raw()
    s["hf_token_enc"] = crypto.encrypt(payload.token)
    _save_raw(s)
    return _redact()


@router.delete("/hf-token", response_model=SettingsRead)
def clear_hf_token() -> SettingsRead:
    s = _load_raw()
    s["hf_token_enc"] = ""
    s["hf_username"] = None
    _save_raw(s)
    return _redact()


# ── Verification probes ────────────────────────────────────────────────────

@router.post("/verify-llm", response_model=VerifyResult)
async def verify_llm() -> VerifyResult:
    """Probe the configured provider's models endpoint. Works for both engines
    by hitting the resolved base URL with the right auth header shape."""
    cfg = get_llm_config()
    if not cfg.provider:
        return VerifyResult(valid=False, detail="No LLM provider configured")

    spec = get_spec(cfg.provider)
    base = resolve_base_url(cfg.provider, cfg.base_url or None).rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            if spec and spec.engine == "anthropic":
                r = await client.get(
                    base + "/v1/models",
                    headers={"x-api-key": cfg.api_key, "anthropic-version": "2023-06-01"},
                )
            else:
                # OpenAI engine: most providers serve /models under the base URL.
                # If the base already ends in /v1 we just append /models.
                url = base + ("/models" if base.endswith("/v1") else "/v1/models")
                headers = {"Authorization": f"Bearer {cfg.api_key}"} if cfg.api_key else {}
                r = await client.get(url, headers=headers)
        if r.status_code != 200:
            return VerifyResult(valid=False, detail=f"HTTP {r.status_code}: {r.text[:200]}")
        data = r.json()
        ids: list[str] = []
        if "data" in data:
            ids = [m.get("id", "") for m in data["data"] if isinstance(m, dict)]
        return VerifyResult(valid=True, detail=f"{cfg.provider} reachable", models=ids[:50])
    except Exception as e:
        return VerifyResult(valid=False, detail=str(e))


@router.post("/verify-hf", response_model=VerifyResult)
async def verify_hf_token() -> VerifyResult:
    token, source = get_hf_token_resolved()
    if not token:
        return VerifyResult(valid=False, detail="No HF token set")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(
                "https://huggingface.co/api/whoami-v2",
                headers={"Authorization": f"Bearer {token}"},
            )
        if r.status_code == 200:
            data = r.json()
            username = data.get("name") or data.get("fullname")
            if source == "ui":
                s = _load_raw()
                s["hf_username"] = username
                _save_raw(s)
            return VerifyResult(valid=True, username=username, detail="HF token OK")
        return VerifyResult(valid=False, detail=f"HTTP {r.status_code}")
    except Exception as e:
        return VerifyResult(valid=False, detail=str(e))


# Backwards-compat shim for any old import sites.
def get_agent_key() -> str:
    return get_llm_config().api_key
