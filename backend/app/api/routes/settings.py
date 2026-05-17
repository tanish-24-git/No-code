"""Settings: LLM provider config, HF token, behaviour flags.

Resolution order for LLM and HF credentials:
    1. Values saved in data/settings.json (set via the UI).
    2. Environment variables / .env (LLM_PROVIDER, LLM_API_KEY, LLM_MODEL,
       LLM_BASE_URL, HF_TOKEN).

Sensitive values are encrypted at rest using a Fernet key under
data/.encryption_key (auto-generated).
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
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


log = logging.getLogger("finetune-studio.settings")

router = APIRouter(prefix="/api/settings", tags=["settings"])


_DEFAULTS: dict[str, Any] = {
    "llm_provider": None,
    "llm_model": "",
    "llm_base_url": "",
    "llm_key_enc": "",
    # JSON-serialised list of encrypted keys. Empty string when only the
    # singular llm_key_enc is configured. Activates key rotation in
    # providers._resolve_api_key (one bucket per key = N× effective
    # throughput on paid tiers / multiple free accounts).
    "llm_keys_enc": "",
    "hf_token_enc": "",
    "hf_username": None,
    "auto_config_on_upload": True,
    "show_agent_reasoning": True,
    "updated_at": None,
}


def _canon_provider(provider: str | None) -> str:
    """Mirror the canonicalisation rule in providers._candidate_env_vars."""
    return (
        (provider or "")
        .upper()
        .replace("-", "_")
        .replace(".", "_")
        .replace("/", "_")
        .replace(" ", "_")
    )


def _apply_keys_to_env(provider: str | None, keys: list[str]) -> None:
    """Mirror UI-saved plural keys into the env vars that
    providers._read_multi_keys() looks up. This is the bridge that lets
    a user enter 3 paid keys in Settings and have the next LLM call
    actually rotate across them without any provider-side change."""
    if not provider or not keys:
        return
    canon = _canon_provider(provider)
    joined = ",".join(k for k in keys if k)
    if not joined:
        return
    os.environ[f"{canon}_API_KEYS"] = joined
    # Also set the generic fallback so callers that don't pass a provider
    # still hit the rotation pool.
    os.environ.setdefault("LLM_API_KEYS", joined)


def _clear_keys_from_env(provider: str | None) -> None:
    if not provider:
        return
    canon = _canon_provider(provider)
    os.environ.pop(f"{canon}_API_KEYS", None)


def _decode_keys_enc(blob: str) -> list[str]:
    if not blob:
        return []
    try:
        encoded = json.loads(blob)
        if not isinstance(encoded, list):
            return []
        out: list[str] = []
        for e in encoded:
            if not isinstance(e, str):
                continue
            try:
                dec = crypto.decrypt(e)
            except Exception:
                continue
            if dec:
                out.append(dec)
        return out
    except Exception:
        return []


def _encode_keys_enc(keys: list[str]) -> str:
    encoded = []
    for k in keys:
        if not k:
            continue
        encoded.append(crypto.encrypt(k))
    return json.dumps(encoded)


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
    api_keys: list[str] = field(default_factory=list)


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
    ui_keys = _decode_keys_enc(s.get("llm_keys_enc", ""))
    ui_model = s.get("llm_model") or ""
    ui_base = s.get("llm_base_url") or ""
    if ui_provider and ui_model and _has_required_key(ui_provider, ui_key or (ui_keys[0] if ui_keys else "")):
        # Prefer the plural list's first entry as the singular fallback if
        # only plural keys are configured (e.g. user pasted three keys
        # without setting a single api_key).
        canonical_single = ui_key or (ui_keys[0] if ui_keys else "")
        return LLMConfig(
            provider=ui_provider,
            api_key=canonical_single,
            model=ui_model,
            base_url=ui_base,
            source="ui",
            api_keys=ui_keys,
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


def get_llm_keys() -> list[str]:
    """Public accessor used by providers._resolve_api_key.

    Returns the plural key list when the user has saved more than one,
    else an empty list (caller should fall back to env vars / singular
    key). Lives here so providers.py doesn't have to know about
    encryption or the settings file layout.
    """
    cfg = get_llm_config()
    return list(cfg.api_keys)


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
        llm_api_keys_count=len(llm.api_keys),
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
    if payload.api_keys is not None:
        clean_keys = [k.strip() for k in payload.api_keys if k and k.strip()]
        s["llm_keys_enc"] = _encode_keys_enc(clean_keys) if clean_keys else ""
        # Best-effort: keep the singular key in sync with the first plural
        # so legacy callers still see *something* if they bypass rotation.
        if clean_keys and not s.get("llm_key_enc"):
            s["llm_key_enc"] = crypto.encrypt(clean_keys[0])
        # Apply to env immediately so the next LLM call rotates.
        _apply_keys_to_env(payload.provider, clean_keys)
    _save_raw(s)
    return _redact()


@router.delete("/llm", response_model=SettingsRead)
def clear_llm() -> SettingsRead:
    s = _load_raw()
    provider = s.get("llm_provider")
    s["llm_provider"] = None
    s["llm_model"] = ""
    s["llm_base_url"] = ""
    s["llm_key_enc"] = ""
    s["llm_keys_enc"] = ""
    _save_raw(s)
    _clear_keys_from_env(provider)
    return _redact()


# ── Module-load: restore env-var bridge from saved settings ────────────────
#
# When the process boots, mirror any UI-saved plural keys into the env vars
# that providers._read_multi_keys() consults. This makes the rotation
# functional from the very first LLM call even on a cold restart, without
# requiring the user to re-save their settings.
try:
    _boot_s = _load_raw()
    _boot_keys = _decode_keys_enc(_boot_s.get("llm_keys_enc", ""))
    if _boot_keys:
        _apply_keys_to_env(_boot_s.get("llm_provider"), _boot_keys)
        log.info("Restored %d plural API key(s) to env for provider=%s",
                 len(_boot_keys), _boot_s.get("llm_provider"))
except Exception:
    log.exception("Failed to restore plural API keys from settings at boot")


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
