"""Quality-aware model-tier routing.

The user configures one model (their "good" one). Different agents have
very different reasoning needs:

    * Wrapped specialty agents (intake, profile, hardware, alchemy,
      restructure) narrate mechanical facts about structured inputs.
      A small fast model is perfectly fine and 50-100x cheaper.
    * The main AgenticLoop reasoning, RecoveryAgent, AuditAgent run the
      decisions that actually move quality (strategy, model selection,
      training-fix proposals). These should stay on the user's
      configured model.

Tier routing detects the user's configured model's quality class and
substitutes a same-provider sibling for the cheap calls. Concretely on
a free Gemini key with ``gemini-2.5-pro`` configured: intake/profile/
hardware calls land on ``gemini-2.0-flash-lite`` (~1500 RPD free) and
the 20 RPD Pro quota only gets spent on the 2-3 strategic decisions
per session. Practical session count goes from ~2/day to ~10/day on
the same key.

Configuration lives in ``data/model_tiers.json`` (auto-seeded with
sensible defaults on first run, user-editable, re-read on every call).
Set ``FT_AUTO_TIER_ROUTING=false`` to disable and pin every call to
the configured model.

The routing rule has one safety lock: it never UPGRADES the user's
choice. If they configured Flash to save money, requesting "strong"
returns Flash, not Pro. We may only swap downward to a cheaper sibling.
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Literal

from app.utils.config import settings


log = logging.getLogger("finetune-studio.agents.model_tiers")

Tier = Literal["fast", "medium", "strong"]
_TIER_ORDER: dict[str, int] = {"fast": 0, "medium": 1, "strong": 2}


# Defaults committed alongside the example file. The runtime file at
# ``data/model_tiers.json`` overrides these and may be edited freely.
# Provider keys are matched by substring against the lowercased
# provider name (so "google-gemini", "vertex-gemini" and bare "gemini"
# all hit the same row).
_DEFAULT_TIERS: dict[str, dict[str, str]] = {
    "gemini": {
        "fast":   "gemini-2.0-flash-lite",
        "medium": "gemini-2.5-flash",
        "strong": "gemini-2.5-pro",
    },
    "google": {
        "fast":   "gemini-2.0-flash-lite",
        "medium": "gemini-2.5-flash",
        "strong": "gemini-2.5-pro",
    },
    "vertex": {
        "fast":   "gemini-2.0-flash-lite",
        "medium": "gemini-2.5-flash",
        "strong": "gemini-2.5-pro",
    },
    "groq": {
        "fast":   "llama-3.1-8b-instant",
        "medium": "llama-3.3-70b-versatile",
        "strong": "llama-3.3-70b-versatile",
    },
    "openai": {
        "fast":   "gpt-4o-mini",
        "medium": "gpt-4o-mini",
        "strong": "gpt-4o",
    },
    "anthropic": {
        "fast":   "claude-haiku-4-5",
        "medium": "claude-sonnet-4-5",
        "strong": "claude-opus-4-5",
    },
    "deepseek": {
        "fast":   "deepseek-chat",
        "medium": "deepseek-chat",
        "strong": "deepseek-chat",
    },
    "mistral": {
        "fast":   "open-mistral-7b",
        "medium": "mistral-small-latest",
        "strong": "mistral-large-latest",
    },
    # OpenRouter / Together / Fireworks etc. proxy many models; we leave
    # them out of the defaults because the right small-model pick varies
    # too much per user. Users can add a row in the JSON if they want
    # tier routing on those providers.
}


# Token-level hints used to infer the user's configured model's tier.
# The model name is split on dashes/underscores/slashes/colons/whitespace
# (dots are KEPT so "0.5b" / "4.1" stay intact), then we look for any
# token equal to one of these hints. Substring matching is unsafe here
# because "mini" appears inside "gemini" — we'd wrongly downgrade
# gemini-2.5-pro to fast.
_TIER_HINTS_FAST = (
    "haiku", "instant", "mini", "nano", "tiny", "lite", "small",
    "0.5b", "1b", "1.5b", "3b", "7b", "8b",
)
_TIER_HINTS_MEDIUM = (
    "sonnet", "flash", "mixtral", "medium", "9b", "11b", "13b", "14b", "22b",
)
_TIER_HINTS_STRONG = (
    "opus", "pro", "ultra", "large", "4o", "4.1", "o1",
    "32b", "34b", "65b", "70b", "72b", "405b",
)


def _tokens(model: str) -> list[str]:
    """Split a model id into comparable tokens.

    Keeps dots so size suffixes like "0.5b" and version codes like "4.1"
    survive as single tokens. Splits on dashes, underscores, slashes,
    colons and whitespace.
    """
    return [t for t in re.split(r"[-_/:\s]+", (model or "").lower()) if t]


def _limits_path() -> Path:
    return settings.data_dir / "model_tiers.json"


def _example_path() -> Path:
    return settings.data_dir / "model_tiers.example.json"


def _load_tiers() -> dict[str, dict[str, str]]:
    path = _limits_path()
    if not path.exists():
        try:
            settings.data_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(_DEFAULT_TIERS, indent=2), encoding="utf-8")
            _example_path().write_text(json.dumps(_DEFAULT_TIERS, indent=2),
                                       encoding="utf-8")
            log.info("seeded model tiers at %s", path)
        except Exception as e:
            log.warning("could not seed model tiers file (%s); using in-memory defaults", e)
            return dict(_DEFAULT_TIERS)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("model_tiers.json must be a dict")
        return raw
    except Exception as e:
        log.warning("could not read model tiers (%s); using defaults", e)
        return dict(_DEFAULT_TIERS)


def _resolve_provider_row(provider: str,
                          model: str,
                          table: dict[str, dict[str, str]]) -> dict[str, str]:
    p = (provider or "").lower()
    m = (model or "").lower()

    # Priority 1: Exact/Substring match on the provider name.
    for key, row in table.items():
        if key in p:
            return row

    # Priority 2: If the provider is generic (openai/custom) or unknown,
    # look for hints in the model name (e.g. "llama" -> groq, "claude" -> anthropic).
    for key, row in table.items():
        if key in m:
            return row

    return {}


def infer_tier(model: str) -> Tier:
    """Best-effort guess of which tier the configured model belongs to.

    Order matters: fast hints are checked first because a name like
    "gpt-4o-mini" contains both "4o" (strong) and "mini" (fast) — the
    user picked the mini variant, so we honor that. Default is
    "strong" — if the user picked an unfamiliar model, assume they
    meant their best.
    """
    toks = _tokens(model)
    if not toks:
        return "strong"
    tok_set = set(toks)
    for hint in _TIER_HINTS_FAST:
        if hint in tok_set:
            return "fast"
    for hint in _TIER_HINTS_MEDIUM:
        if hint in tok_set:
            return "medium"
    for hint in _TIER_HINTS_STRONG:
        if hint in tok_set:
            return "strong"
    return "strong"


def _routing_enabled() -> bool:
    return os.getenv("FT_AUTO_TIER_ROUTING", "true").strip().lower() not in {
        "false", "0", "no", "off",
    }


def pick_model(provider: str, configured_model: str, tier: Tier) -> str:
    """Return the model id to call for the given (provider, tier).

    Never upgrades the user's choice — if they picked a medium model,
    a request for "strong" returns the same medium model. We may only
    swap downward to a cheaper sibling, never upward. When tier routing
    is disabled (env var) or no tier row exists for the provider, the
    configured model is returned unchanged.
    """
    if not _routing_enabled() or not configured_model:
        return configured_model
    table = _load_tiers()
    row = _resolve_provider_row(provider, configured_model, table)
    if not row:
        return configured_model

    configured_tier = infer_tier(configured_model)
    target_tier = tier if tier in _TIER_ORDER else "strong"

    # Safety lock: never escalate above the user's pick.
    if _TIER_ORDER[target_tier] >= _TIER_ORDER[configured_tier]:
        return configured_model

    candidate = row.get(target_tier)
    if not candidate:
        return configured_model
    return candidate


# ── Per-model context window budgets (May 2026) ───────────────────────────
#
# Used by AgenticLoop to size its rolling compaction window. A user on
# Sonnet 4.6 with 200K context can keep many more turns visible than a
# user on Llama-3.1-8B-instant with 8K context. Numbers err conservative
# (75% of the advertised window) because real-world effective context is
# always smaller — system prompts, tool schemas, and provider overhead
# eat the rest.

_CONTEXT_BUDGETS: dict[str, int] = {
    # Anthropic — 200K across the Claude 4.x family.
    "claude-opus":   180000,
    "claude-sonnet": 180000,
    "claude-haiku":  180000,
    # OpenAI — 128K standard, 1M for GPT-5 long-context variant.
    "gpt-5":         900000,   # the 1M variant
    "gpt-4o":        110000,
    "gpt-4.1":       110000,
    "o1":            100000,
    "o3":            100000,
    "o4":            100000,
    # Gemini — 2M flagship, 1M flash, 1M flash-lite.
    "gemini-2.5-pro":  1800000,
    "gemini-2.5-flash":  900000,
    "gemini-2.0-flash":  900000,
    "gemini-flash-lite": 900000,
    "gemini-1.5-pro":  1800000,
    "gemini-1.5-flash":  900000,
    # Groq — Llama-3.x 131K standard.
    "llama-3.3-70b-versatile": 110000,
    "llama-3.1-70b": 110000,
    "llama-3.1-8b-instant": 110000,
    "llama-3.1-8b": 110000,
    # Mistral
    "mistral-large": 110000,
    "mistral-small": 28000,
    "open-mistral":  28000,
    "codestral":     28000,
    # DeepSeek
    "deepseek-chat": 110000,
    "deepseek-r1":   55000,
    # Cohere Command R/R+
    "command-r-plus": 110000,
    "command-r":     110000,
    # Perplexity Sonar
    "sonar-pro":      100000,
    "sonar":          110000,
    "sonar-small":    11000,
    # Last-resort floor for unknown models (~ 8K, the smallest common limit).
    "default":       7000,
}


def context_budget(model: str) -> int:
    """Return the usable context budget (input tokens) for ``model``.

    Substring match against the model id; falls back to the conservative
    default when nothing matches. Operators can override by adding rows
    to ``data/model_context_budgets.json`` (auto-seeded on first run).
    """
    m = (model or "").lower()
    table = _load_context_budgets()
    # Longest-key-first so "gemini-2.5-pro" wins over "gemini-2.0-flash".
    for key in sorted(table.keys(), key=len, reverse=True):
        if key == "default":
            continue
        if key in m:
            try:
                return int(table[key])
            except (TypeError, ValueError):
                continue
    try:
        return int(table.get("default", _CONTEXT_BUDGETS["default"]))
    except (TypeError, ValueError):
        return _CONTEXT_BUDGETS["default"]


def _context_budgets_path() -> Path:
    return settings.data_dir / "model_context_budgets.json"


def _load_context_budgets() -> dict[str, int]:
    path = _context_budgets_path()
    if not path.exists():
        try:
            settings.data_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(_CONTEXT_BUDGETS, indent=2), encoding="utf-8")
        except Exception:
            return dict(_CONTEXT_BUDGETS)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            return {k: int(v) for k, v in raw.items() if isinstance(v, (int, float))}
    except Exception:
        pass
    return dict(_CONTEXT_BUDGETS)


def recommended_keep_last_k(model: str, *, target_fill: float = 0.6) -> int:
    """Recommend a rolling-window depth (in logical turns) based on the
    model's context budget.

    Calibrated so that on average a turn ≈ 2-3K tokens of prompt + tool
    payload; we keep last K turns plus the first user message, summarise
    the middle. target_fill defaults to 0.6, mirroring Claude Code's
    "proactively compact at 60% fill" heuristic.
    """
    budget = context_budget(model)
    avg_turn_tokens = 2500.0
    headroom = max(1.0, budget * float(target_fill))
    raw_k = int(headroom // avg_turn_tokens)
    # Clamp to a sane band: never less than 3 (otherwise tool-call turns
    # orphan their results); never more than 32 (otherwise compaction
    # provides no token savings on huge-context models).
    return max(3, min(32, raw_k))
