"""Provider-level rate-limit gate (RPM + sliding-window TPM).

This is the single chokepoint through which every LLM HTTP call passes
(both the sync `stream_chat` and async `stream_chat_async` in
providers.py). It serializes calls per provider so free-tier users on
Gemini, Groq, etc. never get RESOURCE_EXHAUSTED from burst traffic
emitted by parallel agents (ping_llm + IntakeAgent + AgenticLoop +
RecoveryAgent each hold their own asyncio Task but share the same
quota).

Two enforcement bands:

    RPM   minimum wall-clock gap between consecutive completions on
          the same provider. Implemented as a per-provider "next allowed
          start" timestamp.
    TPM   sliding 60-second window of estimated prompt tokens. A
          reservation deducts the estimate up-front so concurrent
          reservers see the load; once the response returns, the caller
          may correct the bookkeeping via report_actual_tokens.

Per-provider limits live in `data/provider_limits.json` and are
auto-seeded with sensible defaults on first run so the open-source path
works zero-config. Operators can edit the file at any time — it is
re-read on every reservation.

Multi-key rotation is opt-in. Set `<PROVIDER>_API_KEYS=a,b,c` (plural)
and each key gets its own bucket inside the gate, giving N-times the
effective RPM/TPM. The selection helper lives in providers.py because it
needs to know the provider string; the gate just keys state by
(provider_canonical, key_index).
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from app.utils.config import settings


log = logging.getLogger("finetune-studio.agents.rate_limiter")


# Default limits — calibrated to PAID-TIER entry baselines per provider's
# public docs as of May 2026. Paid keys get the throughput they paid for;
# free-tier users on the same code will hit 429s (which surface to the UI
# via the classifier in base.py — that's the user's billing problem, not
# ours). Per-user overrides go in data/provider_limits.json. Users with
# higher tiers can either bump these manually OR enable response-header
# auto-tuning (see `_apply_response_headers` below).
#
# Numbers are sourced from official docs (Anthropic Tier 2, OpenAI Tier 1,
# Gemini paid Tier 1, Groq Dev, Mistral Scale, Cohere Production, etc.).
# Bump _LIMITS_VERSION and rotate values when providers re-tier.
_LIMITS_VERSION = 2
_DEFAULT_LIMITS: dict[str, dict[str, float]] = {
    # Anthropic Tier 2 ($40 spent): 1000 RPM, 80K ITPM, 16K OTPM for Sonnet.
    # Higher tiers (4: 4000 RPM, 2M ITPM Sonnet) need a manual bump.
    "anthropic": {"rpm": 1000, "tpm": 80000,  "min_interval_sec": 0.06},
    # OpenAI Tier 1 ($5 spent): 500 RPM, 30K TPM for GPT-4o, 500K TPM GPT-5.
    "openai":    {"rpm": 500,  "tpm": 500000, "min_interval_sec": 0.12},
    # Gemini paid Tier 1: 150-300 RPM, 1M TPM, RPD limits much higher.
    "gemini":    {"rpm": 200,  "tpm": 1000000, "min_interval_sec": 0.3},
    "google":    {"rpm": 200,  "tpm": 1000000, "min_interval_sec": 0.3},
    "vertex":    {"rpm": 200,  "tpm": 1000000, "min_interval_sec": 0.3},
    # Groq Dev tier (paid w/ credit card): ~10x free; 60-120 RPM, 60-120K TPM.
    "groq":      {"rpm": 100,  "tpm": 100000, "min_interval_sec": 0.6},
    # Mistral Scale (pay-as-you-go): 300 RPM, 500K TPM.
    "mistral":   {"rpm": 300,  "tpm": 500000, "min_interval_sec": 0.2},
    # DeepSeek dynamic; baseline ~300 RPM, 200K TPM. May 429 under load.
    "deepseek":  {"rpm": 300,  "tpm": 200000, "min_interval_sec": 0.2},
    # Perplexity Sonar Tier 1 (paid >= $50 cumulative): conservative 60 RPM.
    "perplexity": {"rpm": 60,  "tpm": 100000, "min_interval_sec": 1.0},
    # Together AI: response-header driven; conservative model-agnostic baseline.
    "together":  {"rpm": 60,   "tpm": 500000, "min_interval_sec": 1.0},
    # Cohere production keys: chat 500 RPM, embed 2000 RPM.
    "cohere":    {"rpm": 500,  "tpm": 1000000, "min_interval_sec": 0.12},
    # OpenRouter paid models: NO platform-level limit (upstream applies).
    # Set high so our gate doesn't add artificial cap; upstream 429s surface.
    "openrouter": {"rpm": 1000, "tpm": 2000000, "min_interval_sec": 0.06},
    # Fallback for unknown providers: assume moderate paid tier.
    "default":   {"rpm": 200,  "tpm": 500000, "min_interval_sec": 0.3},
}


def _limits_path() -> Path:
    return settings.data_dir / "provider_limits.json"


def _example_path() -> Path:
    return settings.data_dir / "provider_limits.example.json"


def _seeded_table() -> dict[str, Any]:
    """Build the on-disk seed: defaults + version tag for migration."""
    out: dict[str, Any] = {"_version": _LIMITS_VERSION}
    out.update(_DEFAULT_LIMITS)
    return out


def _load_limits() -> dict[str, dict[str, float]]:
    """Read the runtime limits file. Auto-seed on first run.

    Returns the in-memory limit table. The lookup is substring-based
    against the lowercased provider name (so "google-gemini",
    "vertex-ai-gemini" and bare "gemini" all map to the gemini row).

    Migration: when the on-disk file predates the current
    _LIMITS_VERSION, we write the new defaults to
    `provider_limits.example.json` so the user can diff/merge, but
    NEVER overwrite their `provider_limits.json` — they may have
    custom enterprise-tier values we shouldn't clobber.
    """
    path = _limits_path()
    if not path.exists():
        try:
            settings.data_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(_seeded_table(), indent=2),
                            encoding="utf-8")
            _example_path().write_text(json.dumps(_seeded_table(), indent=2),
                                       encoding="utf-8")
            log.info("seeded provider limits at %s (v%d)", path, _LIMITS_VERSION)
        except Exception as e:
            log.warning("could not seed provider limits file (%s); using in-memory defaults", e)
            return dict(_DEFAULT_LIMITS)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("provider_limits.json must be a dict")
        loaded_version = int(raw.get("_version", 1))
        if loaded_version < _LIMITS_VERSION:
            # Rewrite the example so the user can see what's new without
            # losing their custom row(s).
            try:
                _example_path().write_text(
                    json.dumps(_seeded_table(), indent=2),
                    encoding="utf-8",
                )
                log.info(
                    "provider_limits.json is v%d (current is v%d); "
                    "wrote new defaults to %s — diff and merge if you "
                    "want the latest paid-tier baselines.",
                    loaded_version, _LIMITS_VERSION, _example_path(),
                )
            except Exception:
                pass
        # Strip the version tag so downstream lookups don't try to match
        # "_version" as a provider name.
        raw.pop("_version", None)
        return raw
    except Exception as e:
        log.warning("could not read provider limits (%s); using defaults", e)
        return dict(_DEFAULT_LIMITS)


def _resolve_limit_row(provider: str, table: dict[str, dict[str, float]]) -> dict[str, float]:
    p = (provider or "").lower()
    for key, row in table.items():
        if key == "default":
            continue
        if key in p:
            return row
    return table.get("default", _DEFAULT_LIMITS["default"])


@dataclass
class _BucketState:
    """Per (provider, key_index) state. Holds the sliding-window of token
    reservations and the next-allowed-start timestamp for RPM gating."""
    next_allowed_start: float = 0.0
    # Each entry: (timestamp_seconds, tokens). Pruned to last 60s on read.
    token_window: list[tuple[float, int]] = field(default_factory=list)


class ProviderGate:
    """Thread- and asyncio-safe gate. One module-level instance.

    Usage:
        async with PROVIDER_GATE.reserve_async("gemini", 5000):
            ... stream call ...
        PROVIDER_GATE.report_actual_tokens("gemini", actual_prompt_tokens)

        with PROVIDER_GATE.reserve_sync("gemini", 5000):
            ... sync stream call ...
    """

    def __init__(self) -> None:
        self._limits = _load_limits()
        self._buckets: dict[tuple[str, int], _BucketState] = {}
        # asyncio locks are loop-bound; lazy-create on first use so the
        # gate object can be constructed at import time (no running loop).
        self._async_locks: dict[tuple[str, int], asyncio.Lock] = {}
        # Single coarse sync lock guards bucket state mutation from
        # threaded callers (legacy sync stream_chat path).
        self._sync_lock = threading.Lock()
        # Last bucket reserved per provider — used by report_actual_tokens
        # to attribute the correction to the right (provider, key) pair.
        self._last_bucket: dict[str, tuple[str, int]] = {}

    # ── Limit reload (hot) ────────────────────────────────────────────

    def reload_limits(self) -> None:
        self._limits = _load_limits()

    def _row_for(self, provider: str) -> dict[str, float]:
        return _resolve_limit_row(provider, self._limits)

    # ── Bookkeeping helpers ───────────────────────────────────────────

    def _bucket(self, provider: str, key_index: int) -> _BucketState:
        canon = _canon(provider)
        return self._buckets.setdefault((canon, key_index), _BucketState())

    def _async_lock(self, provider: str, key_index: int) -> asyncio.Lock:
        canon = _canon(provider)
        key = (canon, key_index)
        lock = self._async_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._async_locks[key] = lock
        return lock

    def _prune_window(self, bucket: _BucketState, now: float) -> int:
        """Drop entries older than 60s. Return current token sum."""
        cutoff = now - 60.0
        bucket.token_window = [(t, n) for (t, n) in bucket.token_window if t >= cutoff]
        return sum(n for _, n in bucket.token_window)

    def _compute_wait(
        self,
        bucket: _BucketState,
        row: dict[str, float],
        estimated_tokens: int,
        now: float,
    ) -> float:
        """How long the caller must wait before its call may proceed."""
        # RPM gate: wait until next_allowed_start.
        rpm_wait = max(0.0, bucket.next_allowed_start - now)

        # TPM gate: if reserving would push the window over the cap, wait
        # for the oldest entry to age out enough to make room.
        tpm = float(row.get("tpm", 0) or 0)
        used = self._prune_window(bucket, now)
        tpm_wait = 0.0
        if tpm > 0 and used + max(0, estimated_tokens) > tpm:
            # Sort the window by age and find the smallest aged-prefix
            # whose departure frees enough headroom.
            entries = sorted(bucket.token_window, key=lambda x: x[0])
            shortfall = (used + estimated_tokens) - int(tpm)
            freed = 0
            wait_to = now
            for ts, n in entries:
                freed += n
                wait_to = max(wait_to, ts + 60.0 + 0.05)
                if freed >= shortfall:
                    break
            tpm_wait = max(0.0, wait_to - now)

        return max(rpm_wait, tpm_wait)

    def _commit(
        self,
        bucket: _BucketState,
        row: dict[str, float],
        estimated_tokens: int,
        now: float,
        provider: str,
        key_index: int,
    ) -> None:
        bucket.next_allowed_start = now + float(row.get("min_interval_sec", 0) or 0)
        if estimated_tokens > 0:
            bucket.token_window.append((now, int(estimated_tokens)))
        self._last_bucket[_canon(provider)] = (_canon(provider), key_index)

    # ── Async path ────────────────────────────────────────────────────

    def reserve_async(self, provider: str, estimated_tokens: int = 0,
                      key_index: int = 0) -> "_AsyncReservation":
        return _AsyncReservation(self, provider, estimated_tokens, key_index)

    async def _await_slot_async(
        self, provider: str, estimated_tokens: int, key_index: int,
    ) -> None:
        # Loop: hold lock, compute wait, release lock, sleep, retry.
        # Keeps the await-on-sleep outside the lock so concurrent
        # reservers don't head-of-line block each other unnecessarily.
        lock = self._async_lock(provider, key_index)
        while True:
            async with lock:
                bucket = self._bucket(provider, key_index)
                row = self._row_for(provider)
                now = time.monotonic()
                wait = self._compute_wait(bucket, row, estimated_tokens, now)
                if wait <= 0:
                    self._commit(bucket, row, estimated_tokens, now, provider, key_index)
                    return
            await asyncio.sleep(wait)

    # ── Sync path ─────────────────────────────────────────────────────

    @contextmanager
    def reserve_sync(self, provider: str, estimated_tokens: int = 0,
                     key_index: int = 0) -> Iterator[None]:
        self._await_slot_sync(provider, estimated_tokens, key_index)
        try:
            yield
        finally:
            pass

    def _await_slot_sync(
        self, provider: str, estimated_tokens: int, key_index: int,
    ) -> None:
        while True:
            with self._sync_lock:
                bucket = self._bucket(provider, key_index)
                row = self._row_for(provider)
                now = time.monotonic()
                wait = self._compute_wait(bucket, row, estimated_tokens, now)
                if wait <= 0:
                    self._commit(bucket, row, estimated_tokens, now, provider, key_index)
                    return
            time.sleep(wait)

    # ── Post-call token correction ────────────────────────────────────

    def report_actual_tokens(self, provider: str, actual_tokens: int) -> None:
        """Replace the last reservation's estimate with the actual count.

        Cheap correctness: many providers stream usage on the final
        chunk. If they do, callers pass the real number here so the
        sliding window reflects truth rather than the estimate.
        """
        if actual_tokens <= 0:
            return
        canon = _canon(provider)
        with self._sync_lock:
            bucket_key = self._last_bucket.get(canon)
            if bucket_key is None:
                return
            bucket = self._buckets.get(bucket_key)
            if not bucket or not bucket.token_window:
                return
            ts, _ = bucket.token_window[-1]
            bucket.token_window[-1] = (ts, int(actual_tokens))


class _AsyncReservation:
    """Async context manager wrapping ProviderGate._await_slot_async."""

    def __init__(self, gate: ProviderGate, provider: str,
                 estimated_tokens: int, key_index: int) -> None:
        self._gate = gate
        self._provider = provider
        self._estimated = estimated_tokens
        self._key_index = key_index

    async def __aenter__(self) -> None:
        await self._gate._await_slot_async(
            self._provider, self._estimated, self._key_index,
        )

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


def _canon(provider: str) -> str:
    return (provider or "").strip().lower()


# Module-level singleton. Importers get the same gate so state is shared
# across the entire process.
PROVIDER_GATE = ProviderGate()


def estimate_prompt_tokens(messages: list[dict], system: str = "") -> int:
    """Cheap upper-bound estimate (chars / 4) of prompt tokens.

    Good enough for TPM bookkeeping — the over-estimate yields a slight
    conservatism that protects free tiers from edge-case overruns.
    """
    total = len(system or "")
    for m in messages or ():
        c = m.get("content")
        if isinstance(c, str):
            total += len(c)
        elif isinstance(c, list):
            for part in c:
                if isinstance(part, dict):
                    txt = part.get("text") or part.get("content") or ""
                    if isinstance(txt, str):
                        total += len(txt)
                elif isinstance(part, str):
                    total += len(part)
    return max(1, total // 4)
