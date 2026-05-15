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
from typing import Iterator

from app.utils.config import settings


log = logging.getLogger("finetune-studio.agents.rate_limiter")


# Default limits committed alongside the example file. The runtime file
# at `data/provider_limits.json` overrides these and may be edited
# without touching the code.
_DEFAULT_LIMITS: dict[str, dict[str, float]] = {
    "gemini":  {"rpm": 5,  "tpm": 250000, "min_interval_sec": 13},
    "google":  {"rpm": 5,  "tpm": 250000, "min_interval_sec": 13},
    "vertex":  {"rpm": 5,  "tpm": 250000, "min_interval_sec": 13},
    "groq":    {"rpm": 30, "tpm": 6000,   "min_interval_sec": 2},
    "openrouter": {"rpm": 20, "tpm": 200000, "min_interval_sec": 3},
    "together": {"rpm": 30, "tpm": 500000, "min_interval_sec": 2},
    "deepseek": {"rpm": 60, "tpm": 200000, "min_interval_sec": 1},
    "mistral":  {"rpm": 30, "tpm": 200000, "min_interval_sec": 2},
    "openai":   {"rpm": 60, "tpm": 1000000, "min_interval_sec": 1},
    "anthropic": {"rpm": 50, "tpm": 400000, "min_interval_sec": 1.2},
    "default":  {"rpm": 60, "tpm": 1000000, "min_interval_sec": 3},
}


def _limits_path() -> Path:
    return settings.data_dir / "provider_limits.json"


def _example_path() -> Path:
    return settings.data_dir / "provider_limits.example.json"


def _load_limits() -> dict[str, dict[str, float]]:
    """Read the runtime limits file. Auto-seed on first run.

    Returns the in-memory limit table. The lookup is substring-based
    against the lowercased provider name (so "google-gemini",
    "vertex-ai-gemini" and bare "gemini" all map to the gemini row).
    """
    path = _limits_path()
    if not path.exists():
        try:
            settings.data_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(_DEFAULT_LIMITS, indent=2),
                            encoding="utf-8")
            _example_path().write_text(json.dumps(_DEFAULT_LIMITS, indent=2),
                                       encoding="utf-8")
            log.info("seeded provider limits at %s", path)
        except Exception as e:
            log.warning("could not seed provider limits file (%s); using in-memory defaults", e)
            return dict(_DEFAULT_LIMITS)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("provider_limits.json must be a dict")
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
