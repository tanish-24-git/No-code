"""Cross-session persistent memory (FINETUNE.md analog of CLAUDE.md).

What lives here:
    * Global preferences the user repeats every session — default HF
      username, license, preferred dataset format, target hardware,
      "I always want LoRA not QLoRA" style biases.
    * Session-scoped notes: per-session FINETUNE.md that survives
      restarts so a user can leave the studio overnight and resume
      tomorrow.

Composition:
    The AgenticLoop calls ``compose_for_prompt(session_id)`` once per
    turn and prepends the result to the system prompt. Global memory
    is shown first, then session-scoped, so a session note can
    override a global default if the user wrote one.

Hard limits:
    * Each file is capped at ``MAX_MEMORY_BYTES`` (default 16 KiB) —
      we refuse to load more so an over-eager LLM auto-writer can't
      grow the system prompt unbounded.
    * Globals live at ``data/FINETUNE.md``.
    * Session-scoped at ``data/sessions/<session_id>/FINETUNE.md``.

The files are plain Markdown and the user can edit them directly with
a text editor, so this module never tries to schema-validate the
contents — that would be hostile to the "this is your memory, write
whatever helps" UX.
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Optional

from app.utils.config import settings


log = logging.getLogger("finetune-studio.memory")


MAX_MEMORY_BYTES = 16 * 1024  # 16 KiB hard cap per file
_GLOBAL_FILENAME = "FINETUNE.md"
_SESSION_FILENAME = "FINETUNE.md"


_SAFE_SESSION_ID = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")


def global_path() -> Path:
    return settings.data_dir / _GLOBAL_FILENAME


def session_path(session_id: str) -> Optional[Path]:
    """Return the session-scoped memory path, or None if session_id is
    not a safe filename component (defence against path traversal —
    callers don't always sanitise)."""
    if not session_id or not _SAFE_SESSION_ID.match(session_id):
        return None
    return settings.data_dir / "sessions" / session_id / _SESSION_FILENAME


def _read_file(p: Path) -> str:
    if not p.exists():
        return ""
    try:
        # Limit read to the cap to keep a malicious 1GB file from
        # blowing the process when loaded.
        with p.open("rb") as f:
            raw = f.read(MAX_MEMORY_BYTES + 1)
        if len(raw) > MAX_MEMORY_BYTES:
            log.warning("memory file %s exceeded %d bytes; truncating",
                        p, MAX_MEMORY_BYTES)
            raw = raw[:MAX_MEMORY_BYTES]
        return raw.decode("utf-8", errors="replace").strip()
    except OSError as e:
        log.warning("failed to read memory file %s: %s", p, e)
        return ""


def _write_file(p: Path, content: str) -> bool:
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        encoded = content.encode("utf-8")
        if len(encoded) > MAX_MEMORY_BYTES:
            encoded = encoded[:MAX_MEMORY_BYTES]
        p.write_bytes(encoded)
        return True
    except OSError as e:
        log.warning("failed to write memory file %s: %s", p, e)
        return False


def read_global() -> str:
    return _read_file(global_path())


def write_global(content: str) -> bool:
    return _write_file(global_path(), content or "")


def read_session(session_id: str) -> str:
    p = session_path(session_id)
    if p is None:
        return ""
    return _read_file(p)


def write_session(session_id: str, content: str) -> bool:
    p = session_path(session_id)
    if p is None:
        log.warning("rejected unsafe session_id for memory write: %r", session_id)
        return False
    return _write_file(p, content or "")


def compose_for_prompt(session_id: str) -> str:
    """Render both files into a system-prompt-ready Markdown block.

    Returns empty string when neither file has content, so callers can
    cleanly skip the section.
    """
    global_md = read_global()
    session_md = read_session(session_id) if session_id else ""
    if not global_md and not session_md:
        return ""

    parts: list[str] = ["## Persistent memory (FINETUNE.md)"]
    if global_md:
        parts.append("### Global preferences")
        parts.append(global_md)
    if session_md:
        parts.append("### Session notes")
        parts.append(session_md)
    parts.append(
        "_The above is loaded from disk at every turn. The user may have "
        "edited it between turns; treat it as the source of truth for "
        "their preferences._"
    )
    return "\n\n".join(parts)


def stats() -> dict[str, object]:
    """Diagnostics for the settings UI: which files exist, how big."""
    g = global_path()
    return {
        "global_exists": g.exists(),
        "global_size": (g.stat().st_size if g.exists() else 0),
        "global_path": str(g),
        "max_bytes": MAX_MEMORY_BYTES,
    }
