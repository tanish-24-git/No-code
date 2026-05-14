"""JSON-on-disk store. One file per record under data/<collection>/<id>.json.

Provides both synchronous (legacy) and asynchronous I/O paths.
The async variants use ``aiofiles`` so the event loop never blocks on disk.

Intentionally simple: no migrations, no locking beyond what the OS gives us.
Good enough for a single-user local tool.
"""
from __future__ import annotations

import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, Iterator

from app.utils.config import settings


def new_id() -> str:
    return uuid.uuid4().hex


def _collection_dir(collection: str) -> Path:
    p = settings.data_dir / collection
    p.mkdir(parents=True, exist_ok=True)
    return p


def _path(collection: str, record_id: str) -> Path:
    return _collection_dir(collection) / f"{record_id}.json"


# ─── Synchronous I/O (legacy callers) ────────────────────────────────────────

def write(collection: str, record_id: str, data: dict[str, Any]) -> None:
    """Atomic write: tmp file + replace. Survives partial writes."""
    target = _path(collection, record_id)
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, target)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def read(collection: str, record_id: str) -> dict[str, Any] | None:
    p = _path(collection, record_id)
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def delete(collection: str, record_id: str) -> bool:
    p = _path(collection, record_id)
    if p.exists():
        p.unlink()
        return True
    return False


def list_all(collection: str) -> Iterator[dict[str, Any]]:
    for p in sorted(_collection_dir(collection).glob("*.json")):
        try:
            with p.open("r", encoding="utf-8") as f:
                yield json.load(f)
        except (json.JSONDecodeError, OSError):
            # Skip a corrupt or vanished file; don't take down the listing.
            continue


# ─── Async I/O (non-blocking event-loop safe) ────────────────────────────────

async def async_write(collection: str, record_id: str, data: dict[str, Any]) -> None:
    """Non-blocking atomic write using aiofiles. Keeps the event loop spinning."""
    try:
        import aiofiles  # type: ignore
    except ImportError:
        # Graceful degradation: fall back to sync write in a thread.
        import asyncio
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, write, collection, record_id, data)
        return

    target = _path(collection, record_id)
    tmp = target.with_suffix(".tmp")
    payload = json.dumps(data, indent=2, default=str)
    async with aiofiles.open(tmp, "w", encoding="utf-8") as f:
        await f.write(payload)
    os.replace(str(tmp), str(target))


async def async_read(collection: str, record_id: str) -> dict[str, Any] | None:
    """Non-blocking read using aiofiles."""
    p = _path(collection, record_id)
    if not p.exists():
        return None
    try:
        import aiofiles  # type: ignore
    except ImportError:
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, read, collection, record_id)

    async with aiofiles.open(p, "r", encoding="utf-8") as f:
        content = await f.read()
    return json.loads(content)


# ─── Singleton-style record (e.g. settings) ─────────────────────────────────

def read_singleton(name: str) -> dict[str, Any] | None:
    p = settings.data_dir / f"{name}.json"
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_singleton(name: str, data: dict[str, Any]) -> None:
    target = settings.data_dir / f"{name}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, target)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


async def async_write_singleton(name: str, data: dict[str, Any]) -> None:
    """Non-blocking singleton write."""
    try:
        import aiofiles  # type: ignore
    except ImportError:
        import asyncio
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, write_singleton, name, data)
        return

    target = settings.data_dir / f"{name}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    payload = json.dumps(data, indent=2, default=str)
    async with aiofiles.open(tmp, "w", encoding="utf-8") as f:
        await f.write(payload)
    os.replace(str(tmp), str(target))
