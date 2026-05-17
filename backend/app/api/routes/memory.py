"""HTTP routes for cross-session persistent memory (FINETUNE.md).

The UI uses these endpoints to surface the global memory file as an
editable textarea on the Settings page so users don't have to find the
file on disk. Session-scoped memory uses the same handlers with the
session_id path parameter — useful for "tag this session as 'medical
QA experiment'" notes that persist across server restarts.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services import memory as memory_service


router = APIRouter(prefix="/api/memory", tags=["memory"])


class MemoryRead(BaseModel):
    content: str
    size: int
    max_bytes: int


class MemoryWrite(BaseModel):
    content: str


@router.get("/global", response_model=MemoryRead)
def read_global_memory() -> MemoryRead:
    content = memory_service.read_global()
    return MemoryRead(
        content=content,
        size=len(content.encode("utf-8")),
        max_bytes=memory_service.MAX_MEMORY_BYTES,
    )


@router.put("/global", response_model=MemoryRead)
def write_global_memory(payload: MemoryWrite) -> MemoryRead:
    if not memory_service.write_global(payload.content):
        raise HTTPException(status_code=500, detail="failed to write memory file")
    return read_global_memory()


@router.get("/session/{session_id}", response_model=MemoryRead)
def read_session_memory(session_id: str) -> MemoryRead:
    p = memory_service.session_path(session_id)
    if p is None:
        raise HTTPException(status_code=400, detail="unsafe session_id")
    content = memory_service.read_session(session_id)
    return MemoryRead(
        content=content,
        size=len(content.encode("utf-8")),
        max_bytes=memory_service.MAX_MEMORY_BYTES,
    )


@router.put("/session/{session_id}", response_model=MemoryRead)
def write_session_memory(session_id: str, payload: MemoryWrite) -> MemoryRead:
    if memory_service.session_path(session_id) is None:
        raise HTTPException(status_code=400, detail="unsafe session_id")
    if not memory_service.write_session(session_id, payload.content):
        raise HTTPException(status_code=500, detail="failed to write memory file")
    return read_session_memory(session_id)
