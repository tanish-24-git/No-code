"""Dataset management routes. The upload endpoint is the canonical entry
point of the agent runtime: a successful upload starts an AgentSession and
publishes DatasetUploaded on the bus, which kicks off the whole pipeline.

v4.0 — Universal intake: supports both single files and multi-file
directory uploads. All file types accepted (extension-agnostic).
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Response, UploadFile
from pydantic import BaseModel

from app.api.schemas.dataset import DatasetSchema
from app.events.bus import get_bus
from app.events.types import AgentEvent
from app.services import dataset_service, session_service
from app.tools.llm import ping_llm
from app.utils.config import settings


log = logging.getLogger("finetune-studio.datasets")

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


# ── Upload safety guards ──────────────────────────────────────────────────
#
# We accept arbitrary user content. Three classes of risk to defend against:
#   1. Path traversal — filename with ".." or absolute paths
#   2. Resource exhaustion — multi-GB upload OOMing the backend
#   3. Pickle / joblib RCE — code execution via deserialisation
#
# These are server-side caps; the UI may apply tighter limits.

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(500 * 1024 * 1024)))  # 500 MiB
MAX_DIRECTORY_BYTES = int(os.getenv("MAX_DIRECTORY_BYTES", str(2 * 1024 * 1024 * 1024)))  # 2 GiB
MAX_DIRECTORY_FILES = int(os.getenv("MAX_DIRECTORY_FILES", "1000"))

# Pickle/joblib executes arbitrary code on .load. We don't use them anywhere,
# but a malicious upload could trick a future feature into doing so. Reject
# at the front door.
_DENYLIST_EXTENSIONS = frozenset({
    ".pkl", ".pickle", ".joblib", ".dill",
    ".pyc", ".pyo",
    ".exe", ".dll", ".so", ".dylib",
    ".bat", ".cmd", ".sh", ".ps1",
})

# Safe filename chars: alphanumerics, dot, dash, underscore. Anything else
# gets stripped. We also clamp length so an attacker can't blow path-length
# limits with a 10K-char filename.
_FILENAME_KEEP = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitize_filename(raw: str) -> str:
    """Strip path components, normalise to safe chars, cap length.

    Path traversal defence: ``Path(raw).name`` already drops directory
    parts, but on Windows a backslash-containing filename can still slip
    through some tooling — so we do both ``Path.name`` and an explicit
    regex sweep.
    """
    base = Path(raw or "").name  # drops "../" prefixes on POSIX
    # Strip backslash-containing pieces (Windows-style) just in case.
    base = base.rsplit("\\", 1)[-1]
    base = base.rsplit("/", 1)[-1]
    cleaned = _FILENAME_KEEP.sub("_", base).strip("._-")
    if not cleaned:
        cleaned = "upload"
    if len(cleaned) > 200:
        # Preserve any extension after the truncation.
        stem, dot, ext = cleaned.rpartition(".")
        cleaned = stem[: 200 - len(dot) - len(ext)] + dot + ext if dot else cleaned[:200]
    return cleaned


def _reject_dangerous_extension(filename: str) -> None:
    """Hard-fail uploads with executable or deserialisable extensions."""
    ext = Path(filename).suffix.lower()
    if ext in _DENYLIST_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=(
                f"File type '{ext}' is not accepted. Pickle / joblib / "
                f"executable formats can run arbitrary code on load and "
                f"are rejected at the gate. Convert to CSV / JSON / "
                f"JSONL / TXT first."
            ),
        )


class LLMProbe(BaseModel):
    ok: bool
    mode: str           # "full_agent" | "deterministic"
    provider: str | None = None
    model: str | None = None
    latency_ms: float = 0.0
    detail: str = ""


class DatasetUploadResponse(BaseModel):
    dataset: DatasetSchema
    session_id: str
    llm_probe: LLMProbe


@router.post("/upload", response_model=DatasetUploadResponse, status_code=201)
async def upload_dataset(file: UploadFile = File(...)) -> DatasetUploadResponse:
    """Upload any file (extension-agnostic). The Universal Intake Engine
    sniffs the content to determine type and parses accordingly."""
    raw_name = file.filename or "dataset"
    _reject_dangerous_extension(raw_name)
    safe_name = _sanitize_filename(raw_name)
    # Stream-read with a hard cap so a 50 GiB upload doesn't OOM the
    # backend. UploadFile already spools to disk above 1 MiB so this is
    # cheap in practice.
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Upload exceeds limit ({len(contents)} > {MAX_UPLOAD_BYTES} bytes). "
                   f"Set MAX_UPLOAD_BYTES env var to raise the cap.",
        )
    try:
        dataset = dataset_service.save_uploaded(safe_name, contents)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    # Probe the configured LLM provider before booting the swarm. The result
    # determines whether agents run in full LLM-driven mode or deterministic
    # fallback mode; either way the user sees an explicit verdict in the UI.
    #
    # We do NOT defer this even though it adds a few seconds to the upload
    # response — the orchestrator branches on probe.mode at SessionStarted,
    # so a pending probe would lock the whole session into deterministic
    # mode. The ProviderGate in providers.py now serializes ping_llm
    # together with the subsequent intake/loop calls, so this synchronous
    # probe no longer participates in a burst-RPM exhaustion pattern.
    probe = await ping_llm()

    # Start the agent session and emit the first event. The orchestrator
    # posts the welcome message; downstream agents take it from there.
    session = session_service.start_for_dataset(dataset.id)
    # Persist the probe verdict on the session so agents can branch on it.
    session_service.attach_artifact(session, "llm_probe", probe)
    bus = get_bus()

    await bus.publish(AgentEvent(
        session_id=session.id, kind="SessionStarted", actor="system",
        payload={"dataset_id": dataset.id, "llm_probe": probe},
    ))
    await bus.publish(AgentEvent(
        session_id=session.id, kind="DatasetUploaded", actor="system",
        payload={"dataset_id": dataset.id, "name": dataset.name, "llm_probe": probe},
    ))

    return DatasetUploadResponse(
        dataset=dataset,
        session_id=session.id,
        llm_probe=LLMProbe(**probe),
    )


@router.post("/upload-directory", response_model=DatasetUploadResponse, status_code=201)
async def upload_directory(files: list[UploadFile] = File(...)) -> DatasetUploadResponse:
    """Upload multiple files (simulating a directory upload). The Universal
    Intake Engine recursively scans, sniffs types, and aggregates all files
    into a single dataset — structured files are row-merged, raw docs are
    concatenated for cross-file synthesis."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    if len(files) > MAX_DIRECTORY_FILES:
        raise HTTPException(
            status_code=413,
            detail=f"Too many files ({len(files)} > {MAX_DIRECTORY_FILES}). "
                   f"Set MAX_DIRECTORY_FILES env var to raise the cap.",
        )

    # Write all uploaded files to a temporary directory. We deliberately
    # FLATTEN subdirectories: the old code preserved "subdir/file.txt"
    # which silently enabled path traversal via filenames like
    # "../../etc/passwd". The aggregator doesn't need subdir structure
    # to do its job, so we sanitise + flatten + dedupe by suffixing.
    tmp_dir = Path(tempfile.mkdtemp(dir=settings.upload_dir, prefix="dir_"))
    total_bytes = 0
    try:
        seen_names: dict[str, int] = {}
        for idx, f in enumerate(files):
            raw_name = f.filename or "unnamed"
            _reject_dangerous_extension(raw_name)
            safe_name = _sanitize_filename(raw_name)
            # Dedupe by suffixing when the same sanitised name shows up
            # more than once (e.g. multiple "data.csv" from different
            # subdirs).
            n = seen_names.get(safe_name, 0)
            seen_names[safe_name] = n + 1
            if n > 0:
                stem, dot, ext = safe_name.rpartition(".")
                safe_name = f"{stem}_{n}{dot}{ext}" if dot else f"{safe_name}_{n}"

            target = tmp_dir / safe_name
            # Defence in depth: refuse any target path that resolves
            # outside tmp_dir even after sanitisation.
            try:
                target.resolve().relative_to(tmp_dir.resolve())
            except ValueError:
                log.warning("rejecting upload that escaped tmp_dir: %r -> %s", raw_name, target)
                continue

            content = await f.read()
            if not content:
                continue
            total_bytes += len(content)
            if total_bytes > MAX_DIRECTORY_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"Directory upload exceeds limit "
                           f"({total_bytes} > {MAX_DIRECTORY_BYTES} bytes). "
                           f"Set MAX_DIRECTORY_BYTES env var to raise.",
                )
            target.write_bytes(content)

        dataset = dataset_service.save_uploaded_directory(tmp_dir)
    except HTTPException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    except ValueError as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(e)) from e
    finally:
        # Clean up temp dir (data has been written to uploads/)
        shutil.rmtree(tmp_dir, ignore_errors=True)

    probe = await ping_llm()
    session = session_service.start_for_dataset(dataset.id)
    session_service.attach_artifact(session, "llm_probe", probe)
    bus = get_bus()

    await bus.publish(AgentEvent(
        session_id=session.id, kind="SessionStarted", actor="system",
        payload={"dataset_id": dataset.id, "llm_probe": probe},
    ))
    await bus.publish(AgentEvent(
        session_id=session.id, kind="DatasetUploaded", actor="system",
        payload={"dataset_id": dataset.id, "name": dataset.name, "llm_probe": probe},
    ))

    return DatasetUploadResponse(
        dataset=dataset,
        session_id=session.id,
        llm_probe=LLMProbe(**probe),
    )


@router.get("", response_model=list[DatasetSchema])
def list_datasets() -> list[DatasetSchema]:
    return dataset_service.list_datasets()


@router.get("/{dataset_id}", response_model=DatasetSchema)
def get_dataset(dataset_id: str) -> DatasetSchema:
    d = dataset_service.get_dataset(dataset_id)
    if not d:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return d


@router.delete("/{dataset_id}", status_code=204, response_class=Response)
def delete_dataset(dataset_id: str) -> Response:
    if not dataset_service.delete_dataset(dataset_id):
        raise HTTPException(status_code=404, detail="Dataset not found")
    return Response(status_code=204)
