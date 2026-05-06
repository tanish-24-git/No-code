"""Job runtime. Runs in a FastAPI BackgroundTask (single-process, thread).

Pipeline = node graph. We topo-sort the nodes and dispatch each one to a
node-type handler. Logs are appended to an in-memory deque per job and to
data/jobs/<id>.log on disk so SSE consumers (or restarts) can replay them.
"""
from __future__ import annotations

import asyncio
import threading
import time
import traceback
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from app.api.schemas.job import JobMetric, JobRecord, JobStatus
from app.api.schemas.pipeline import NodeGraph, PipelineConfig, PipelineRecord
from app.events.bus import get_bus
from app.events.types import AgentEvent
from app.services import pipeline_service
from app.storage import store
from app.utils.config import settings


# In-memory state. Lost on restart; the on-disk log file persists.
_log_buffers: dict[str, deque[str]] = {}
_log_locks: dict[str, threading.Lock] = {}
_log_subscribers: dict[str, list[asyncio.Queue]] = {}
_terminal: dict[str, bool] = {}
_stop_flags: dict[str, threading.Event] = {}

# job_id -> session_id, populated by ExecutionAgent. Lets the worker thread
# publish TrainingMetricUpdated events to the right session bus.
_job_to_session: dict[str, str] = {}


def bind_job_to_session(job_id: str, session_id: str) -> None:
    _job_to_session[job_id] = session_id


def session_for_job(job_id: str) -> str | None:
    return _job_to_session.get(job_id)


def _publish_metric_event(job: JobRecord, terminal: bool = False) -> None:
    sid = _job_to_session.get(job.id)
    if not sid:
        return
    payload = {
        "job_id": job.id,
        "step": job.current_step,
        "epoch": job.current_epoch,
        "loss": job.current_loss,
        "status": job.status,
        "terminal": terminal,
    }
    get_bus().publish_threadsafe(AgentEvent(
        session_id=sid, kind="TrainingMetricUpdated",
        actor="job_service", payload=payload,
    ))


def _job_log_path(job_id: str) -> Path:
    return settings.data_dir / "jobs" / f"{job_id}.log"


def _persist(job: JobRecord) -> None:
    store.write("jobs", job.id, job.model_dump(mode="json"))


def _broadcast(job_id: str, line: str) -> None:
    buf = _log_buffers.setdefault(job_id, deque(maxlen=10000))
    lock = _log_locks.setdefault(job_id, threading.Lock())
    with lock:
        buf.append(line)
    # File append (best-effort).
    try:
        with _job_log_path(job_id).open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass
    # Non-blocking fan-out to any SSE subscribers.
    for q in list(_log_subscribers.get(job_id, [])):
        try:
            q.put_nowait(line)
        except asyncio.QueueFull:
            pass


def get_buffered_logs(job_id: str) -> list[str]:
    """Return the buffered log lines (in-memory). Used when an SSE subscriber
    connects mid-run so it gets backlog first."""
    if job_id in _log_buffers:
        return list(_log_buffers[job_id])
    p = _job_log_path(job_id)
    if p.exists():
        return p.read_text(encoding="utf-8", errors="replace").splitlines()
    return []


def is_terminal(job_id: str) -> bool:
    return _terminal.get(job_id, False)


def subscribe(job_id: str) -> asyncio.Queue:
    q: asyncio.Queue[str] = asyncio.Queue(maxsize=1000)
    _log_subscribers.setdefault(job_id, []).append(q)
    return q


def unsubscribe(job_id: str, q: asyncio.Queue) -> None:
    subs = _log_subscribers.get(job_id, [])
    if q in subs:
        subs.remove(q)


def request_stop(job_id: str) -> bool:
    flag = _stop_flags.get(job_id)
    if not flag:
        return False
    flag.set()
    return True


# ─── Public API ─────────────────────────────────────────────────────────────

def create(pipeline_id: str) -> JobRecord:
    pl = pipeline_service.get(pipeline_id)
    if not pl:
        raise ValueError("Pipeline not found")
    job_id = store.new_id()
    now = datetime.now(timezone.utc)
    job = JobRecord(
        id=job_id,
        pipeline_id=pipeline_id,
        status="queued",
        total_epochs=pl.config.epochs,
        created_at=now,
    )
    _persist(job)
    return job


def get(job_id: str) -> JobRecord | None:
    raw = store.read("jobs", job_id)
    if not raw:
        return None
    return JobRecord(**raw)


def list_all() -> list[JobRecord]:
    out: list[JobRecord] = []
    for raw in store.list_all("jobs"):
        try:
            out.append(JobRecord(**raw))
        except Exception:
            continue
    out.sort(key=lambda j: j.created_at, reverse=True)
    return out


def delete(job_id: str) -> bool:
    job = get(job_id)
    if not job:
        return False
    if job.status == "running":
        raise ValueError("Cannot delete a running job; stop it first.")
    store.delete("jobs", job_id)
    p = _job_log_path(job_id)
    if p.exists():
        try:
            p.unlink()
        except OSError:
            pass
    return True


def run_in_background(job_id: str) -> None:
    """Entrypoint for FastAPI BackgroundTasks. Spawns a worker thread so the
    BackgroundTask returns quickly even for long-running jobs."""
    t = threading.Thread(target=_execute, args=(job_id,), daemon=True)
    t.start()


# ─── Internal execution ─────────────────────────────────────────────────────

def _topo_sort(graph: NodeGraph) -> list[dict[str, Any]]:
    """Kahn's algorithm. Returns nodes in execution order."""
    nodes = {n["id"]: n for n in graph.nodes}
    indegree = {nid: 0 for nid in nodes}
    adj: dict[str, list[str]] = {nid: [] for nid in nodes}
    for e in graph.edges:
        s, t = e.get("source"), e.get("target")
        if s in nodes and t in nodes:
            adj[s].append(t)
            indegree[t] += 1
    queue = [nid for nid, d in indegree.items() if d == 0]
    out: list[dict[str, Any]] = []
    while queue:
        nid = queue.pop(0)
        out.append(nodes[nid])
        for nxt in adj[nid]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)
    if len(out) != len(nodes):
        raise ValueError("Pipeline graph has a cycle")
    return out


def _set_status(job: JobRecord, status: JobStatus, **patch: Any) -> JobRecord:
    job.status = status
    if status == "running" and not job.started_at:
        job.started_at = datetime.now(timezone.utc)
    if status in ("completed", "failed", "stopped"):
        job.completed_at = datetime.now(timezone.utc)
    for k, v in patch.items():
        setattr(job, k, v)
    _persist(job)
    if status in ("completed", "failed", "stopped"):
        _publish_metric_event(job, terminal=True)
    return job


def _execute(job_id: str) -> None:
    job = get(job_id)
    if not job:
        return
    pl = pipeline_service.get(job.pipeline_id)
    if not pl:
        _broadcast(job_id, "[ERROR] Pipeline missing")
        _set_status(job, "failed", error_message="Pipeline missing")
        _terminal[job_id] = True
        return

    stop_flag = threading.Event()
    _stop_flags[job_id] = stop_flag
    _terminal[job_id] = False
    _set_status(job, "running")
    _broadcast(job_id, f"[INFO] Job {job_id} starting for pipeline '{pl.name}'")

    try:
        ordered = _topo_sort(pl.node_graph)
        _broadcast(job_id, f"[INFO] Plan: {' -> '.join(n.get('type', n['id']) for n in ordered)}")
        ctx: dict[str, Any] = {"pipeline": pl, "config": pl.config}
        total = max(len(ordered), 1)
        for idx, node in enumerate(ordered):
            if stop_flag.is_set():
                _broadcast(job_id, "[WARN] Stop requested")
                _set_status(job, "stopped")
                _terminal[job_id] = True
                return
            ntype = node.get("type") or node["id"]
            _broadcast(job_id, f"[STEP] ({idx + 1}/{total}) running '{ntype}'")
            handler = _HANDLERS.get(ntype, _handler_passthrough)
            try:
                handler(job, pl.config, node, ctx, lambda line, jid=job_id: _broadcast(jid, line), stop_flag)
            except Exception as e:
                _broadcast(job_id, f"[ERROR] {ntype} failed: {e}")
                _broadcast(job_id, traceback.format_exc())
                _set_status(job, "failed", error_message=str(e))
                _terminal[job_id] = True
                return
            job.progress_pct = round(((idx + 1) / total) * 100.0, 1)
            _persist(job)

        out_path = ctx.get("model_output_path")
        _set_status(job, "completed", model_output_path=out_path, progress_pct=100.0)
        _broadcast(job_id, "[DONE] Pipeline completed")
    except Exception as e:
        _broadcast(job_id, f"[FATAL] {e}")
        _broadcast(job_id, traceback.format_exc())
        _set_status(job, "failed", error_message=str(e))
    finally:
        _terminal[job_id] = True


# ─── Node handlers ─────────────────────────────────────────────────────────
#
# Each handler signature:
#   (job, config, node, ctx, log, stop_flag) -> None
# Mutates `ctx` to pass artifacts forward.

LogFn = Callable[[str], None]


def _handler_passthrough(job: JobRecord, config: PipelineConfig, node: dict[str, Any], ctx: dict[str, Any], log: LogFn, stop: threading.Event) -> None:
    log(f"[INFO] node '{node.get('type') or node['id']}' has no handler — skipping")


def _handler_dataset(job: JobRecord, config: PipelineConfig, node: dict[str, Any], ctx: dict[str, Any], log: LogFn, stop: threading.Event) -> None:
    from app.services import dataset_service
    dataset_id = node.get("data", {}).get("dataset_id") or config.dataset_id
    if not dataset_id:
        raise ValueError("No dataset_id on dataset node or pipeline config")
    ds = dataset_service.get_dataset(dataset_id)
    if not ds:
        raise ValueError(f"Dataset {dataset_id} not found")
    log(f"[INFO] dataset '{ds.name}' rows={ds.row_count} cols={len(ds.column_names)}")
    ctx["dataset"] = ds


def _handler_preprocess(job: JobRecord, config: PipelineConfig, node: dict[str, Any], ctx: dict[str, Any], log: LogFn, stop: threading.Event) -> None:
    log("[INFO] preprocess: clean, dedupe, format prompts (placeholder)")
    # Real preprocessing imports are heavy; we leave the existing code under
    # app/preprocessing/ to be wired as the pipeline matures.
    time.sleep(0.2)


def _handler_train(job: JobRecord, config: PipelineConfig, node: dict[str, Any], ctx: dict[str, Any], log: LogFn, stop: threading.Event) -> None:
    """Lightweight, agent-friendly placeholder: simulates an epoch loop and
    streams metrics. Swapping in a real trainer (LoRATrainer) is a one-liner.
    """
    epochs = max(config.epochs, 1)
    job.total_epochs = epochs
    for epoch in range(1, epochs + 1):
        if stop.is_set():
            return
        # Pretend training: 5 steps per epoch.
        for step in range(1, 6):
            if stop.is_set():
                return
            time.sleep(0.2)
            loss = max(0.05, 2.5 / (epoch * step))
            job.current_epoch = epoch
            job.current_step += 1
            job.current_loss = round(loss, 4)
            job.metrics.append(
                JobMetric(
                    step=job.current_step,
                    epoch=epoch,
                    loss=job.current_loss,
                    learning_rate=config.learning_rate,
                    timestamp=datetime.now(timezone.utc),
                )
            )
            _persist(job)
            log(f"[METRIC] epoch={epoch} step={job.current_step} loss={job.current_loss}")
            _publish_metric_event(job)
    out = settings.models_dir / job.id
    out.mkdir(parents=True, exist_ok=True)
    (out / "README.md").write_text(f"# Stub artifact for job {job.id}\n", encoding="utf-8")
    ctx["model_output_path"] = str(out)
    log(f"[INFO] saved stub artifact to {out}")


def _handler_evaluate(job: JobRecord, config: PipelineConfig, node: dict[str, Any], ctx: dict[str, Any], log: LogFn, stop: threading.Event) -> None:
    log("[INFO] evaluate: computing metrics (placeholder)")
    time.sleep(0.2)


def _handler_export(job: JobRecord, config: PipelineConfig, node: dict[str, Any], ctx: dict[str, Any], log: LogFn, stop: threading.Event) -> None:
    out = ctx.get("model_output_path")
    if not out:
        log("[WARN] no model_output_path in context; nothing to export")
        return
    log(f"[INFO] export: artifact at {out}")


_HANDLERS: dict[str, Callable[..., None]] = {
    "dataset": _handler_dataset,
    "preprocess": _handler_preprocess,
    "train": _handler_train,
    "evaluate": _handler_evaluate,
    "export": _handler_export,
}
