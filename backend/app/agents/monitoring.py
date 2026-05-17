"""TrainingMonitorAgent: watches metrics, runs an anomaly detector,
emits TrainingAnomalyDetected when needed.

The LLM-based "soft anomaly" detector is throttled: at most ONE call per
~10% progress of the run, capped at 8 calls total. This prevents per-step
LLM calls from racking up costs on multi-thousand-step runs.
"""
from __future__ import annotations

from app.agents.base import BaseAgent
from app.agents.schemas import LLMSchemaError
from app.api.schemas.session import FSMState
from app.events.types import AgentEvent
from app.services import session_service


class TrainingMonitorAgent(BaseAgent):
    name = "TrainingMonitorAgent"
    role = "Watch live metrics, detect anomalies, gate training completion."
    directive_scope = "training"
    allowed_tools = ("metrics.read", "metrics.detect_anomaly", "audit.write")
    triggers = ("TrainingMetricUpdated", "PipelineExecutionStarted")

    # Per-session throttle state: how many soft-anomaly LLM calls we've made.
    _llm_call_count: dict[str, int] = {}
    _last_progress_seen: dict[str, float] = {}

    _MAX_LLM_CALLS = 8
    _MIN_PROGRESS_DELTA = 0.10   # 10%

    async def handle(self, event: AgentEvent) -> None:
        session_id = event.session_id
        session = self.get_session(session_id)
        if not session:
            return

        if session.state == FSMState.EXECUTING:
            session_service.advance_state(session, FSMState.MONITORING, reason="metrics flowing")

        if event.kind == "PipelineExecutionStarted":
            self._llm_call_count[session_id] = 0
            self._last_progress_seen[session_id] = 0.0
            return

        terminal = bool(event.payload.get("terminal"))
        status = event.payload.get("status")
        job_id = event.payload.get("job_id") or session.job_id
        if not job_id:
            return

        metrics_resp = await self.call_tool("metrics.read", {"job_id": job_id}, session_id)
        if "error" in metrics_resp:
            return
        metrics = metrics_resp.get("metrics") or []

        # Deterministic anomaly check (cheap, runs every event).
        # Passing job_id enables the log-buffer scan for OOM / data
        # corruption / CUDA-unavailable errors that wouldn't show up in
        # the metric stream because training crashed before the next
        # on_log callback could fire.
        anomaly = await self.call_tool(
            "metrics.detect_anomaly",
            {"metrics": metrics, "window": 5, "job_id": job_id},
            session_id,
        )

        # LLM check - throttled.
        if (
            metrics
            and not anomaly.get("anomaly")
            and session.llm_provider
            and self._should_call_llm(session_id, event)
        ):
            llm_anomaly = await self._llm_anomaly_check(session_id, metrics, event.id)
            if llm_anomaly:
                anomaly = llm_anomaly

        if anomaly.get("anomaly"):
            await self.emit(
                "TrainingAnomalyDetected",
                session_id,
                payload={
                    "anomaly": anomaly["anomaly"],
                    "reason": anomaly.get("reason"),
                    "job_id": job_id,
                },
                parent_event_id=event.id,
            )

        if terminal:
            if status == "completed":
                await self.emit_message(session_id, "Training completed.", parent=event.id)
                await self.emit(
                    "TrainingCompleted",
                    session_id,
                    payload={"job_id": job_id, "metrics": metrics_resp},
                    parent_event_id=event.id,
                )
            elif status == "failed":
                await self.emit_error(session_id, f"job {job_id} failed")
                session_service.fail(session, f"job {job_id} failed")
            elif status == "stopped":
                await self.emit_message(session_id, "Training stopped.", parent=event.id)

    def _should_call_llm(self, session_id: str, event: AgentEvent) -> bool:
        calls = self._llm_call_count.get(session_id, 0)
        if calls >= self._MAX_LLM_CALLS:
            return False
        # Use job progress (FE attaches a 0..100 progress_pct on completion;
        # we approximate with current step / total when missing).
        progress = float(event.payload.get("progress_pct") or 0.0) / 100.0
        if progress <= 0.0:
            # Fall back to step counter.
            step = float(event.payload.get("step") or 0)
            progress = min(step / 1000.0, 1.0)
        last = self._last_progress_seen.get(session_id, 0.0)
        if progress - last < self._MIN_PROGRESS_DELTA:
            return False
        self._last_progress_seen[session_id] = progress
        self._llm_call_count[session_id] = calls + 1
        return True

    async def _llm_anomaly_check(self, session_id: str, metrics: list[dict], parent: str) -> dict | None:
        recent = "\n".join(
            f"step={m.get('step')} epoch={m.get('epoch')} loss={m.get('loss')} val_loss={m.get('val_loss')}"
            for m in metrics[-10:]
        )
        from app.agents.schemas import parse_llm_json
        from pydantic import BaseModel

        class _Anomaly(BaseModel):
            anomaly: bool
            reason: str = ""

        raw = await self.call_llm(
            session_id,
            (
                f"Recent training metrics:\n{recent}\n\n"
                "Is the training going off the rails? Look for: "
                "stagnating loss, exploding loss, widening train/val gap. "
                'Return JSON: {"anomaly": bool, "reason": "..."}'
            ),
            system="You are an expert ML training monitor. Output only JSON.",
            stream_thoughts=False,
            parent=parent,
        )
        if not raw:
            return None
        try:
            parsed = parse_llm_json(raw, _Anomaly)
        except LLMSchemaError:
            return None
        if not parsed.anomaly:
            return None
        return {"anomaly": True, "reason": parsed.reason}
