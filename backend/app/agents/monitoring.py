"""TrainingMonitorAgent: subscribes to TrainingMetricUpdated, runs an
anomaly detector, and emits TrainingAnomalyDetected when needed.

It also watches for a job-completion signal piggybacked on the metric
events (payload.terminal=true) and emits TrainingCompleted.
"""
from __future__ import annotations

from app.agents.base import BaseAgent
from app.api.schemas.session import FSMState
from app.events.types import AgentEvent
from app.services import session_service


class TrainingMonitorAgent(BaseAgent):
    name = "TrainingMonitorAgent"
    role = "Watch live metrics, detect anomalies, gate training completion."
    allowed_tools = ("metrics.read", "metrics.detect_anomaly", "audit.write")
    triggers = ("TrainingMetricUpdated", "PipelineExecutionStarted")

    async def handle(self, event: AgentEvent) -> None:
        session_id = event.session_id
        session = self.get_session(session_id)
        if not session:
            return

        # Move to MONITORING the first time we see metrics for this job.
        if session.state == FSMState.EXECUTING:
            session_service.advance_state(session, FSMState.MONITORING, reason="metrics flowing")

        if event.kind == "PipelineExecutionStarted":
            return  # Just a state nudge.

        terminal = bool(event.payload.get("terminal"))
        status = event.payload.get("status")
        job_id = event.payload.get("job_id") or session.job_id
        if not job_id:
            return

        metrics_resp = await self.call_tool("metrics.read", {"job_id": job_id}, session_id)
        if "error" in metrics_resp:
            return
        metrics = metrics_resp.get("metrics") or []

        # Deterministic check for hard failures (OOM, etc)
        anomaly = await self.call_tool("metrics.detect_anomaly", {"metrics": metrics, "window": 5}, session_id)
        
        # LLM check for "soft" failures (stagnating loss, bad convergence)
        if session.llm_provider and metrics and not anomaly.get("anomaly"):
            await self.think(session_id, "Analyzing training metrics for subtle health issues...", parent=event.id)
            metrics_summary = "\n".join([f"Step {m['step']}: loss={m.get('loss')}, eval_loss={m.get('eval_loss')}" for m in metrics[-10:]])
            prompt = (
                f"Recent metrics:\n{metrics_summary}\n\n"
                "As an AI training supervisor, is this training progressing well? "
                "Look for: stagnating loss, exploding loss, or widening gap between train/eval. "
                "If it's failing, return a JSON object with 'anomaly': true, 'reason': 'detailed explanation'. "
                "Otherwise return 'anomaly': false."
            )
            try:
                import json, re
                res = await self.call_llm(session_id, prompt, system="You are an expert MLOps monitor.", parent=event.id)
                m = re.search(r"\{.*\}", res, re.DOTALL)
                if m:
                    llm_anomaly = json.loads(m.group(0))
                    if llm_anomaly.get("anomaly"):
                        anomaly = {"anomaly": True, "reason": llm_anomaly.get("reason")}
            except Exception:
                pass

        if anomaly.get("anomaly"):
            await self.emit(
                "TrainingAnomalyDetected",
                session_id,
                payload={"anomaly": anomaly["anomaly"], "reason": anomaly.get("reason"), "job_id": job_id},
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

