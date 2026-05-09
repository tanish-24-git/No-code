"""RecoveryAgent: on TrainingAnomalyDetected, propose a plan diff. Whether
the diff is auto-applied or requires user confirmation depends on policy
and the number of recoveries already performed."""
from __future__ import annotations

from datetime import datetime, timezone

from app.agents.base import BaseAgent
from app.api.schemas.session import FSMState, RecoveryRecord
from app.events.types import AgentEvent
from app.orchestration.policy import decide_recovery_approval
from app.services import pipeline_service, session_service


class RecoveryAgent(BaseAgent):
    name = "RecoveryAgent"
    role = "Propose a plan diff in response to a training anomaly."
    allowed_tools = ("recovery.propose_plan", "pipeline.apply_config", "audit.write")
    triggers = ("TrainingAnomalyDetected", "RetryApproved")

    async def handle(self, event: AgentEvent) -> None:
        session_id = event.session_id
        session = self.get_session(session_id)
        if not session:
            return

        if event.kind == "TrainingAnomalyDetected":
            await self._propose(session, event)
            return
        if event.kind == "RetryApproved":
            await self._apply(session, event)
            return

    async def _propose(self, session, event: AgentEvent) -> None:
        anomaly = event.payload.get("anomaly")
        pipeline = pipeline_service.get(session.pipeline_id) if session.pipeline_id else None
        cfg = pipeline.config.model_dump() if pipeline else {}

        await self.think_delta(session.id, f"Training anomaly detected: {anomaly}. Analyzing error trace and generating recovery strategy...", is_final=False, parent=event.id)
        
        prompt = f"Anomaly: {anomaly}. Config: {cfg}. Propose a JSON recovery plan. Return ONLY JSON with 'diff_id' (string), 'operations' (list of dicts with 'op', 'path', 'new'), 'rationale' (string), 'confidence' (float), 'estimated_extra_minutes' (float)."
        system_prompt = "You are a training recovery agent. Fix OOMs by halving batch size, etc. Output valid JSON."
        
        result_str = await self.call_llm(session.id, prompt, system=system_prompt, stream_thoughts=True, parent=event.id)
        
        import json
        import uuid
        try:
            if "```json" in result_str:
                result_str = result_str.split("```json")[1].split("```")[0]
            elif "```" in result_str:
                result_str = result_str.split("```")[1].split("```")[0]
            plan = json.loads(result_str.strip())
            if "diff_id" not in plan:
                plan["diff_id"] = str(uuid.uuid4())
        except Exception:
            # Fallback if LLM fails
            plan = await self.call_tool(
                "recovery.propose_plan",
                {"anomaly": anomaly, "config": cfg, "extra_minutes": 5.0},
                session.id,
            )
            
        if "error" in plan or not plan.get("operations"):
            await self.think_delta(session.id, f"\nRecovery: no automatic fix for `{anomaly}`.", is_final=True, parent=event.id)
            return
            
        await self.think_delta(session.id, f"\nRecovery strategy formulated with confidence {plan.get('confidence', 0.0)}.", is_final=True, parent=event.id)

        record = RecoveryRecord(
            diff_id=plan["diff_id"],
            operations=plan["operations"],
            rationale=plan["rationale"],
            confidence=float(plan["confidence"]),
            estimated_extra_minutes=float(plan["estimated_extra_minutes"]),
            approved=None,
            created_at=datetime.now(timezone.utc),
        )
        session.recovery_history.append(record)
        session_service.attach_artifact(session, "recovery_plan", record.model_dump(mode="json"))

        decision = decide_recovery_approval(record.confidence, prior_recoveries=len(session.recovery_history) - 1)
        if decision == "block":
            await self.emit_message(session.id, "Reached recovery cap; not retrying automatically.", parent=event.id)
            return

        await self.emit(
            "RecoveryPlanGenerated",
            session.id,
            payload=record.model_dump(mode="json"),
            confidence=record.confidence,
            parent_event_id=event.id,
        )
        if decision == "approve":
            await self.emit(
                "RetryApproved",
                session.id,
                payload={"diff_id": record.diff_id, "auto": True},
                parent_event_id=event.id,
            )
        else:
            await self.emit(
                "RetryRequested",
                session.id,
                payload={"diff_id": record.diff_id, "summary": record.rationale},
                parent_event_id=event.id,
            )

    async def _apply(self, session, event: AgentEvent) -> None:
        diff_id = event.payload.get("diff_id")
        rec = next((r for r in session.recovery_history if r.diff_id == diff_id), None)
        if not rec:
            await self.emit_error(session.id, f"recovery diff {diff_id} not found")
            return

        # Translate ops into a config patch.
        patch: dict = {}
        for op in rec.operations:
            if op.get("op") != "set":
                continue
            path = op.get("path", "")
            new = op.get("new")
            if path.startswith("config."):
                patch[path[len("config."):]] = new

        if patch and session.pipeline_id:
            await self.call_tool(
                "pipeline.apply_config",
                {"pipeline_id": session.pipeline_id, "config": patch, "reasoning": {"recovery": rec.rationale}},
                session.id,
            )
        # Mark the diff approved on the session.
        for r in session.recovery_history:
            if r.diff_id == diff_id:
                r.approved = True
        session_service._persist(session)  # pylint: disable=protected-access

        # Restart execution.
        if session.state in (FSMState.MONITORING, FSMState.EXECUTING):
            session_service.advance_state(session, FSMState.RECOVERING, reason="applying recovery diff")
        await self.emit(
            "PipelineApproved",
            session.id,
            payload={"pipeline_id": session.pipeline_id, "auto": True, "from_recovery": diff_id},
            parent_event_id=event.id,
            rationale="recovery diff approved",
        )
