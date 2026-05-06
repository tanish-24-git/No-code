"""DatasetIntakeAgent: first responder to a fresh upload. Inspects the
dataset, posts a friendly opening summary, and emits IntakeCompleted."""
from __future__ import annotations

from app.agents.base import BaseAgent
from app.api.schemas.session import FSMState
from app.events.types import AgentEvent
from app.services import session_service


class DatasetIntakeAgent(BaseAgent):
    name = "DatasetIntakeAgent"
    role = "First-pass dataset inspection"
    allowed_tools = ("dataset.inspect", "dataset.read_sample", "dataset.summarize_fields", "audit.write")
    triggers = ("DatasetUploaded",)

    async def handle(self, event: AgentEvent) -> None:
        session_id = event.session_id
        dataset_id = event.payload.get("dataset_id")
        if not dataset_id:
            await self.emit_error(session_id, "DatasetUploaded missing dataset_id")
            return

        session = self.get_session(session_id)
        if session and session.state == FSMState.INIT:
            session_service.advance_state(session, FSMState.PROFILING, reason="intake started")

        await self.emit("IntakeStarted", session_id, payload={"dataset_id": dataset_id})

        info = await self.call_tool("dataset.inspect", {"dataset_id": dataset_id}, session_id)
        if "error" in info:
            await self.emit_error(session_id, info["error"])
            return
        sample = await self.call_tool("dataset.read_sample", {"dataset_id": dataset_id, "n": 3}, session_id)
        buckets = await self.call_tool("dataset.summarize_fields", {"dataset_id": dataset_id}, session_id)

        # Stash facts on the session so downstream agents don't re-query.
        if session:
            session_service.attach_artifact(session, "dataset_facts", {
                "info": info,
                "sample": sample.get("sample", []),
                "field_buckets": buckets.get("field_buckets", {}),
            })

        # Friendly chat bubble.
        cols = info.get("column_names", [])
        col_preview = ", ".join(f"`{c}`" for c in cols[:8]) + ("…" if len(cols) > 8 else "")
        await self.emit_message(
            session_id,
            f"Got your dataset **{info.get('name')}** — {info.get('row_count')} rows, "
            f"{len(cols)} columns ({col_preview}).",
            parent=event.id,
        )

        await self.emit(
            "IntakeCompleted",
            session_id,
            payload={"dataset_id": dataset_id, "info": info, "field_buckets": buckets.get("field_buckets", {})},
            parent_event_id=event.id,
        )
