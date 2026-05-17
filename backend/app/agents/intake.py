"""DatasetIntakeAgent: first responder to a fresh upload. Inspects the
dataset, posts a friendly opening summary, and emits IntakeCompleted.

The DataRestructurerAgent runs *before* this for raw-doc uploads and
re-emits ``DatasetUploaded`` once it has produced a structured dataset.
"""
from __future__ import annotations

from app.agents.base import BaseAgent
from app.api.schemas.session import FSMState
from app.events.types import AgentEvent
from app.services import dataset_service, session_service


class DatasetIntakeAgent(BaseAgent):
    name = "DatasetIntakeAgent"
    role = "First-pass dataset inspection"
    directive_scope = "data"
    allowed_tools = (
        "dataset.inspect",
        "dataset.read_sample",
        "dataset.summarize_fields",
        "web_search",  # look up dataset-domain context for unfamiliar uploads (#8)
        "audit.write",
    )
    triggers = ("DatasetUploaded",)
    tier_preference = "fast"  # first-pass narration of fields + sample rows

    async def handle(self, event: AgentEvent) -> None:
        session_id = event.session_id
        session = self.get_session(session_id)
        if not session:
            return

        dataset_id = event.payload.get("dataset_id") or session.dataset_id
        if not dataset_id:
            await self.emit_error(session_id, "Missing dataset_id")
            return

        ds = dataset_service.get_dataset(dataset_id)
        if dataset_service.is_raw_doc(ds):
            # Raw-doc uploads skip the structured inspection tools (they
            # would fail). Just emit IntakeCompleted with a raw_doc flag
            # so the AgenticLoop wakes up and runs
            # synthesize_unified_dataset. The loop produces all narration.
            await self.emit(
                "IntakeCompleted",
                session_id,
                payload={
                    "dataset_id": dataset_id,
                    "raw_doc": True,
                    "info": {
                        "name": ds.name,
                        "row_count": ds.row_count,
                        "file_type": ds.file_type,
                    },
                    "field_buckets": {},
                },
                parent_event_id=event.id,
            )
            return

        if session.state == FSMState.INIT:
            session_service.advance_state(session, FSMState.PROFILING, reason="intake started")

        # Keep the dataset node for the canvas view; skip the chat-bubble
        # narration - the AgenticLoop owns the user-facing voice now.
        await self.materialize_node(
            session_id,
            {"id": "dataset", "type": "dataset", "position": {"x": 40, "y": 80},
             "data": {"label": "dataset", "dataset_id": dataset_id}},
            parent=event.id,
        )

        await self.emit("IntakeStarted", session_id, payload={"dataset_id": dataset_id})

        info = await self.call_tool("dataset.inspect", {"dataset_id": dataset_id}, session_id)
        if "error" in info:
            await self.emit_error(session_id, info["error"])
            return
        sample = await self.call_tool("dataset.read_sample", {"dataset_id": dataset_id, "n": 3}, session_id)
        buckets = await self.call_tool("dataset.summarize_fields", {"dataset_id": dataset_id}, session_id)

        if session:
            session_service.attach_artifact(session, "dataset_facts", {
                "info": info,
                "sample": sample.get("sample", []),
                "field_buckets": buckets.get("field_buckets", {}),
            })

        # IntakeCompleted is the only chat-visible signal we still emit
        # from here; the loop wakes on this and produces its own
        # contextual narration on the first LLM turn.
        await self.emit(
            "IntakeCompleted",
            session_id,
            payload={"dataset_id": dataset_id, "info": info, "field_buckets": buckets.get("field_buckets", {})},
            parent_event_id=event.id,
        )

    @staticmethod
    def _deterministic_outcomes(buckets: dict, info: dict) -> str:
        has_instruction = bool(buckets.get("instruction_like"))
        has_input = bool(buckets.get("input_like"))
        has_output = bool(buckets.get("output_like"))
        has_label = bool(buckets.get("label_like"))
        rows = int(info.get("row_count") or 0)

        options: list[str] = []
        if has_instruction and has_output:
            options.append("instruction-following / chat fine-tune")
        if has_input and has_output:
            options.append("input-output transformation (translation, summarization, rewrite)")
        if has_label:
            options.append("classification / intent detection")
        if has_input and not has_output and not has_label:
            options.append("language modeling on free-text continuation")
        if not options:
            options.append("general continuation tuning")

        if rows < 200:
            warning = " (small dataset - synthetic augmentation recommended)"
        elif rows > 100_000:
            warning = " (large dataset - QLoRA + 1 epoch recommended)"
        else:
            warning = ""
        return ", ".join(options) + warning


def _forecast_outcomes(buckets: dict, info: dict) -> str:
    """A short, friendly forecast of what this dataset can train for."""
    has_instruction = bool(buckets.get("instruction_like"))
    has_input = bool(buckets.get("input_like"))
    has_output = bool(buckets.get("output_like"))
    has_label = bool(buckets.get("label_like"))
    rows = int(info.get("row_count") or 0)

    options: list[str] = []
    if has_instruction and has_output:
        options.append("instruction-following / chat fine-tune")
    if has_input and has_output:
        options.append("input-output transformation (translation, summarization, rewrite)")
    if has_label:
        options.append("classification / intent detection")
    if has_input and not has_output and not has_label:
        options.append("language modeling on free-text continuation")
    if not options:
        options.append("general continuation tuning")

    if rows < 200:
        warning = " (small dataset - synthetic augmentation recommended)"
    elif rows > 100_000:
        warning = " (large dataset - QLoRA + 1 epoch recommended)"
    else:
        warning = ""
    return ", ".join(options) + warning
