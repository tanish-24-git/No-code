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
    allowed_tools = ("dataset.inspect", "dataset.read_sample", "dataset.summarize_fields", "audit.write")
    triggers = ("DatasetUploaded",)

    async def handle(self, event: AgentEvent) -> None:
        session_id = event.session_id
        session = self.get_session(session_id)
        if not session:
            return

        dataset_id = event.payload.get("dataset_id") or session.dataset_id
        if not dataset_id:
            await self.emit_error(session_id, "Missing dataset_id")
            return

        # If the user uploaded a raw doc, the DataRestructurerAgent will
        # produce a structured dataset and re-emit DatasetUploaded. Skip
        # this run; we'll re-fire on the structured one.
        ds = dataset_service.get_dataset(dataset_id)
        if dataset_service.is_raw_doc(ds):
            return

        if session.state == FSMState.INIT:
            session_service.advance_state(session, FSMState.PROFILING, reason="intake started")

        await self.announce(
            session_id,
            phase="intake",
            title="Reading your dataset",
            summary="Inspecting metadata, sampling rows, and bucketing columns by purpose.",
            steps=[
                "Read row count and column types",
                "Sample 3 rows for sanity-check",
                "Classify columns: instruction-like / input-like / output-like / label-like",
            ],
            outputs=["dataset_facts artifact"],
            requires_approval=False,
            parent=event.id,
        )

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

        cols = info.get("column_names", [])
        col_preview = ", ".join(f"`{c}`" for c in cols[:8]) + ("..." if len(cols) > 8 else "")
        outcomes_msg = self._deterministic_outcomes(buckets.get("field_buckets", {}), info)

        # If the LLM is configured, replace the outlook with an LLM-grade
        # forecast. The shared-context block already includes user
        # directives, so "convert to chat" said earlier flows through here.
        llm_msg = await self.call_llm(
            session_id,
            (
                f"Dataset sample: {sample.get('sample', [])}\n"
                f"Columns: {cols}\n\n"
                "In one tight sentence (no preamble), what fine-tuning "
                "outcomes are most natural for this data?"
            ),
            system="You are an expert data strategist. Be concrete and brief.",
            stream_thoughts=False,
            parent=event.id,
        )
        if llm_msg.strip():
            outcomes_msg = llm_msg.strip()

        await self.emit_message(
            session_id,
            (
                f"Got your dataset **{info.get('name')}** - {info.get('row_count')} rows, "
                f"{len(cols)} columns ({col_preview}).\n\n"
                f"**Engineering outlook:** {outcomes_msg}"
            ),
            parent=event.id,
        )

        await self.complete(
            session_id,
            phase="intake",
            summary=f"{info.get('row_count', 0)} rows, {len(cols)} columns",
            artifacts={"dataset_facts_keys": list((buckets.get("field_buckets") or {}).keys())},
            parent=event.id,
        )

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
