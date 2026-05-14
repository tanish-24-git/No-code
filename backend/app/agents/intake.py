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

        ds = dataset_service.get_dataset(dataset_id)
        if dataset_service.is_raw_doc(ds):
            # For raw-doc uploads we do not run the structured-only
            # inspection tools (they would fail). Post a brief intake
            # message and emit IntakeCompleted with a raw_doc flag so the
            # AgenticLoop wakes up and runs synthesize_unified_dataset.
            await self.announce(
                session_id,
                phase="intake",
                title="Reading your upload",
                summary=(
                    f"Detected raw document(s): **{ds.name}**. I will "
                    "synthesize a trainable dataset from it shortly."
                ),
                steps=[
                    "Walk the upload(s)",
                    "Extract text from PDFs / DOCX / HTML",
                    "Synthesize unified instruction-output dataset",
                ],
                requires_approval=False,
                parent=event.id,
            )
            await self.emit_message(
                session_id,
                f"Got a raw-doc upload: **{ds.name}**. Tell me what you "
                "want to do with it, or send any message and I will pick "
                "a sensible default.",
                parent=event.id,
            )
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
                f"I've analyzed your dataset **{info.get('name')}**. "
                f"It contains **{info.get('row_count')}** rows across **{len(cols)}** columns: {col_preview}.\n\n"
                f"Based on the structure, I've identified this as a potential **{outcomes_msg}** project. "
                "However, I want to tailor this session to your specific needs.\n\n"
                "**What is your primary goal for this training run?** "
                "(e.g., 'I want a chat model for customer support', 'I need a medical Q&A assistant', 'Instruction-tune for coding')."
            ),
            parent=event.id,
        )

        await self.complete(
            session_id,
            phase="intake",
            summary=f"Identified {info.get('row_count', 0)} rows; awaiting user goal.",
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
