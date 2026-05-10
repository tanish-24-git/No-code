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
    triggers = ("DatasetUploaded", "PhaseApproved")
    
    async def handle(self, event: AgentEvent) -> None:
        session_id = event.session_id
        session = self.get_session(session_id)
        if not session:
            return

        # Guard: ignore PhaseApproved events that aren't for us.
        if event.kind == "PhaseApproved" and event.payload.get("phase") != "intake":
            return

        dataset_id = event.payload.get("dataset_id") or session.dataset_id
        if not dataset_id:
            await self.emit_error(session_id, "Missing dataset_id")
            return

        # If we are resuming from a restart after approval
        resuming = event.kind == "PhaseApproved" and event.payload.get("phase") == "intake"
        
        if not resuming:
            if session.state == FSMState.INIT:
                session_service.advance_state(session, FSMState.PROFILING, reason="intake started")

            # Phase narration + canvas pop.
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
                requires_approval=True,
                parent=event.id,
            )

            await self.materialize_node(
                session_id,
                {"id": "dataset", "type": "dataset", "position": {"x": 40, "y": 80},
                 "data": {"label": "dataset", "dataset_id": dataset_id}},
                parent=event.id,
            )

            # Socratic deliberation on the dataset purpose.
            await self.think(session_id, "Analyzing the dataset source and metadata for initial intent...", parent=event.id)
            
            comment = await self.wait_for_approval(session_id, "intake")
            if comment and session.llm_provider:
                await self.think(session_id, f"Processing your feedback: '{comment}'")




        # --- Start the work (either after approval or on resume) ---
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

        # Friendly chat bubble + outcome forecast.
        cols = info.get("column_names", [])
        col_preview = ", ".join(f"`{c}`" for c in cols[:8]) + ("..." if len(cols) > 8 else "")

        # Agentic deliberation on possible outcomes.
        outcomes_msg = "Possible outcomes I can train towards: instruction-following / chat fine-tune."
        if session.llm_provider:
            await self.think(session_id, "Deliberating on training objectives for this data...", parent=event.id)
            forecast_prompt = (
                f"Dataset Sample: {sample.get('sample', [])}\n"
                f"Columns: {cols}\n"
                f"User instructions: '{comment if 'comment' in locals() else 'none'}'\n\n"
                "What training outcomes are possible with this data? (e.g., chat, translation, summarization). "
                "Provide a 1-sentence engineering forecast."
            )
            try:
                outcomes_msg = await self.call_llm(session_id, forecast_prompt, system="You are an expert Data Strategist.", parent=event.id)
            except Exception:
                pass

        await self.emit_message(
            session_id,
            (
                f"Got your dataset **{info.get('name')}** - {info.get('row_count')} rows, "
                f"{len(cols)} columns ({col_preview}).\n\n"
                f"**Engineering Outlook:** {outcomes_msg}"
            ),
            parent=event.id,
        )

        await self.complete(
            session_id,
            phase="intake",
            summary=f"{info.get('row_count', 0)} rows, {len(cols)} columns, "
                    f"{len(buckets.get('field_buckets', {}).get('output_like', []))} output-like fields",
            artifacts={"dataset_facts_keys": list((buckets.get("field_buckets") or {}).keys())},
            parent=event.id,
        )

        await self.emit(
            "IntakeCompleted",
            session_id,
            payload={"dataset_id": dataset_id, "info": info, "field_buckets": buckets.get("field_buckets", {})},
            parent_event_id=event.id,
        )


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
