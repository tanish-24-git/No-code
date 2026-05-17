"""DataRestructurerAgent — turns raw documents into a structured dataset.

v4.0 — Universal Intake Engine: handles ANY uploaded content, regardless of
extension. Triggered as the very first stage in the pipeline (right after
intake) when the uploaded dataset is classified as ``raw_doc``. Now supports:

    * Cross-file synthesis — aggregated directory uploads where multiple
      documents (PDF + JSON logs + TXT notes) are combined.
    * Autonomous de-noising — boilerplate, headers, and irrelevant content
      are stripped before LLM restructuring.
    * Extension-agnostic parsing — type sniffed from content, not filename.

The agent:
    1. Asks the user (with a sensible default) which target format to
       produce: instruction, chat, qa, or classification.
    2. Chunks the source text via ``alchemy.chunk_text``.
    3. For each chunk, asks the configured LLM to emit a batch of pairs.
    4. Validates each batch against ``RestructureBatch`` (pydantic).
    5. Writes a JSONL dataset via ``dataset_service.register_synthetic``.
    6. Re-binds the session to the new structured dataset and emits
       ``DatasetUploaded`` so the rest of the pipeline runs unchanged.

In deterministic mode (no LLM provider), it falls back to producing a
language-modelling pretrain dataset: each chunk becomes one row keyed
"text". This is genuinely useful for continued pretraining and is honest
about what it produced.
"""
from __future__ import annotations

import json
from typing import Any

from app.agents.base import BaseAgent
from app.agents.schemas import RestructureBatch
from app.api.schemas.session import FSMState
from app.events.types import AgentEvent
from app.services import dataset_service, session_service
from app.services import directives as directives_service


_FORMAT_OPTIONS = ("instruction", "chat", "qa", "classification", "language_modeling")


class DataRestructurerAgent(BaseAgent):
    name = "DataRestructurerAgent"
    role = "Convert any uploaded raw content into a fine-tuning dataset (extension-agnostic, cross-file capable)."
    directive_scope = "data"
    allowed_tools = (
        "alchemy.chunk_text",
        "alchemy.build_restructure_prompt",
    )
    triggers = ("DatasetUploaded",)
    # Restructuring is the high-volume LLM workload — many chunks per
    # document. Cheap tier saves the most quota here.
    tier_preference = "fast"

    async def handle(self, event: AgentEvent) -> None:
        session_id = event.session_id
        session = self.get_session(session_id)
        if not session:
            return

        ds = dataset_service.get_dataset(session.dataset_id)
        if not ds:
            return  # Dataset hasn't been registered yet; ignore.

        if not dataset_service.is_raw_doc(ds):
            return  # Already structured; nothing to do.

        # Ask the user which target format we should produce.
        # Determine if this is a multi-file source (directory upload)
        analysis = ds.analysis or {}
        source_type = analysis.get("source", "file")
        file_count = analysis.get("file_count", 1)
        source_desc = (
            f"**{file_count} files** from directory **{ds.name}**"
            if source_type == "directory"
            else f"**{ds.name}**"
        )

        await self.announce(
            session_id,
            phase="restructure",
            title="Restructure raw content into a dataset",
            summary=(
                f"Your upload {source_desc} contains raw content. "
                "I'll chunk the text, de-noise it, and synthesize fine-tuning pairs. "
                "Tell me what format you want or comment with details "
                "(e.g. 'medical Q&A', 'code completion'). I default to "
                "**instruction** if you just approve."
            ),
            steps=[
                f"Choose target format ({' / '.join(_FORMAT_OPTIONS)})",
                "Chunk the document into overlapping passages",
                "Generate pairs per chunk (LLM-driven when configured)",
                "Validate each batch against the RestructureBatch schema",
                "Register the synthesized JSONL as a new dataset",
            ],
            outputs=["new structured dataset", "DatasetUploaded re-emit"],
            requires_approval=True,
            parent=event.id,
        )

        comment = await self.wait_for_approval(session_id, "restructure")

        # Resolve target format from the comment, the standing directives,
        # or the default.
        target_format = self._resolve_format(session_id, comment) or "instruction"
        user_intent = comment or self._derive_intent_from_directives(session_id)

        await self.think(
            session_id,
            f"Restructuring as **{target_format}**. Chunking the source text...",
            parent=event.id,
        )

        # 1. Pull the full text and chunk it.
        text = dataset_service.read_doc_text(ds)
        if not text or not text.strip():
            await self.emit_error(session_id, "could not extract text from the uploaded document")
            return

        chunked = await self.call_tool(
            "alchemy.chunk_text",
            {"text": text, "chunk_chars": 4000, "overlap_chars": 400},
            session_id,
        )
        chunks = chunked.get("chunks") or []
        if not chunks:
            await self.emit_error(session_id, "document produced no usable chunks")
            return

        await self.emit_message(
            session_id,
            f"Chunked into {len(chunks)} passages. Synthesizing pairs...",
            parent=event.id,
        )

        # 2. Per-chunk LLM call -> validated pairs.
        all_rows: list[dict[str, Any]] = []
        any_llm_used = False
        for i, ch in enumerate(chunks):
            tmpl = await self.call_tool(
                "alchemy.build_restructure_prompt",
                {
                    "text": ch["text"],
                    "format": target_format,
                    "user_intent": user_intent or "",
                    "max_pairs": 12,
                },
                session_id,
            )
            batch = await self.call_llm_typed(
                session_id,
                tmpl["prompt"],
                schema=RestructureBatch,
                system=tmpl["system"],
                stream_thoughts=False,
                parent=event.id,
            )
            if batch is None:
                # Either no provider, or the LLM keeps returning bad JSON.
                # Use the deterministic fallback: this chunk becomes one
                # language-modeling row.
                all_rows.append({"text": ch["text"]})
                continue

            any_llm_used = True
            for pair in batch.pairs:
                row = self._pair_to_row(pair, target_format)
                if row:
                    all_rows.append(row)

            # Periodic progress narration so the user sees motion.
            if i and (i % 5 == 0 or i == len(chunks) - 1):
                await self.think(
                    session_id,
                    f"Processed {i + 1}/{len(chunks)} chunks; "
                    f"{len(all_rows)} pairs so far.",
                    parent=event.id,
                )

        if not all_rows:
            await self.emit_error(session_id, "restructuring produced zero rows")
            return

        # 3. Register the synthesized dataset.
        new_name = f"{ds.name}.{target_format}.jsonl"
        new_ds = dataset_service.register_synthetic(
            name=new_name,
            rows=all_rows,
            parent_dataset_id=ds.id,
            note=f"restructured-{target_format}{' (LLM)' if any_llm_used else ' (deterministic LM)'}",
        )

        # 4. Re-bind the session to the new dataset and re-emit
        # DatasetUploaded so intake / profiling / etc. run normally.
        session.dataset_id = new_ds.id
        session_service.attach_artifact(session, "restructure", {
            "from_dataset_id": ds.id,
            "to_dataset_id": new_ds.id,
            "format": target_format,
            "rows": len(all_rows),
            "llm_used": any_llm_used,
        })
        # Persist the bound dataset through the official setter so locks
        # are respected.
        session_service.set_dataset(session, new_ds.id)

        # Record what we did as a directive so model selection knows the
        # pipeline is now training in `target_format` mode.
        directives_service.record(
            session_id,
            f"The training dataset is in {target_format} format with "
            f"{len(all_rows)} pairs (restructured from {ds.name}).",
            source_phase="restructure",
            scope="global",
            actor=self.name,
        )

        await self.emit_message(
            session_id,
            f"Restructured **{ds.name}** into **{len(all_rows)}** "
            f"{target_format} pairs. New dataset: {new_ds.name}",
            parent=event.id,
        )

        await self.complete(
            session_id,
            phase="restructure",
            summary=f"{len(all_rows)} {target_format} pairs",
            artifacts={"new_dataset_id": new_ds.id, "rows": len(all_rows)},
            parent=event.id,
        )

        # Re-emit DatasetUploaded so the rest of the pipeline runs against
        # the structured dataset.
        await self.emit(
            "DatasetUploaded",
            session_id,
            payload={"dataset_id": new_ds.id, "name": new_ds.name},
            parent_event_id=event.id,
        )

    # ── helpers ──────────────────────────────────────────────────────────

    def _resolve_format(self, session_id: str, comment: str | None) -> str | None:
        """Pull the target format from the user's comment (highest priority)
        or from any earlier global directives."""
        haystack = (comment or "").lower()
        if not haystack:
            haystack = " ".join(
                d.text.lower()
                for d in directives_service.read_for_scope(session_id, "data")
            )
        for fmt in _FORMAT_OPTIONS:
            if fmt in haystack:
                return fmt
        # Common synonyms.
        if "conversation" in haystack or "dialogue" in haystack:
            return "chat"
        if "instruct" in haystack:
            return "instruction"
        if "question" in haystack and "answer" in haystack:
            return "qa"
        if "classify" in haystack or "label" in haystack:
            return "classification"
        if "pretrain" in haystack or "continued" in haystack:
            return "language_modeling"
        return None

    def _derive_intent_from_directives(self, session_id: str) -> str:
        directives = directives_service.read_for_scope(session_id, "data")
        return " | ".join(d.text for d in directives[-5:])

    def _pair_to_row(self, pair, fmt: str) -> dict[str, Any] | None:
        """Map a validated RestructurePair to a JSONL training row.

        The shape is chosen to be loadable by ``trl.SFTTrainer`` directly:

            instruction format -> {"instruction": ..., "input": ..., "output": ...}
            chat format        -> {"messages": [{role, content}, ...]}
            qa format          -> {"instruction": question, "output": answer}
            classification     -> {"text": input, "label": label}
        """
        if fmt in ("chat",):
            if pair.messages:
                return {"messages": pair.messages}
            # If LLM returned instruction/output for a chat ask, lift it.
            if pair.instruction and pair.output:
                return {"messages": [
                    {"role": "user", "content": pair.instruction},
                    {"role": "assistant", "content": pair.output},
                ]}
            return None
        if fmt == "qa":
            if pair.question and pair.answer:
                return {"instruction": pair.question, "output": pair.answer}
            if pair.instruction and pair.output:
                return {"instruction": pair.instruction, "output": pair.output}
            return None
        if fmt == "classification":
            inp = pair.input or pair.instruction
            if inp and pair.label:
                return {"text": inp, "label": pair.label}
            return None
        if fmt == "language_modeling":
            text = pair.output or pair.input or pair.instruction
            if text:
                return {"text": text}
            return None
        # default: instruction
        if pair.instruction and pair.output:
            return {
                "instruction": pair.instruction,
                "input": pair.input or "",
                "output": pair.output,
            }
        if pair.question and pair.answer:
            return {"instruction": pair.question, "input": "", "output": pair.answer}
        return None
