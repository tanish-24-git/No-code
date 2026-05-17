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

import asyncio
import hashlib
import json
import os
from typing import Any

from app.agents.base import BaseAgent
from app.agents.schemas import RestructureBatch
from app.api.schemas.session import FSMState
from app.events.types import AgentEvent
from app.services import dataset_service, jsonl_validator, session_service
from app.services import directives as directives_service


_FORMAT_OPTIONS = ("instruction", "chat", "qa", "classification", "language_modeling")


# Sliding-window concurrency for chunk restructuring. Per-provider RPM/TPM
# throttling is already enforced inside call_llm_typed via ProviderGate, so
# the window just caps inflight depth; throughput scales naturally with
# multi-key rotation (when configured in Settings).
_MAX_INFLIGHT_CHUNKS = max(1, int(os.environ.get("MAX_INFLIGHT_CHUNKS", "8")))


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
            f"Chunked into {len(chunks)} passages. Synthesizing pairs in "
            f"windows of {_MAX_INFLIGHT_CHUNKS} (parallel, with cross-chunk "
            f"continuity)...",
            parent=event.id,
        )

        # 2. Per-chunk LLM call -> validated pairs. Windowed parallelism
        # so a 1 MB book doesn't take 30 minutes of serial calls. The
        # running_context carries forward a few representative pairs from
        # earlier chunks so the LLM doesn't repeat itself or lose theme
        # focus on hyper-specific extractions (e.g. Krishna-Arjuna only).
        all_rows: list[dict[str, Any]] = []
        any_llm_used = False
        chunk_total = len(chunks)
        running_context = ""

        async def _process_chunk(idx: int, ch: dict[str, Any], ctx: str) -> tuple[int, "RestructureBatch | None", dict[str, Any]]:
            tmpl = await self.call_tool(
                "alchemy.build_restructure_prompt",
                {
                    "text": ch["text"],
                    "format": target_format,
                    "user_intent": user_intent or "",
                    "max_pairs": 12,
                    "running_context": ctx,
                    "chunk_index": idx,
                    "chunk_total": chunk_total,
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
            return idx, batch, ch

        for window_start in range(0, chunk_total, _MAX_INFLIGHT_CHUNKS):
            window = chunks[window_start : window_start + _MAX_INFLIGHT_CHUNKS]
            # Snapshot context at window start so all tasks in this window
            # see the same one (cheap; avoids ordering issues).
            ctx_for_window = running_context
            try:
                results = await asyncio.gather(
                    *[
                        _process_chunk(window_start + j, ch, ctx_for_window)
                        for j, ch in enumerate(window)
                    ],
                    return_exceptions=True,
                )
            except asyncio.CancelledError:
                raise

            for res in results:
                if isinstance(res, BaseException):
                    # One chunk's task crashed. Skip — we'd rather have a
                    # partial dataset than abort the whole pipeline.
                    continue
                idx, batch, ch = res
                if batch is None:
                    # No provider OR repeated bad JSON. Fall back to LM row.
                    all_rows.append({"text": ch["text"]})
                    continue
                any_llm_used = True
                for pair in batch.pairs:
                    row = self._pair_to_row(pair, target_format)
                    if row:
                        all_rows.append(row)

            # Refresh running context for the next window — 3 most recent
            # representative pairs (short slugs) so the LLM stays on-theme
            # without burning prompt budget. Computed deterministically;
            # no extra LLM call.
            running_context = self._build_running_context(all_rows, target_format, limit=3)

            processed = min(window_start + len(window), chunk_total)
            await self.think(
                session_id,
                f"Processed {processed}/{chunk_total} chunks; "
                f"{len(all_rows)} pairs so far.",
                parent=event.id,
            )

        if not all_rows:
            await self.emit_error(session_id, "restructuring produced zero rows")
            return

        # 2a. Deduplicate identical rows produced by overlapping chunk
        # boundaries (the chunker emits 400-char overlap so the LLM sees
        # the same passage twice at boundaries). Hash-only; cheap and
        # deterministic.
        all_rows = self._hash_dedup_rows(all_rows)

        # 2b. Validate every row against the target format schema. Drop
        # malformed ones, lightly repair common LLM mistakes. This makes
        # trainer crashes on bad JSONL impossible — the trainer never sees
        # a row that doesn't match its expected shape.
        clean_rows, vsummary = jsonl_validator.validate_rows(all_rows, target_format)
        if not clean_rows:
            await self.emit_error(
                session_id,
                "restructuring produced rows, but none matched the "
                f"`{target_format}` schema. Reasons: "
                + "; ".join(vsummary.get("reasons") or ["unknown"]),
            )
            return
        if vsummary.get("dropped") or vsummary.get("repaired"):
            await self.emit_message(
                session_id,
                f"Validation: kept {vsummary['kept']}, repaired "
                f"{vsummary['repaired']}, dropped {vsummary['dropped']} "
                f"malformed row(s).",
                parent=event.id,
            )
        all_rows = clean_rows

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

    def _build_running_context(self, rows: list[dict[str, Any]], fmt: str, *, limit: int) -> str:
        """Pick a few representative pairs so the next window's prompt
        knows what's already been covered. Format-aware so it surfaces
        the right field per pair shape."""
        if not rows:
            return ""
        recent = rows[-32:]
        examples: list[str] = []
        for r in recent:
            if fmt in ("instruction", "qa") and r.get("instruction"):
                examples.append(str(r["instruction"]).strip()[:120])
            elif fmt == "chat" and isinstance(r.get("messages"), list) and r["messages"]:
                first = r["messages"][0]
                if isinstance(first, dict):
                    examples.append(str(first.get("content", "")).strip()[:120])
            elif fmt == "classification" and r.get("text"):
                examples.append(str(r["text"]).strip()[:120])
            elif fmt == "language_modeling" and r.get("text"):
                examples.append(str(r["text"]).strip()[:120])
            if len(examples) >= limit:
                break
        if not examples:
            return f"Have extracted {len(rows)} pair(s) so far."
        bullets = "\n".join(f"- {e}" for e in examples)
        return f"Already extracted {len(rows)} pair(s). Recent examples:\n{bullets}"

    def _hash_dedup_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Drop rows whose canonical form hashes the same. Cheap, in-process,
        catches the duplicates produced by overlapping chunk boundaries."""
        seen: set[str] = set()
        out: list[dict[str, Any]] = []
        for r in rows:
            try:
                canonical = json.dumps(r, sort_keys=True, ensure_ascii=False, default=str)
            except Exception:
                out.append(r)
                continue
            h = hashlib.sha1(canonical.encode("utf-8")).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            out.append(r)
        return out

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
