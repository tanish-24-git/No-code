"""Schema validation + light repair for the rows the DataRestructurerAgent
emits before they hit the trainer.

The restructurer's per-chunk LLM passes occasionally produce rows that don't
match the target format (missing keys, wrong types, empty strings, accidental
double-wrapping). The trainer is strict: a single malformed row can crash
``trl.SFTTrainer`` mid-run. We validate here so the bad rows are dropped or
lightly repaired before they ever land on disk.

Public API:
    validate_rows(rows, target_format) -> (clean_rows, summary)
        summary is a dict: {kept, dropped, repaired, reasons: list[str]}

Repair is intentionally conservative — we only fix obvious mistakes:
    * bare string ``messages`` wrapped as a user-role list
    * whitespace-only fields trimmed to empty (then dropped if required)
    * None / null fields dropped if the field is optional
"""
from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator


log = logging.getLogger("finetune-studio.jsonl-validator")


# ── per-format pydantic schemas ────────────────────────────────────────────


class _InstructionRow(BaseModel):
    instruction: str
    input: str = ""
    output: str

    @field_validator("instruction", "output")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("must be non-empty")
        return v.strip()


class _ChatMessage(BaseModel):
    role: str
    content: str

    @field_validator("role")
    @classmethod
    def _role_ok(cls, v: str) -> str:
        v = (v or "").strip().lower()
        if v not in ("system", "user", "assistant", "tool"):
            raise ValueError(f"invalid role: {v}")
        return v

    @field_validator("content")
    @classmethod
    def _content_non_empty(cls, v: str) -> str:
        if v is None:
            raise ValueError("content cannot be null")
        v = str(v)
        if not v.strip():
            raise ValueError("content cannot be empty")
        return v


class _ChatRow(BaseModel):
    messages: list[_ChatMessage] = Field(min_length=1)


class _QARow(BaseModel):
    instruction: str
    output: str

    @field_validator("instruction", "output")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("must be non-empty")
        return v.strip()


class _ClassificationRow(BaseModel):
    text: str
    label: str

    @field_validator("text", "label")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("must be non-empty")
        return v.strip()


class _LanguageModelingRow(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("must be non-empty")
        return v.strip()


_FORMAT_TO_SCHEMA: dict[str, type[BaseModel]] = {
    "instruction": _InstructionRow,
    "chat": _ChatRow,
    "qa": _QARow,
    "classification": _ClassificationRow,
    "language_modeling": _LanguageModelingRow,
}


# ── repair pass ────────────────────────────────────────────────────────────


def _repair_chat(row: dict[str, Any]) -> dict[str, Any] | None:
    """Coerce common chat-row mistakes back to ``{messages: [{role, content}, ...]}``."""
    msgs = row.get("messages")
    # Bare string → wrap as a single user turn.
    if isinstance(msgs, str) and msgs.strip():
        return {"messages": [{"role": "user", "content": msgs.strip()}]}
    # Single dict → wrap in a list.
    if isinstance(msgs, dict):
        return {"messages": [msgs]}
    # Mistakenly emitted instruction/output → lift into a 2-turn chat.
    if "instruction" in row and "output" in row:
        i, o = row.get("instruction"), row.get("output")
        if isinstance(i, str) and i.strip() and isinstance(o, str) and o.strip():
            return {"messages": [
                {"role": "user", "content": i.strip()},
                {"role": "assistant", "content": o.strip()},
            ]}
    # Mistakenly emitted prompt/completion.
    if "prompt" in row and "completion" in row:
        p, c = row.get("prompt"), row.get("completion")
        if isinstance(p, str) and p.strip() and isinstance(c, str) and c.strip():
            return {"messages": [
                {"role": "user", "content": p.strip()},
                {"role": "assistant", "content": c.strip()},
            ]}
    return None


def _repair_instruction(row: dict[str, Any]) -> dict[str, Any] | None:
    # Sometimes the LLM emits ``question/answer`` for an instruction ask.
    if "question" in row and "answer" in row:
        q, a = row.get("question"), row.get("answer")
        if isinstance(q, str) and q.strip() and isinstance(a, str) and a.strip():
            return {
                "instruction": q.strip(),
                "input": (row.get("input") or "").strip() if isinstance(row.get("input"), str) else "",
                "output": a.strip(),
            }
    return None


def _repair_lm(row: dict[str, Any]) -> dict[str, Any] | None:
    # Any non-empty stringy field becomes the LM text.
    for k in ("text", "output", "content", "answer", "instruction"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return {"text": v.strip()}
    return None


_REPAIRERS = {
    "chat": _repair_chat,
    "instruction": _repair_instruction,
    "qa": _repair_instruction,  # qa ≈ instruction (sans input)
    "language_modeling": _repair_lm,
}


# ── public API ─────────────────────────────────────────────────────────────


def validate_rows(
    rows: list[dict[str, Any]],
    target_format: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Filter rows that don't match ``target_format``. Lightly repair common
    mistakes. Returns ``(clean_rows, summary)``.

    summary: {
        "kept": int, "dropped": int, "repaired": int,
        "reasons": list[str],  # up to 5 representative error messages
    }
    """
    schema = _FORMAT_TO_SCHEMA.get(target_format)
    if schema is None:
        # Unknown format: pass through unchanged (caller is responsible).
        return list(rows), {
            "kept": len(rows),
            "dropped": 0,
            "repaired": 0,
            "reasons": [f"unknown format '{target_format}' — validation skipped"],
        }

    repairer = _REPAIRERS.get(target_format)
    clean: list[dict[str, Any]] = []
    dropped = 0
    repaired = 0
    reasons: list[str] = []

    for row in rows:
        if not isinstance(row, dict):
            dropped += 1
            if len(reasons) < 5:
                reasons.append(f"row is not a dict (type={type(row).__name__})")
            continue
        try:
            validated = schema.model_validate(row)
            clean.append(validated.model_dump(mode="json"))
            continue
        except ValidationError as e:
            # Try repair before giving up.
            if repairer is not None:
                fixed = repairer(row)
                if fixed is not None:
                    try:
                        validated = schema.model_validate(fixed)
                        clean.append(validated.model_dump(mode="json"))
                        repaired += 1
                        continue
                    except ValidationError:
                        pass
            dropped += 1
            if len(reasons) < 5:
                # Take the first error message (terse).
                first = e.errors()[0] if e.errors() else {}
                reasons.append(
                    f"{'.'.join(str(x) for x in first.get('loc', ())) or 'row'}: {first.get('msg', 'invalid')}"
                )

    summary = {
        "kept": len(clean),
        "dropped": dropped,
        "repaired": repaired,
        "reasons": reasons,
    }
    if dropped or repaired:
        log.info(
            "jsonl_validator: format=%s kept=%d dropped=%d repaired=%d",
            target_format, len(clean), dropped, repaired,
        )
    return clean, summary
