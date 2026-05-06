"""Dataset inspection tools. Pure read-only; no LLM calls."""
from __future__ import annotations

import collections
from pathlib import Path
from typing import Any

from app.services import dataset_service
from app.tools.registry import ToolContext, tool


def _load_dataset(dataset_id: str):
    ds = dataset_service.get_dataset(dataset_id)
    if not ds:
        return None
    return ds


@tool(
    name="dataset.inspect",
    description="Return dataset metadata (id, name, row_count, columns, types, size).",
    input_schema={
        "type": "object",
        "properties": {"dataset_id": {"type": "string"}},
        "required": ["dataset_id"],
    },
)
async def dataset_inspect(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    ds = _load_dataset(args["dataset_id"])
    if not ds:
        return {"error": "dataset not found"}
    return {
        "id": ds.id,
        "name": ds.name,
        "file_type": ds.file_type,
        "row_count": ds.row_count,
        "column_names": ds.column_names,
        "column_types": ds.column_types,
        "size_bytes": ds.size_bytes,
    }


@tool(
    name="dataset.read_sample",
    description="Read up to N sample rows from the dataset for human-readable inspection.",
    input_schema={
        "type": "object",
        "properties": {
            "dataset_id": {"type": "string"},
            "n": {"type": "integer", "default": 5, "minimum": 1, "maximum": 50},
        },
        "required": ["dataset_id"],
    },
)
async def dataset_read_sample(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    ds = _load_dataset(args["dataset_id"])
    if not ds:
        return {"error": "dataset not found"}
    n = int(args.get("n", 5))
    return {"sample": ds.sample_rows[:n]}


@tool(
    name="dataset.profile_tokens",
    description="Estimate per-row token-length distribution. Uses a cheap whitespace heuristic by default; if a tokenizer name is given, uses transformers AutoTokenizer.",
    input_schema={
        "type": "object",
        "properties": {
            "dataset_id": {"type": "string"},
            "text_field": {"type": "string"},
            "tokenizer": {"type": "string"},
        },
        "required": ["dataset_id"],
    },
)
async def dataset_profile_tokens(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    ds = _load_dataset(args["dataset_id"])
    if not ds:
        return {"error": "dataset not found"}
    text_field = args.get("text_field")
    rows = _read_all_rows(ds.file_path, ds.file_type)
    if not rows:
        return {"error": "could not read rows"}

    # Pick a text field if not given: longest avg-string column.
    if not text_field:
        text_field = _pick_text_field(rows)
    if not text_field:
        return {"error": "no text-like field found"}

    sizes: list[int] = []
    tokenizer_name = args.get("tokenizer")
    enc = None
    if tokenizer_name:
        try:
            from transformers import AutoTokenizer  # type: ignore
            enc = AutoTokenizer.from_pretrained(tokenizer_name)
        except Exception:
            enc = None
    for r in rows:
        v = r.get(text_field)
        if not isinstance(v, str):
            continue
        if enc is not None:
            sizes.append(len(enc.encode(v, add_special_tokens=False)))
        else:
            # Whitespace heuristic, ~1.3 tokens/word for English.
            sizes.append(int(len(v.split()) * 1.3))

    if not sizes:
        return {"error": "no string values in field", "field": text_field}
    sizes.sort()
    n = len(sizes)
    return {
        "field": text_field,
        "tokenizer": tokenizer_name or "whitespace_heuristic",
        "n": n,
        "mean": sum(sizes) / n,
        "min": sizes[0],
        "p50": sizes[n // 2],
        "p95": sizes[max(0, int(n * 0.95) - 1)],
        "max": sizes[-1],
    }


@tool(
    name="dataset.detect_duplicates",
    description="Estimate duplicate row percentage by hashing rows.",
    input_schema={
        "type": "object",
        "properties": {"dataset_id": {"type": "string"}},
        "required": ["dataset_id"],
    },
)
async def dataset_detect_duplicates(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    ds = _load_dataset(args["dataset_id"])
    if not ds:
        return {"error": "dataset not found"}
    rows = _read_all_rows(ds.file_path, ds.file_type)
    if not rows:
        return {"duplicate_count": 0, "duplicate_pct": 0.0}
    seen: set[str] = set()
    dup = 0
    import json
    for r in rows:
        key = json.dumps(r, sort_keys=True, default=str)
        if key in seen:
            dup += 1
        else:
            seen.add(key)
    pct = (dup / len(rows)) * 100.0
    return {"duplicate_count": dup, "duplicate_pct": round(pct, 2), "row_count": len(rows)}


@tool(
    name="dataset.detect_missing",
    description="Per-column missing-value count and percentage.",
    input_schema={
        "type": "object",
        "properties": {"dataset_id": {"type": "string"}},
        "required": ["dataset_id"],
    },
)
async def dataset_detect_missing(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    ds = _load_dataset(args["dataset_id"])
    if not ds:
        return {"error": "dataset not found"}
    rows = _read_all_rows(ds.file_path, ds.file_type)
    n = len(rows) or 1
    out: dict[str, dict[str, Any]] = {}
    for col in ds.column_names:
        missing = sum(1 for r in rows if col not in r or r[col] in (None, ""))
        out[col] = {"missing": missing, "missing_pct": round((missing / n) * 100.0, 2)}
    return {"per_column": out, "row_count": len(rows)}


@tool(
    name="dataset.detect_imbalance",
    description="If a column looks like a label (low cardinality), report class frequencies.",
    input_schema={
        "type": "object",
        "properties": {
            "dataset_id": {"type": "string"},
            "label_field": {"type": "string"},
        },
        "required": ["dataset_id"],
    },
)
async def dataset_detect_imbalance(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    ds = _load_dataset(args["dataset_id"])
    if not ds:
        return {"error": "dataset not found"}
    rows = _read_all_rows(ds.file_path, ds.file_type)
    field = args.get("label_field") or _pick_label_field(rows, ds.column_names)
    if not field:
        return {"label_field": None, "balanced": None, "note": "no low-cardinality field detected"}
    counter: collections.Counter = collections.Counter()
    for r in rows:
        v = r.get(field)
        if v is None:
            continue
        counter[str(v)] += 1
    if not counter:
        return {"label_field": field, "balanced": None, "note": "no values"}
    total = sum(counter.values())
    freqs = {k: round(v / total, 4) for k, v in counter.items()}
    minority = min(freqs.values()) if freqs else 0.0
    return {
        "label_field": field,
        "classes": dict(counter),
        "frequencies": freqs,
        "minority_pct": round(minority * 100, 2),
        "balanced": minority >= 0.1,
    }


@tool(
    name="dataset.summarize_fields",
    description="Quick semantic-flavored summary: which fields look like input text, target, instruction, etc.",
    input_schema={
        "type": "object",
        "properties": {"dataset_id": {"type": "string"}},
        "required": ["dataset_id"],
    },
)
async def dataset_summarize_fields(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    ds = _load_dataset(args["dataset_id"])
    if not ds:
        return {"error": "dataset not found"}
    candidates = {
        "instruction_like": [],
        "input_like": [],
        "output_like": [],
        "label_like": [],
        "id_like": [],
    }
    name_hints = {
        "instruction_like": ("instruction", "prompt", "question", "query"),
        "input_like":       ("input", "context", "passage", "text", "content"),
        "output_like":      ("output", "answer", "response", "completion", "target"),
        "label_like":       ("label", "category", "class", "tag", "intent"),
        "id_like":          ("id", "uuid"),
    }
    for col in ds.column_names:
        lower = col.lower()
        for bucket, hints in name_hints.items():
            if any(h in lower for h in hints):
                candidates[bucket].append(col)
                break
    return {
        "field_buckets": candidates,
        "all_fields": ds.column_names,
        "type_hints": ds.column_types,
    }


# ── helpers ────────────────────────────────────────────────────────────────

def _read_all_rows(file_path: str, file_type: str) -> list[dict[str, Any]]:
    p = Path(file_path)
    if not p.exists():
        return []
    try:
        if file_type == "csv":
            import pandas as pd  # heavy; lazy
            df = pd.read_csv(p)
            return df.to_dict(orient="records")
        if file_type == "jsonl":
            import json
            rows: list[dict[str, Any]] = []
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rows.append(json.loads(line))
            return rows
        if file_type == "json":
            import json
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return [r for r in data if isinstance(r, dict)] if isinstance(data, list) else []
    except Exception:
        return []
    return []


def _pick_text_field(rows: list[dict[str, Any]]) -> str | None:
    if not rows:
        return None
    counts: dict[str, int] = {}
    for r in rows[:500]:
        for k, v in r.items():
            if isinstance(v, str) and len(v) > 16:
                counts[k] = counts.get(k, 0) + len(v)
    if not counts:
        return None
    return max(counts, key=lambda k: counts[k])


def _pick_label_field(rows: list[dict[str, Any]], cols: list[str]) -> str | None:
    """A column is label-like if cardinality / row_count <= 0.05 and >= 2."""
    if not rows:
        return None
    n = len(rows)
    best: tuple[float, str] | None = None
    for col in cols:
        values = {str(r.get(col)) for r in rows if r.get(col) is not None}
        card = len(values)
        if card < 2 or card > max(20, int(n * 0.05)):
            continue
        score = card / max(n, 1)
        if best is None or score < best[0]:
            best = (score, col)
    return best[1] if best else None
