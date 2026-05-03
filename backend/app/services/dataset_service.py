"""Dataset upload, parsing, schema introspection. Files live in uploads/;
metadata lives in data/datasets/<id>.json."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.api.schemas.dataset import DatasetSchema
from app.storage import store
from app.utils.config import settings


_SUPPORTED = {".csv", ".json", ".jsonl"}


def _infer_types(rows: list[dict[str, Any]], columns: list[str]) -> dict[str, str]:
    types: dict[str, str] = {}
    for col in columns:
        sample = next((r[col] for r in rows if col in r and r[col] is not None), None)
        if sample is None:
            types[col] = "unknown"
        elif isinstance(sample, bool):
            types[col] = "bool"
        elif isinstance(sample, int):
            types[col] = "int"
        elif isinstance(sample, float):
            types[col] = "float"
        elif isinstance(sample, (list, dict)):
            types[col] = "json"
        else:
            types[col] = "string"
    return types


def parse_file(path: Path, sample_n: int = 10) -> tuple[int, list[str], dict[str, str], list[dict[str, Any]]]:
    """Returns (row_count, columns, column_types, sample_rows)."""
    suffix = path.suffix.lower()
    if suffix not in _SUPPORTED:
        raise ValueError(f"Unsupported file type: {suffix}. Use CSV, JSON, or JSONL.")

    if suffix == ".csv":
        import pandas as pd  # local import; pandas is a heavy dep
        df = pd.read_csv(path)
        sample = df.head(sample_n).to_dict(orient="records")
        cols = [str(c) for c in df.columns.tolist()]
        return len(df), cols, _infer_types(sample, cols), sample

    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        cols = sorted({k for r in rows for k in r.keys()})
        return len(rows), cols, _infer_types(rows[:sample_n], cols), rows[:sample_n]

    # .json
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("JSON file must be an array of objects.")
    cols = sorted({k for r in data if isinstance(r, dict) for k in r.keys()})
    return len(data), cols, _infer_types(data[:sample_n], cols), data[:sample_n]


def save_uploaded(filename: str, contents: bytes) -> DatasetSchema:
    record_id = store.new_id()
    safe_name = Path(filename).name
    target = settings.upload_dir / f"{record_id}_{safe_name}"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(contents)

    row_count, cols, types, sample = parse_file(target)

    record = DatasetSchema(
        id=record_id,
        name=safe_name,
        file_path=str(target),
        file_type=target.suffix.lower().lstrip("."),
        row_count=row_count,
        column_names=cols,
        column_types=types,
        sample_rows=sample,
        size_bytes=target.stat().st_size,
        is_analyzed=False,
        analysis=None,
        created_at=datetime.now(timezone.utc),
    )
    store.write("datasets", record_id, record.model_dump(mode="json"))
    return record


def list_datasets() -> list[DatasetSchema]:
    out: list[DatasetSchema] = []
    for raw in store.list_all("datasets"):
        try:
            out.append(DatasetSchema(**raw))
        except Exception:
            continue
    out.sort(key=lambda d: d.created_at, reverse=True)
    return out


def get_dataset(dataset_id: str) -> DatasetSchema | None:
    raw = store.read("datasets", dataset_id)
    if not raw:
        return None
    return DatasetSchema(**raw)


def delete_dataset(dataset_id: str) -> bool:
    raw = store.read("datasets", dataset_id)
    if not raw:
        return False
    p = Path(raw.get("file_path", ""))
    if p.exists():
        try:
            p.unlink()
        except OSError:
            pass
    return store.delete("datasets", dataset_id)


def set_analysis(dataset_id: str, analysis: dict[str, Any]) -> DatasetSchema | None:
    raw = store.read("datasets", dataset_id)
    if not raw:
        return None
    raw["analysis"] = analysis
    raw["is_analyzed"] = True
    store.write("datasets", dataset_id, raw)
    return DatasetSchema(**raw)
