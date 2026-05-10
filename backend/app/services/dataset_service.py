"""Dataset upload, parsing, schema introspection. Files live in uploads/;
metadata lives in data/datasets/<id>.json.

Supports two flavours of input:

    structured        .csv .json .jsonl - already in pair / row form,
                      flows directly to the profiling agent.

    raw_doc           .txt .md .pdf .docx - free text. Stored as a single
                      pseudo-row {"document": "...full text..."}; the
                      DataRestructurerAgent later converts it into a
                      proper instruction / chat / qa dataset.

The "kind" attribute is set on the DatasetSchema (when the schema permits)
so downstream agents can branch on raw vs structured without re-sniffing
the file extension.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.api.schemas.dataset import DatasetSchema
from app.storage import store
from app.utils.config import settings


_STRUCTURED = {".csv", ".json", ".jsonl"}
_RAW_DOC = {".txt", ".md", ".rst", ".pdf", ".docx", ".html"}
_SUPPORTED = _STRUCTURED | _RAW_DOC


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


def _extract_doc_text(path: Path) -> str:
    """Universal text extractor for raw-doc inputs. Each branch is wrapped
    in a defensive try so missing optional deps surface as a helpful error
    rather than crashing the whole upload."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            import pypdf  # type: ignore
        except Exception as e:
            raise ValueError(f"PDF support requires pypdf - pip install pypdf ({e})")
        reader = pypdf.PdfReader(str(path))
        return "\n\n".join((p.extract_text() or "") for p in reader.pages)
    if suffix == ".docx":
        try:
            import docx  # type: ignore  # python-docx
        except Exception as e:
            raise ValueError(f"DOCX support requires python-docx - pip install python-docx ({e})")
        d = docx.Document(str(path))
        return "\n".join(p.text for p in d.paragraphs if p.text.strip())
    if suffix == ".html":
        try:
            from bs4 import BeautifulSoup  # type: ignore
        except Exception as e:
            raise ValueError(f"HTML support requires beautifulsoup4 - pip install beautifulsoup4 ({e})")
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
        return soup.get_text(separator="\n")
    # .txt / .md / .rst (and anything else falling through)
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def parse_file(path: Path, sample_n: int = 10) -> tuple[str, int, list[str], dict[str, str], list[dict[str, Any]]]:
    """Returns (kind, row_count, columns, column_types, sample_rows).

    For raw_doc, row_count is 1 (the document) and the schema is a single
    "document" string column.
    """
    suffix = path.suffix.lower()
    if suffix not in _SUPPORTED:
        raise ValueError(
            f"Unsupported file type: {suffix}. "
            f"Supported: structured ({', '.join(sorted(_STRUCTURED))}) "
            f"or raw docs ({', '.join(sorted(_RAW_DOC))})."
        )

    if suffix in _RAW_DOC:
        text = _extract_doc_text(path)
        if not text.strip():
            raise ValueError("could not extract any text from document")
        sample = [{"document": text[:2000]}]  # preview
        return "raw_doc", 1, ["document"], {"document": "string"}, sample

    if suffix == ".csv":
        import pandas as pd  # local import; pandas is a heavy dep
        df = pd.read_csv(path)
        sample = df.head(sample_n).to_dict(orient="records")
        cols = [str(c) for c in df.columns.tolist()]
        return "structured", len(df), cols, _infer_types(sample, cols), sample

    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        cols = sorted({k for r in rows for k in r.keys()})
        return "structured", len(rows), cols, _infer_types(rows[:sample_n], cols), rows[:sample_n]

    # .json
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("JSON file must be an array of objects.")
    cols = sorted({k for r in data if isinstance(r, dict) for k in r.keys()})
    return "structured", len(data), cols, _infer_types(data[:sample_n], cols), data[:sample_n]


def save_uploaded(filename: str, contents: bytes) -> DatasetSchema:
    record_id = store.new_id()
    safe_name = Path(filename).name
    target = settings.upload_dir / f"{record_id}_{safe_name}"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(contents)

    kind, row_count, cols, types, sample = parse_file(target)

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
        analysis={"kind": kind},
        created_at=datetime.now(timezone.utc),
    )
    store.write("datasets", record_id, record.model_dump(mode="json"))
    return record


def register_synthetic(
    *,
    name: str,
    rows: list[dict[str, Any]],
    parent_dataset_id: str | None = None,
    file_type: str = "jsonl",
    note: str = "agent-generated",
) -> DatasetSchema:
    """Register a fresh dataset produced by an agent (e.g. the restructurer).

    Writes the rows to ``uploads/<id>_<name>.jsonl`` and creates a normal
    DatasetSchema so the rest of the pipeline treats it identically to a
    user upload.
    """
    record_id = store.new_id()
    safe_name = Path(name).name
    target = settings.upload_dir / f"{record_id}_{safe_name}"
    target.parent.mkdir(parents=True, exist_ok=True)

    with target.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str, ensure_ascii=False) + "\n")

    cols = sorted({k for r in rows for k in r.keys()})
    types = _infer_types(rows[:10], cols)
    record = DatasetSchema(
        id=record_id,
        name=safe_name,
        file_path=str(target),
        file_type=file_type,
        row_count=len(rows),
        column_names=cols,
        column_types=types,
        sample_rows=rows[:10],
        size_bytes=target.stat().st_size,
        is_analyzed=True,
        analysis={"kind": "structured", "synthetic": True, "note": note,
                  "parent_dataset_id": parent_dataset_id},
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


def is_raw_doc(dataset: DatasetSchema | None) -> bool:
    if not dataset:
        return False
    analysis = dataset.analysis or {}
    return analysis.get("kind") == "raw_doc"


def read_doc_text(dataset: DatasetSchema) -> str:
    """Fetch the full extracted text for a raw_doc dataset."""
    p = Path(dataset.file_path)
    if not p.exists():
        return ""
    return _extract_doc_text(p)


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
    existing = raw.get("analysis") or {}
    existing.update(analysis)
    raw["analysis"] = existing
    raw["is_analyzed"] = True
    store.write("datasets", dataset_id, raw)
    return DatasetSchema(**raw)
