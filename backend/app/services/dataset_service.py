"""Dataset upload, parsing, schema introspection. Files live in uploads/;
metadata lives in data/datasets/<id>.json.

v4.0 — Universal Intake Engine.

Replaces hardcoded extension maps with a probabilistic, extension-agnostic
parser. Supports:

    * Recursive folder ingestion — upload a directory of mixed files.
    * Magic-byte header sniffing — infers type from content, not extension.
    * Cross-file aggregation — multiple files merged into one dataset.
    * Autonomous de-noising — strips boilerplate / headers before ingestion.

Kind resolution priority:
    1. Structured files (.csv, .json, .jsonl, or sniffed as such)
    2. Raw documents (PDF, DOCX, HTML, or any text-decodable file)
    3. Binary files are skipped with a warning.
"""
from __future__ import annotations

import json
import logging
import mimetypes
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.api.schemas.dataset import DatasetSchema
from app.storage import store
from app.utils.config import settings


log = logging.getLogger("finetune-studio.dataset")

# ── Known structural types (fast-path, no sniffing needed) ────────────────
_STRUCTURED_EXTS = {".csv", ".json", ".jsonl", ".tsv", ".parquet"}
_RAW_DOC_EXTS = {".txt", ".md", ".rst", ".pdf", ".docx", ".html", ".htm",
                 ".xml", ".yaml", ".yml", ".log", ".tex", ".rtf"}


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


# ── Universal file-type sniffer ───────────────────────────────────────────

def _sniff_file_type(path: Path) -> str:
    """Infer file type from content headers, not just extension.

    Returns one of: 'csv', 'json', 'jsonl', 'pdf', 'docx', 'html', 'text'.
    Falls back to extension-based guess, then to 'text' as the universal
    catch-all (if the file is decodable).
    """
    # 1. Check the magic bytes (file header)
    try:
        with path.open("rb") as f:
            header = f.read(2048)
    except Exception:
        return "text"

    # PDF magic: %PDF
    if header[:5] == b"%PDF-":
        return "pdf"

    # DOCX / XLSX / ZIP magic: PK\x03\x04
    if header[:4] == b"PK\x03\x04":
        return "docx"  # Could also be xlsx/pptx — docx handler will verify

    # HTML detection
    if b"<html" in header[:500].lower() or b"<!doctype html" in header[:500].lower():
        return "html"

    # Try to decode as text — if it works, inspect structure
    try:
        text = header.decode("utf-8", errors="strict")
    except (UnicodeDecodeError, ValueError):
        # Binary file we can't read as text
        return "binary"

    # JSON array or object?
    stripped = text.lstrip()
    if stripped.startswith("[") or stripped.startswith("{"):
        # Is it JSONL (one JSON object per line)?
        lines = text.strip().splitlines()
        if len(lines) > 1:
            try:
                json.loads(lines[0])
                json.loads(lines[1])
                return "jsonl"
            except Exception:
                pass
        try:
            json.loads(text + "...")  # Partial won't parse; read full file
        except Exception:
            pass
        return "json"

    # CSV heuristic: tab or comma separated with consistent column count
    lines = text.strip().splitlines()[:5]
    if len(lines) >= 2:
        for sep in (",", "\t"):
            counts = [len(line.split(sep)) for line in lines]
            if counts[0] >= 2 and len(set(counts)) == 1:
                return "csv" if sep == "," else "tsv"

    # Fallback by extension
    ext = path.suffix.lower()
    if ext in _STRUCTURED_EXTS:
        return ext.lstrip(".")
    if ext in _RAW_DOC_EXTS:
        return ext.lstrip(".")

    # Ultimate fallback: it's text
    return "text"


def _is_structured_type(file_type: str) -> bool:
    return file_type in ("csv", "tsv", "json", "jsonl", "parquet")


# ── Document text extraction (extension-agnostic) ────────────────────────

def _extract_doc_text(path: Path, file_type: str | None = None) -> str:
    """Universal text extractor. Uses sniffed type, not extension."""
    if file_type is None:
        file_type = _sniff_file_type(path)

    if file_type == "pdf":
        try:
            import pypdf  # type: ignore
        except Exception as e:
            raise ValueError(f"PDF support requires pypdf - pip install pypdf ({e})")
        reader = pypdf.PdfReader(str(path))
        return "\n\n".join((p.extract_text() or "") for p in reader.pages)

    if file_type == "docx":
        try:
            import docx  # type: ignore  # python-docx
        except Exception as e:
            raise ValueError(f"DOCX support requires python-docx - pip install python-docx ({e})")
        d = docx.Document(str(path))
        return "\n".join(p.text for p in d.paragraphs if p.text.strip())

    if file_type == "html":
        try:
            from bs4 import BeautifulSoup  # type: ignore
        except Exception as e:
            raise ValueError(f"HTML support requires beautifulsoup4 - pip install beautifulsoup4 ({e})")
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
        return soup.get_text(separator="\n")

    # Everything else: try reading as plain text
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""


def _clean_text(text: str) -> str:
    """Autonomous de-noising: strip common boilerplate patterns."""
    import re

    # Remove excessive whitespace (3+ blank lines → 2)
    text = re.sub(r"\n{4,}", "\n\n\n", text)

    # Remove page headers/footers (repeated short lines)
    lines = text.splitlines()
    cleaned: list[str] = []
    line_counts: dict[str, int] = {}
    for line in lines:
        stripped = line.strip()
        if stripped:
            line_counts[stripped] = line_counts.get(stripped, 0) + 1

    # Lines repeated more than 3 times and shorter than 80 chars are likely
    # headers/footers/boilerplate.
    boilerplate = {line for line, count in line_counts.items()
                   if count > 3 and len(line) < 80}

    for line in lines:
        if line.strip() not in boilerplate:
            cleaned.append(line)

    return "\n".join(cleaned).strip()


# ── Recursive directory scanning ──────────────────────────────────────────

def _recursive_scan(root: Path) -> list[Path]:
    """Walk a directory recursively and return all files, regardless of
    extension. Skips hidden files/dirs and common junk."""
    _SKIP_DIRS = {".git", ".svn", "__pycache__", ".DS_Store", "node_modules", ".venv", "venv"}
    _SKIP_FILES = {".gitignore", ".dockerignore", "Thumbs.db", ".DS_Store"}

    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune hidden and junk directories
        dirnames[:] = [d for d in dirnames
                       if d not in _SKIP_DIRS and not d.startswith(".")]
        for fname in filenames:
            if fname in _SKIP_FILES or fname.startswith("."):
                continue
            files.append(Path(dirpath) / fname)
    return sorted(files)


# ── Core parsing (extension-agnostic) ─────────────────────────────────────

def parse_file(path: Path, sample_n: int = 10) -> tuple[str, int, list[str], dict[str, str], list[dict[str, Any]]]:
    """Returns (kind, row_count, columns, column_types, sample_rows).

    v4.0: Extension-agnostic. Sniffs the file content to determine type.
    For raw_doc, row_count is 1 (the document) and the schema is a single
    "document" string column.
    """
    file_type = _sniff_file_type(path)

    if file_type == "binary":
        raise ValueError(
            f"Cannot parse binary file: {path.name}. "
            "Only text-based and structured data files are supported."
        )

    # ── Structured paths ──────────────────────────────────────────────
    if file_type == "csv" or file_type == "tsv":
        import pandas as pd  # local import; pandas is a heavy dep
        sep = "\t" if file_type == "tsv" else ","
        df = pd.read_csv(path, sep=sep)
        sample = df.head(sample_n).to_dict(orient="records")
        cols = [str(c) for c in df.columns.tolist()]
        return "structured", len(df), cols, _infer_types(sample, cols), sample

    if file_type == "jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        cols = sorted({k for r in rows for k in r.keys()})
        return "structured", len(rows), cols, _infer_types(rows[:sample_n], cols), rows[:sample_n]

    if file_type == "json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            cols = sorted({k for r in data if isinstance(r, dict) for k in r.keys()})
            return "structured", len(data), cols, _infer_types(data[:sample_n], cols), data[:sample_n]
        # Single JSON object — treat as raw doc
        text = json.dumps(data, indent=2)
        sample = [{"document": text[:2000]}]
        return "raw_doc", 1, ["document"], {"document": "string"}, sample

    if file_type == "parquet":
        try:
            import pandas as pd  # type: ignore
            df = pd.read_parquet(path)
            sample = df.head(sample_n).to_dict(orient="records")
            cols = [str(c) for c in df.columns.tolist()]
            return "structured", len(df), cols, _infer_types(sample, cols), sample
        except Exception as e:
            log.warning("Failed to read parquet %s: %s", path, e)

    # ── Raw document path ─────────────────────────────────────────────
    text = _extract_doc_text(path, file_type)
    if text:
        text = _clean_text(text)
    if not text or not text.strip():
        raise ValueError(f"Could not extract any text from {path.name}")
    sample = [{"document": text[:2000]}]
    return "raw_doc", 1, ["document"], {"document": "string"}, sample


# ── Multi-file aggregation ────────────────────────────────────────────────

def _aggregate_files(
    paths: list[Path],
) -> tuple[str, int, list[str], dict[str, str], list[dict[str, Any]], list[dict[str, Any]]]:
    """Parse multiple files and aggregate into one dataset.

    Returns (kind, row_count, columns, column_types, sample_rows, all_rows).
    """
    all_structured_rows: list[dict[str, Any]] = []
    all_doc_texts: list[str] = []
    structured_count = 0
    raw_count = 0

    for path in paths:
        try:
            file_type = _sniff_file_type(path)
            if file_type == "binary":
                log.warning("Skipping binary file: %s", path)
                continue

            kind, row_count, cols, types, sample = parse_file(path)
            if kind == "structured":
                structured_count += 1
                # Load all rows for structured files
                if file_type in ("csv", "tsv"):
                    import pandas as pd
                    sep = "\t" if file_type == "tsv" else ","
                    df = pd.read_csv(path, sep=sep)
                    all_structured_rows.extend(df.to_dict(orient="records"))
                elif file_type == "jsonl":
                    with path.open("r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                all_structured_rows.append(json.loads(line))
                elif file_type == "json":
                    with path.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        all_structured_rows.extend(r for r in data if isinstance(r, dict))
                else:
                    all_structured_rows.extend(sample)
            else:
                raw_count += 1
                text = _extract_doc_text(path, file_type)
                if text and text.strip():
                    text = _clean_text(text)
                    all_doc_texts.append(f"--- {path.name} ---\n{text}")
        except Exception as e:
            log.warning("Failed to parse %s: %s", path, e)
            continue

    # If we got structured rows, prefer those
    if all_structured_rows:
        cols = sorted({k for r in all_structured_rows for k in r.keys()})
        types = _infer_types(all_structured_rows[:10], cols)
        return (
            "structured",
            len(all_structured_rows),
            cols,
            types,
            all_structured_rows[:10],
            all_structured_rows,
        )

    # Otherwise, merge all raw docs into one big document
    if all_doc_texts:
        merged = "\n\n".join(all_doc_texts)
        sample = [{"document": merged[:2000]}]
        return (
            "raw_doc",
            1,
            ["document"],
            {"document": "string"},
            sample,
            [{"document": merged}],
        )

    raise ValueError("No parseable files found in the uploaded directory")


# ── Public API ────────────────────────────────────────────────────────────

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
        file_type=_sniff_file_type(target),
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


def save_uploaded_directory(dir_path: Path) -> DatasetSchema:
    """Ingest an entire directory recursively.

    Scans all files, sniffs their types, parses them, and aggregates into
    a single dataset. Structured files are row-merged; raw docs are
    concatenated into a cross-file document for LLM restructuring.
    """
    if not dir_path.is_dir():
        raise ValueError(f"{dir_path} is not a directory")

    files = _recursive_scan(dir_path)
    if not files:
        raise ValueError(f"No files found in {dir_path}")

    log.info("Recursive scan found %d files in %s", len(files), dir_path)

    kind, row_count, cols, types, sample, all_rows = _aggregate_files(files)

    record_id = store.new_id()
    safe_name = dir_path.name or "uploaded_directory"

    # Write aggregated data to a JSONL file
    target = settings.upload_dir / f"{record_id}_{safe_name}.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")

    total_size = sum(p.stat().st_size for p in files if p.exists())

    record = DatasetSchema(
        id=record_id,
        name=safe_name,
        file_path=str(target),
        file_type="jsonl",
        row_count=row_count,
        column_names=cols,
        column_types=types,
        sample_rows=sample,
        size_bytes=total_size,
        is_analyzed=False,
        analysis={
            "kind": kind,
            "source": "directory",
            "file_count": len(files),
            "files": [str(f.name) for f in files[:50]],
        },
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
    file_type = _sniff_file_type(p)
    text = _extract_doc_text(p, file_type)
    return _clean_text(text) if text else ""


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
