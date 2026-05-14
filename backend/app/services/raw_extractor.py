"""Universal raw extraction for "upload anything, get a trainable dataset".

Thin async-friendly wrappers around the synchronous primitives already in
``dataset_service``. All blocking IO and parsing runs via ``asyncio.to_thread``
so the AgenticLoop can call these without blocking the event loop.

Layout:
    walk_folder(root)              recursive file listing, async
    sniff_kind(path)               magic-byte + heuristic classifier
    extract_pdf / docx / html      typed extractors
    extract_text(path)             auto-routed extraction
    extract_many(paths)            parallel extraction, returns candidate records
    dedupe_records(records)        cheap n-gram dedupe at the line level
    write_jsonl(records, path)     dump records to disk for synthesize_unified_dataset

A "candidate record" is just ``{"source": str, "kind": str, "text": str,
"page": int|None}``. The AgenticLoop reads these, asks the LLM to project
them into a unified instruction/output schema, and writes the final JSONL
via dataset_service.register_synthetic.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from app.services import dataset_service as _ds


log = logging.getLogger("finetune-studio.raw-extractor")


@dataclass
class RawRecord:
    source: str
    kind: str
    text: str
    page: Optional[int] = None
    bytes_: int = 0


# ── Folder walk ───────────────────────────────────────────────────────────

async def walk_folder(root: str | Path) -> list[Path]:
    """Recursive listing. Skips hidden dirs, common junk, and binaries we
    cannot read."""
    return await asyncio.to_thread(_ds._recursive_scan, Path(root))


# ── Sniffing ──────────────────────────────────────────────────────────────

async def sniff_kind(path: str | Path) -> str:
    """Return one of: csv, tsv, json, jsonl, pdf, docx, html, text, binary,
    parquet. Same classifier dataset_service uses on upload."""
    return await asyncio.to_thread(_ds._sniff_file_type, Path(path))


# ── Text extraction ───────────────────────────────────────────────────────

async def extract_pdf(path: str | Path) -> list[RawRecord]:
    """One record per page. Pages with no text are skipped silently."""
    def _do() -> list[RawRecord]:
        import pypdf  # type: ignore
        p = Path(path)
        reader = pypdf.PdfReader(str(p))
        out: list[RawRecord] = []
        for i, page in enumerate(reader.pages, start=1):
            txt = (page.extract_text() or "").strip()
            if not txt:
                continue
            out.append(RawRecord(source=str(p), kind="pdf", text=txt, page=i,
                                 bytes_=len(txt.encode("utf-8"))))
        return out
    return await asyncio.to_thread(_do)


async def extract_docx(path: str | Path) -> RawRecord:
    def _do() -> RawRecord:
        import docx  # type: ignore
        p = Path(path)
        d = docx.Document(str(p))
        text = "\n".join(par.text for par in d.paragraphs if par.text.strip())
        return RawRecord(source=str(p), kind="docx", text=text,
                         bytes_=len(text.encode("utf-8")))
    return await asyncio.to_thread(_do)


async def extract_html(path: str | Path) -> RawRecord:
    def _do() -> RawRecord:
        from bs4 import BeautifulSoup  # type: ignore
        p = Path(path)
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
        for tag in soup(["script", "style", "noscript", "nav", "header",
                         "footer", "aside"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
        return RawRecord(source=str(p), kind="html", text=text,
                         bytes_=len(text.encode("utf-8")))
    return await asyncio.to_thread(_do)


async def extract_text(path: str | Path) -> Optional[RawRecord]:
    """Auto-route to the right extractor based on sniff_kind. Returns None
    for unrecognized binary files."""
    kind = await sniff_kind(path)
    p = Path(path)
    if kind == "pdf":
        records = await extract_pdf(p)
        if not records:
            return None
        merged = "\n\n".join(r.text for r in records)
        return RawRecord(source=str(p), kind="pdf", text=merged,
                         bytes_=len(merged.encode("utf-8")))
    if kind == "docx":
        return await extract_docx(p)
    if kind == "html":
        return await extract_html(p)
    if kind in ("text", "markdown") or kind.lower() in ("md", "txt"):
        def _read() -> RawRecord:
            txt = p.read_text(encoding="utf-8", errors="ignore")
            txt = _ds._clean_text(txt)
            return RawRecord(source=str(p), kind="text", text=txt,
                             bytes_=len(txt.encode("utf-8")))
        return await asyncio.to_thread(_read)
    if kind == "binary":
        return None
    # Structured types (csv, json, jsonl) - render as JSON text so the LLM
    # can reason about them as candidates without us forcing a schema.
    def _read_structured() -> Optional[RawRecord]:
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return None
        return RawRecord(source=str(p), kind=kind, text=text[:200_000],
                         bytes_=len(text.encode("utf-8")))
    return await asyncio.to_thread(_read_structured)


async def extract_many(paths: list[str | Path], *, concurrency: int = 6) -> list[RawRecord]:
    """Parallel extraction with bounded concurrency. Failures are logged
    and skipped, not raised."""
    sem = asyncio.Semaphore(concurrency)

    async def _one(pth: str | Path) -> Optional[RawRecord]:
        async with sem:
            try:
                return await extract_text(pth)
            except Exception as e:
                log.warning("extract failed for %s: %s", pth, e)
                return None

    results = await asyncio.gather(*(_one(p) for p in paths))
    return [r for r in results if r is not None]


# ── Dedupe ────────────────────────────────────────────────────────────────

def _line_fingerprints(text: str, *, min_len: int = 24) -> set[str]:
    """Cheap fingerprint set: lowercased lines, length-filtered, hashed.
    Good enough to spot copy-pasted headers/footers across files."""
    out: set[str] = set()
    for line in text.splitlines():
        s = line.strip().lower()
        if len(s) < min_len:
            continue
        out.add(hashlib.md5(s.encode("utf-8")).hexdigest()[:16])
    return out


def dedupe_records(records: list[RawRecord], *, jaccard: float = 0.85) -> list[RawRecord]:
    """Drop records whose line-fingerprint Jaccard similarity vs any earlier
    record exceeds the threshold. Stable order; first occurrence wins."""
    kept: list[tuple[RawRecord, set[str]]] = []
    out: list[RawRecord] = []
    for r in records:
        fp = _line_fingerprints(r.text)
        if not fp:
            out.append(r)
            continue
        is_dup = False
        for _, prev in kept:
            if not prev:
                continue
            inter = len(fp & prev)
            union = len(fp | prev) or 1
            if inter / union >= jaccard:
                is_dup = True
                break
        if not is_dup:
            kept.append((r, fp))
            out.append(r)
    return out


# ── Persistence ───────────────────────────────────────────────────────────

async def write_jsonl(records: list[dict[str, Any]], path: str | Path) -> int:
    """Write records as one JSON object per line. Returns count."""
    def _do() -> int:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        return len(records)
    return await asyncio.to_thread(_do)


def record_to_dict(r: RawRecord) -> dict[str, Any]:
    return asdict(r)
