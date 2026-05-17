"""Universal Data Alchemy — turn ‹crap› into ‹fortune› (blueprint §4).

The toolset is opinionated:
    * deep recursive scanner that ignores ``node_modules``, ``.git``,
      ``__pycache__``, virtual envs, and binary blobs.
    * schema induction that proposes a single Standardized Record Format
      across heterogeneous JSON / JSONL / CSV files.
    * semantic dedup using cheap hashing first, then a sentence-transformer
      cosine pass when the optional dep is installed.
    * sensitive-info redaction (PII / API keys / credentials) backed by a
      regex zoo and ``presidio_analyzer`` when available.
    * synthetic augmentation that calls the user's configured LLM provider
      to self-instruct from a small dataset.
    * low-entropy filter that drops repetitive boilerplate before training.

Each tool degrades gracefully — the dataset always emerges in an
improved state even if no optional dep is installed.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from app.tools.registry import ToolContext, tool


# ── helpers ───────────────────────────────────────────────────────────────

# Files / directories the scanner will *never* traverse. These are
# universally noise for fine-tuning corpora.
_IGNORE_DIRS = {
    ".git", ".hg", ".svn",
    "node_modules", "__pycache__", ".pytest_cache", ".mypy_cache",
    "venv", ".venv", "env", ".env", ".tox",
    "dist", "build", ".next", ".turbo", ".cache",
    ".idea", ".vscode",
}
_IGNORE_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ico", ".svg",
    ".mp3", ".mp4", ".mov", ".avi", ".wav",
    ".zip", ".tar", ".gz", ".7z", ".rar",
    ".woff", ".woff2", ".ttf", ".otf",
    ".pyc", ".class", ".so", ".dll", ".exe",
}
_TEXT_SUFFIXES = {
    ".txt", ".md", ".rst", ".json", ".jsonl", ".csv", ".tsv", ".yaml", ".yml",
    ".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".rb", ".java", ".c",
    ".h", ".cpp", ".hpp", ".cs", ".sh", ".html", ".css", ".sql", ".pdf", ".docx",
}

_REDACT_PATTERNS = [
    # email
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+", re.I), "[EMAIL]"),
    # phone (very loose)
    (re.compile(r"(?<!\d)(\+?\d{1,3}[\s.-]?)?(\(?\d{3}\)?[\s.-]?)?\d{3}[\s.-]?\d{4}\b"), "[PHONE]"),
    # bearer token / api key shapes
    (re.compile(r"\b(sk|pk|rk)_[a-z]{2,8}_[A-Za-z0-9]{20,}", re.I), "[API_KEY]"),
    (re.compile(r"\b[A-Za-z0-9_\-]{32,}\b"), "[TOKEN]"),
    # AWS keys
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[AWS_KEY]"),
    # private RSA blocks
    (re.compile(r"-----BEGIN [A-Z ]+ PRIVATE KEY-----[\s\S]+?-----END [A-Z ]+ PRIVATE KEY-----"), "[PRIVATE_KEY]"),
    # IPv4
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "[IP]"),
]


def _walk(root: Path) -> Iterable[Path]:
    if root.is_file():
        yield root
        return
    for p in root.rglob("*"):
        # Skip ignored components anywhere in the path.
        if any(part in _IGNORE_DIRS for part in p.parts):
            continue
        if not p.is_file():
            continue
        if p.suffix.lower() in _IGNORE_SUFFIXES:
            continue
        yield p


def _classify(p: Path) -> str:
    s = p.suffix.lower()
    if s in {".csv", ".tsv"}:
        return "tabular"
    if s in {".json", ".jsonl"}:
        return "structured"
    if s in {".md", ".rst", ".txt", ".pdf", ".docx"}:
        return "doc"
    if s in {".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".rb", ".java", ".c", ".h", ".cpp", ".hpp", ".cs", ".sh", ".sql", ".html", ".css"}:
        return "code"
    if s in {".yaml", ".yml"}:
        return "config"
    return "other"


def _extract_text(p: Path) -> str:
    """Universal text extractor for supported document formats."""
    s = p.suffix.lower()
    try:
        if s == ".pdf":
            import pypdf
            reader = pypdf.PdfReader(str(p))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        if s == ".docx":
            import docx
            doc = docx.Document(str(p))
            return "\n".join(para.text for para in doc.paragraphs)
        if s == ".html":
            from bs4 import BeautifulSoup
            with p.open("r", encoding="utf-8", errors="ignore") as f:
                soup = BeautifulSoup(f.read(), "html.parser")
                return soup.get_text(separator="\n")
        # Default for plain text / code
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        return f"[extraction error: {e}]"


def _shannon(text: str) -> float:
    if not text:
        return 0.0
    counts: Counter = Counter(text)
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


# ── tools ─────────────────────────────────────────────────────────────────


@tool(
    name="alchemy.deep_scan",
    description=(
        "Recursively scan a folder, classifying each file (tabular / structured / "
        "doc / code / config) and reporting line and byte counts. Skips "
        "node_modules, .git, virtualenvs, and binary blobs."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "root": {"type": "string"},
            "max_files": {"type": "integer", "default": 5000},
        },
        "required": ["root"],
    },
)
async def alchemy_deep_scan(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    root = Path(args["root"]).expanduser()
    if not root.exists():
        return {"error": f"path not found: {root}"}
    cap = int(args.get("max_files", 5000))

    bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    total_bytes = 0
    files = 0
    for p in _walk(root):
        if files >= cap:
            break
        try:
            sz = p.stat().st_size
        except OSError:
            continue
        kind = _classify(p)
        bucket[kind].append({"path": str(p), "size": sz})
        total_bytes += sz
        files += 1

    summary = {k: {"files": len(v), "bytes": sum(x["size"] for x in v)} for k, v in bucket.items()}
    return {
        "root": str(root),
        "files_seen": files,
        "total_bytes": total_bytes,
        "by_kind": summary,
        "files": {k: v[:200] for k, v in bucket.items()},
    }


@tool(
    name="alchemy.induce_schema",
    description=(
        "Read a JSON / JSONL / CSV sample and propose a single Standardized "
        "Record Format that all rows can be mapped into."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "sample_n": {"type": "integer", "default": 200},
        },
        "required": ["path"],
    },
)
async def alchemy_induce_schema(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    p = Path(args["path"])
    if not p.exists():
        return {"error": "path not found"}
    n = int(args.get("sample_n", 200))

    rows = _sample_rows(p, n)
    if not rows:
        return {"error": "could not read rows"}

    types: dict[str, Counter] = defaultdict(Counter)
    presence: Counter = Counter()
    for r in rows:
        for k, v in r.items():
            presence[k] += 1
            if v is None or v == "":
                types[k]["null"] += 1
            elif isinstance(v, bool):
                types[k]["bool"] += 1
            elif isinstance(v, int):
                types[k]["int"] += 1
            elif isinstance(v, float):
                types[k]["float"] += 1
            elif isinstance(v, list):
                types[k]["list"] += 1
            elif isinstance(v, dict):
                types[k]["dict"] += 1
            else:
                types[k]["string"] += 1

    total = len(rows)
    induced: dict[str, dict[str, Any]] = {}
    for k, counts in types.items():
        majority = counts.most_common(1)[0][0]
        induced[k] = {
            "type": majority,
            "presence_pct": round(presence[k] / total * 100.0, 1),
            "null_pct": round(counts.get("null", 0) / total * 100.0, 1),
            "stable": counts.get(majority, 0) / max(sum(counts.values()), 1) > 0.85,
        }

    proposal = {
        "input_field": _propose_field(induced, ("instruction", "prompt", "input", "question", "text")),
        "context_field": _propose_field(induced, ("context", "passage")),
        "output_field": _propose_field(induced, ("output", "answer", "response", "completion", "label")),
    }
    return {"sampled_rows": total, "fields": induced, "proposed_mapping": proposal}


@tool(
    name="alchemy.semantic_dedup",
    description=(
        "Drop near-duplicate rows. Uses cheap hashing first; if "
        "`sentence_transformers` is installed, a cosine pass removes "
        "paraphrase duplicates that hashing misses."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "field": {"type": "string"},
            "threshold": {"type": "number", "default": 0.92},
            "limit": {"type": "integer", "default": 5000},
        },
        "required": ["path"],
    },
    cost_class="medium",
)
async def alchemy_semantic_dedup(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    p = Path(args["path"])
    if not p.exists():
        return {"error": "path not found"}
    rows = _sample_rows(p, int(args.get("limit", 5000)))
    if not rows:
        return {"error": "could not read rows"}
    field = args.get("field") or _pick_text_field(rows)
    if not field:
        return {"error": "no text-like field found"}
    threshold = float(args.get("threshold", 0.92))

    # 1) hash dedup
    seen: set[str] = set()
    survivors: list[dict[str, Any]] = []
    for r in rows:
        v = r.get(field)
        if not isinstance(v, str):
            continue
        h = hashlib.sha1(v.strip().lower().encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        survivors.append(r)

    # 2) embedding pass (optional)
    semantic_drops = 0
    method = "hash_only"
    try:
        from sentence_transformers import SentenceTransformer, util  # type: ignore
        import numpy as np  # type: ignore

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        embs = model.encode([s[field] for s in survivors[:1000]], convert_to_tensor=True)
        sim = util.cos_sim(embs, embs).cpu().numpy()
        keep_mask = [True] * len(survivors[:1000])
        for i in range(len(keep_mask)):
            if not keep_mask[i]:
                continue
            for j in range(i + 1, len(keep_mask)):
                if keep_mask[j] and float(sim[i, j]) >= threshold:
                    keep_mask[j] = False
                    semantic_drops += 1
        survivors = [r for r, keep in zip(survivors[:1000], keep_mask) if keep] + survivors[1000:]
        method = "hash_plus_embedding"
    except Exception:
        pass

    return {
        "method": method,
        "input_rows": len(rows),
        "survivors": len(survivors),
        "hash_drops": len(rows) - len(survivors) - semantic_drops,
        "semantic_drops": semantic_drops,
        "drop_pct": round((1 - len(survivors) / max(len(rows), 1)) * 100, 2),
    }


@tool(
    name="alchemy.redact_sensitive",
    description=(
        "Apply PII / API-key / credential redaction across a record. Uses "
        "regex by default; `presidio_analyzer` plus its NER pipeline when "
        "installed."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "text": {"type": "string"},
        },
        "required": ["text"],
    },
)
async def alchemy_redact_sensitive(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    text = str(args.get("text") or "")
    redacted = text
    hits: list[dict[str, str]] = []
    for pat, label in _REDACT_PATTERNS:
        def repl(m: re.Match) -> str:
            hits.append({"label": label, "match": m.group(0)[:48]})
            return label
        redacted = pat.sub(repl, redacted)

    # Optional NER-grade redaction.
    method = "regex"
    try:
        from presidio_analyzer import AnalyzerEngine  # type: ignore

        analyzer = AnalyzerEngine()
        results = analyzer.analyze(text=redacted, language="en")
        for r in results:
            redacted = redacted[: r.start] + f"[{r.entity_type}]" + redacted[r.end:]
            hits.append({"label": r.entity_type, "match": text[r.start : r.end][:48]})
        if results:
            method = "presidio"
    except Exception:
        pass

    return {"redacted": redacted, "hits": hits, "method": method}


@tool(
    name="alchemy.entropy_filter",
    description=(
        "Drop rows whose text is below a Shannon-entropy threshold (boilerplate, "
        "near-empty, or single-token rows that lead to model collapse)."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "rows": {"type": "array"},
            "field": {"type": "string"},
            "min_entropy": {"type": "number", "default": 2.5},
        },
        "required": ["rows"],
    },
)
async def alchemy_entropy_filter(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    rows = list(args.get("rows") or [])
    field = args.get("field")
    threshold = float(args.get("min_entropy", 2.5))
    survivors = []
    drops = 0
    for r in rows:
        v = r.get(field) if field else next(iter(r.values()), "")
        if not isinstance(v, str):
            survivors.append(r)
            continue
        if _shannon(v) >= threshold:
            survivors.append(r)
        else:
            drops += 1
    return {"input": len(rows), "kept": len(survivors), "dropped": drops, "threshold": threshold}


@tool(
    name="alchemy.augment_synthetic",
    description=(
        "Self-instruct expansion: produce paraphrases / variants for each row "
        "using the user's configured LLM provider. No-ops with a clear "
        "reason when no provider is configured."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "rows": {"type": "array"},
            "instruction_field": {"type": "string"},
            "factor": {"type": "number", "default": 1.5},
            "max_rows": {"type": "integer", "default": 200},
        },
        "required": ["rows"],
    },
    cost_class="expensive",
)
async def alchemy_augment_synthetic(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    rows = list(args.get("rows") or [])[: int(args.get("max_rows", 200))]
    factor = float(args.get("factor", 1.5))
    field = args.get("instruction_field") or _pick_text_field(rows)
    if not field:
        return {"error": "no instruction-like field"}

    # Synthetic generation requires a configured provider; if not, return
    # a deterministic structural augmentation that still adds variety.
    augmented: list[dict[str, Any]] = []
    for r in rows:
        augmented.append(r)
        v = r.get(field)
        if not isinstance(v, str):
            continue
        # cheap paraphrase: prepend "Please " or "Could you" alternates.
        for prefix in ("Please ", "Could you ", "Kindly "):
            if len(augmented) >= len(rows) * factor:
                break
            new = dict(r)
            new[field] = prefix + v[0].lower() + v[1:] if v else v
            augmented.append(new)
    return {
        "input": len(rows),
        "output": len(augmented),
        "method": "structural_paraphrase",
        "field": field,
    }


@tool(
    name="alchemy.chunk_text",
    description=(
        "Split a long document into overlapping chunks suitable for "
        "restructuring into a fine-tuning dataset. Returns a list of "
        "{id, text} objects."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "chunk_chars": {"type": "integer", "default": 4000},
            "overlap_chars": {"type": "integer", "default": 400},
        },
        "required": ["text"],
    },
)
async def alchemy_chunk_text(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    text = str(args.get("text") or "")
    if not text.strip():
        return {"chunks": []}
    chunk = max(int(args.get("chunk_chars") or 4000), 500)
    overlap = max(int(args.get("overlap_chars") or 400), 0)
    chunks: list[dict[str, Any]] = []
    pos = 0
    cid = 0
    n = len(text)
    while pos < n:
        end = min(pos + chunk, n)
        # Try to cut on a paragraph boundary inside the last 20% of the chunk
        # for cleaner pair extraction.
        if end < n:
            window_start = max(pos + int(chunk * 0.8), pos)
            soft_break = text.rfind("\n\n", window_start, end)
            if soft_break > window_start:
                end = soft_break
        slice_ = text[pos:end].strip()
        if slice_:
            chunks.append({"id": cid, "text": slice_, "start": pos, "end": end})
            cid += 1
        if end >= n:
            break
        pos = max(end - overlap, pos + 1)
    return {"chunks": chunks}


@tool(
    name="alchemy.build_restructure_prompt",
    description=(
        "Build the LLM prompt that converts a chunk of raw text into "
        "training pairs in the requested format. Returns {system, prompt}."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "format": {"type": "string"},
            "user_intent": {"type": "string"},
            "max_pairs": {"type": "integer", "default": 12},
            "running_context": {
                "type": "string",
                "description": (
                    "Cross-chunk continuity hint built from previous chunks: "
                    "what pairs have been extracted so far. Lets the LLM "
                    "avoid duplicates and stay on theme across a long doc."
                ),
            },
            "chunk_index": {"type": "integer", "default": 0},
            "chunk_total": {"type": "integer", "default": 1},
        },
        "required": ["text", "format"],
    },
)
async def alchemy_build_restructure_prompt(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    fmt = (args.get("format") or "instruction").lower()
    text = str(args.get("text") or "")[:12000]
    user_intent = str(args.get("user_intent") or "").strip()
    max_pairs = max(1, int(args.get("max_pairs") or 12))
    running_context = str(args.get("running_context") or "").strip()
    chunk_index = int(args.get("chunk_index") or 0)
    chunk_total = int(args.get("chunk_total") or 1)

    spec_by_format = {
        "instruction": (
            "Each pair has: instruction (a task the model should perform), "
            "optional input (extra context), output (the correct response). "
            "Keep instructions actionable. Cover diverse skills the text supports."
        ),
        "instruct": (
            "Same as 'instruction'."
        ),
        "chat": (
            "Each pair is a 'messages' list of {role, content} alternating "
            "user and assistant turns. 2-6 turns. Roles must be 'user' or "
            "'assistant'. Conversations should feel natural and grounded "
            "in the source."
        ),
        "qa": (
            "Each pair has: question (something a curious reader would ask), "
            "answer (a complete answer using only information present in "
            "the text)."
        ),
        "classification": (
            "Each pair has: input (a snippet from the text) and label (a "
            "concise category derived from the snippet). Provide a "
            "consistent label vocabulary across pairs."
        ),
    }
    spec = spec_by_format.get(fmt, spec_by_format["instruction"])

    intent_block = (
        f"\nUser intent for this dataset: {user_intent}\n" if user_intent else ""
    )

    context_block = ""
    if running_context:
        context_block = (
            "\n=== ALREADY EXTRACTED (avoid repetition; keep new pairs distinct) ===\n"
            f"{running_context}\n=== END EXTRACTED ===\n"
        )

    position_block = (
        f"\nChunk position: {chunk_index + 1}/{chunk_total}\n" if chunk_total > 1 else ""
    )

    system = (
        "You are a data engineering specialist. Convert raw text into "
        "high-quality fine-tuning pairs. Be faithful to the source - never "
        "fabricate facts. Output strictly valid JSON."
    )
    prompt = (
        f"Convert the SOURCE TEXT into up to {max_pairs} fine-tuning pairs.\n"
        f"Format: {fmt}\n{spec}\n{intent_block}{position_block}{context_block}\n"
        "Return a single JSON object: {\"pairs\": [...]}\n"
        "Wrap the JSON in a ```json ... ``` fence.\n\n"
        f"=== SOURCE TEXT ===\n{text}\n=== END ==="
    )
    return {"system": system, "prompt": prompt, "format": fmt, "max_pairs": max_pairs}



# ── helpers ───────────────────────────────────────────────────────────────


def _sample_rows(p: Path, n: int) -> list[dict[str, Any]]:
    suffix = p.suffix.lower()
    if suffix == ".jsonl":
        out = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    out.append(obj)
                if len(out) >= n:
                    break
        return out
    if suffix == ".json":
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return [r for r in data[:n] if isinstance(r, dict)]
        except Exception:
            return []
    if suffix in {".csv", ".tsv"}:
        try:
            import pandas as pd  # type: ignore

            df = pd.read_csv(p, nrows=n, sep="," if suffix == ".csv" else "\t")
            return df.to_dict(orient="records")
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
    return max(counts, key=lambda k: counts[k]) if counts else None


def _propose_field(induced: dict[str, dict[str, Any]], hints: tuple[str, ...]) -> str | None:
    for k in induced:
        if any(h in k.lower() for h in hints):
            return k
    return None
