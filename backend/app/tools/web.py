
"""Web grounding tools: search the open web, fetch pages.

User-facing contract: zero configuration. The only things the operator
sets are LLM_PROVIDER + LLM_API_KEY + HF_TOKEN. Web search Just Works
out of a chain of free, no-API-key backends:

    1. DuckDuckGo            via the duckduckgo-search library
    2. Google                via the googlesearch-python library (scrape)
    3. SearXNG public pool   via httpx JSON queries (rotating instances)
    4. Wikipedia opensearch  via httpx (narrow, but always available)

Each backend has a short timeout. The chain attempts them in order until
one returns at least one usable result. None of them require the user to
sign up anywhere or paste a key.

If all four fall over (rare - Wikipedia rarely fails), we return a bland
"search backend busy, try again on the next turn" message. The model
sees this and either retries with a different query, asks the user for
clarification, or proceeds without web context.

`web_fetch` reads a single URL and extracts its main text via
trafilatura with a bs4 fallback. No JS execution; capped at 8000 chars.
"""
from __future__ import annotations

import asyncio
import logging
import os
import random
import re
import time
from typing import Any, Awaitable, Callable

from app.tools.registry import ToolContext, tool


log = logging.getLogger("finetune-studio.tools.web")


# In-process TTL cache so multi-agent workflows don't re-hit the chain on
# identical queries. Cleared by process restart.
_SEARCH_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_SEARCH_CACHE_TTL = 600.0  # 10 min


def _cache_get(key: str) -> dict[str, Any] | None:
    entry = _SEARCH_CACHE.get(key)
    if not entry:
        return None
    ts, payload = entry
    if time.time() - ts > _SEARCH_CACHE_TTL:
        _SEARCH_CACHE.pop(key, None)
        return None
    return payload


def _cache_put(key: str, payload: dict[str, Any]) -> None:
    _SEARCH_CACHE[key] = (time.time(), payload)
    if len(_SEARCH_CACHE) > 256:
        # Cheap LRU-ish eviction: drop the oldest entry.
        oldest = min(_SEARCH_CACHE.items(), key=lambda kv: kv[1][0])
        _SEARCH_CACHE.pop(oldest[0], None)


_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64; rv:122.0) Gecko/20100101 Firefox/122.0"
)
_BACKEND_TIMEOUT = 8.0
_BUSY_RESPONSE: dict[str, Any] = {
    "error": "search_backend_unavailable",
    "advice": "the web search backend is busy; will try again on the next turn",
}


# ── Backend -1: Gemini native grounding (opt-in via LLM_PROVIDER=gemini) ──
#
# When the user is on a Gemini paid key that supports Vertex AI / Google
# Search grounding, this returns crisp citations with URLs. Free-tier
# Gemini doesn't support grounding; the call fails fast and the chain
# moves on to DDG/etc.

def _resolve_gemini_credentials() -> tuple[str, str]:
    """Return (api_key, model_name) if the user is on Gemini, else ('', '')."""
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    model_name = ""
    try:
        from app.api.routes.settings import get_llm_config
        cfg = get_llm_config()
        if (cfg.provider or "").lower() == "gemini":
            api_key = api_key or cfg.api_key
            model_name = cfg.model or model_name
    except Exception:
        pass
    if not api_key:
        prov = (os.environ.get("LLM_PROVIDER") or "").lower()
        if "gemini" in prov:
            api_key = os.environ.get("LLM_API_KEY", "").strip()
    if not model_name:
        model_name = os.environ.get("GEMINI_GROUNDING_MODEL", "gemini-2.0-flash")
    return api_key, model_name


async def _gemini_grounding(query: str, max_results: int) -> dict[str, Any]:
    """Use Gemini's built-in Google Search grounding tool when available.

    Best-effort: returns ``{"error": "not_configured"}`` when the user
    isn't on Gemini, the SDK isn't installed, the key lacks grounding
    permission, or the API call fails. The chain falls through to the
    next backend in any of those cases — grounding is an upgrade, not a
    requirement.
    """
    api_key, model_name = _resolve_gemini_credentials()
    if not api_key:
        return {"error": "not_configured"}
    # Allow opt-out without uninstalling the SDK.
    if os.environ.get("GEMINI_GROUNDING_DISABLED", "").lower() in ("1", "true", "yes"):
        return {"error": "disabled"}

    def _do() -> dict[str, Any]:
        # Modern google-genai SDK (preferred) first; fall back to legacy.
        try:
            from google import genai as _genai_new  # type: ignore
            from google.genai import types as _genai_types  # type: ignore
            client = _genai_new.Client(api_key=api_key)
            tool_obj = _genai_types.Tool(google_search=_genai_types.GoogleSearch())
            cfg = _genai_types.GenerateContentConfig(tools=[tool_obj])
            resp = client.models.generate_content(
                model=model_name,
                contents=(
                    "Use Google Search to find authoritative pages that answer "
                    f"this query. Return concise URLs + titles. Query: {query}"
                ),
                config=cfg,
            )
            text = getattr(resp, "text", "") or ""
            citations: list[dict[str, str]] = []
            for cand in getattr(resp, "candidates", []) or []:
                cm = getattr(cand, "grounding_metadata", None)
                if not cm:
                    continue
                for ch in getattr(cm, "grounding_chunks", []) or []:
                    w = getattr(ch, "web", None)
                    if w and getattr(w, "uri", None):
                        citations.append({
                            "title": str(getattr(w, "title", "") or ""),
                            "url": str(w.uri),
                            "snippet": text[:240],
                        })
            if citations:
                return {"backend": "gemini_grounding", "results": citations[:max_results]}
            return {"error": "gemini_empty"}
        except ImportError:
            pass
        except Exception as e:
            return {"error": "gemini_failed", "_detail": str(e)}
        # Legacy google-generativeai fallback.
        try:
            import google.generativeai as _genai_legacy  # type: ignore
        except ImportError:
            return {"error": "missing_lib"}
        try:
            _genai_legacy.configure(api_key=api_key)
            model_obj = _genai_legacy.GenerativeModel(model_name)
            response = model_obj.generate_content(
                f"Search and list source URLs for: {query}",
                tools=[{"google_search_retrieval": {}}],
            )
            text = getattr(response, "text", "") or ""
            urls = re.findall(r"https?://[^\s)\]]+", text)[:max_results]
            results = [
                {"title": "", "url": u, "snippet": text[:240]} for u in urls
            ]
            if results:
                return {"backend": "gemini_grounding", "results": results}
            return {"error": "gemini_empty"}
        except Exception as e:
            return {"error": "gemini_failed", "_detail": str(e)}

    try:
        return await asyncio.to_thread(_do)
    except Exception as e:
        log.info("gemini grounding crashed: %s", e)
        return {"error": "gemini_crashed", "_detail": str(e)}


# ── Backend 0: Tavily (opt-in via TAVILY_API_KEY) ─────────────────────────
#
# Tavily returns cleaner markdown than scraped backends and survives
# bot-detection that breaks DDG/Google when the server runs in a data
# center. Skipped entirely when no key is set, so the zero-config
# default chain below still works for users who do not opt in.

async def _tavily_search(query: str, max_results: int) -> dict[str, Any]:
    key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not key:
        return {"error": "not_configured"}
    import httpx
    try:
        async with httpx.AsyncClient(timeout=_BACKEND_TIMEOUT,
                                     headers={"User-Agent": _USER_AGENT}) as client:
            resp = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": key,
                    "query": query,
                    "max_results": max_results,
                    "search_depth": "basic",
                },
            )
        if resp.status_code >= 400:
            return {"error": f"tavily_http_{resp.status_code}"}
        data = resp.json() or {}
        hits = data.get("results") or []
        out = [
            {
                "title": h.get("title", "") or "",
                "url": h.get("url", "") or "",
                "snippet": h.get("content", "") or "",
            }
            for h in hits[:max_results]
            if h.get("url")
        ]
        if out:
            return {"backend": "tavily", "results": out}
        return {"error": "tavily_empty"}
    except Exception as e:
        log.info("tavily search failed: %s", e)
        return {"error": "tavily_failed", "_detail": str(e)}


# ── Backend 1: DuckDuckGo ─────────────────────────────────────────────────

async def _ddg_search(query: str, max_results: int) -> dict[str, Any]:
    try:
        from duckduckgo_search import DDGS  # type: ignore
    except ImportError:
        return {"error": "missing_lib"}

    def _do() -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        with DDGS() as ddgs:
            for hit in ddgs.text(query, max_results=max_results, region="wt-wt"):
                out.append({
                    "title": hit.get("title", ""),
                    "url": hit.get("href", ""),
                    "snippet": hit.get("body", ""),
                })
        return out

    backoffs = (0.0, 1.0)  # one quick retry
    last_err: Exception | None = None
    for delay in backoffs:
        if delay:
            await asyncio.sleep(delay)
        try:
            results = await asyncio.to_thread(_do)
            if results:
                return {"backend": "duckduckgo", "results": results}
            last_err = RuntimeError("empty results")
        except Exception as e:
            last_err = e
            log.info("ddg attempt failed: %s", e)
    return {"error": "ddg_failed", "_detail": str(last_err) if last_err else ""}


# ── Backend 2: Google (direct scrape via googlesearch-python) ─────────────

async def _googlesearch_search(query: str, max_results: int) -> dict[str, Any]:
    try:
        from googlesearch import search as gs_search  # type: ignore
    except ImportError:
        return {"error": "missing_lib"}

    def _do() -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        # advanced=True returns SearchResult objects with title/url/description
        try:
            iterator = gs_search(query, num_results=max_results, advanced=True,
                                 sleep_interval=1)
        except TypeError:
            # Older versions don't accept sleep_interval
            iterator = gs_search(query, num_results=max_results, advanced=True)
        for hit in iterator:
            title = getattr(hit, "title", "") or ""
            url = getattr(hit, "url", "") or ""
            desc = getattr(hit, "description", "") or ""
            if url:
                out.append({"title": title, "url": url, "snippet": desc})
        return out

    try:
        results = await asyncio.wait_for(asyncio.to_thread(_do), timeout=_BACKEND_TIMEOUT * 2)
        if results:
            return {"backend": "google", "results": results}
        return {"error": "google_empty"}
    except asyncio.TimeoutError:
        return {"error": "google_timeout"}
    except Exception as e:
        log.info("googlesearch attempt failed: %s", e)
        return {"error": "google_failed", "_detail": str(e)}


# ── Backend 3: SearXNG public instances ───────────────────────────────────
#
# Public SearXNG hosts that historically allow JSON responses. We shuffle
# the list per query for crude load distribution. Operators can override
# via the SEARXNG_INSTANCES env var (comma-separated URLs).

_SEARXNG_DEFAULTS = (
    "https://searx.be",
    "https://searx.tiekoetter.com",
    "https://search.bus-hit.me",
    "https://search.hbubli.cc",
    "https://searx.work",
    "https://opnxng.com",
    "https://search.inetol.net",
    "https://searx.foss.wtf",
)


def _searxng_instances() -> list[str]:
    import os
    raw = os.environ.get("SEARXNG_INSTANCES", "").strip()
    if raw:
        return [u.strip().rstrip("/") for u in raw.split(",") if u.strip()]
    return [u.rstrip("/") for u in _SEARXNG_DEFAULTS]


async def _searxng_search(query: str, max_results: int) -> dict[str, Any]:
    import httpx
    instances = _searxng_instances()
    random.shuffle(instances)
    headers = {"User-Agent": _USER_AGENT, "Accept": "application/json"}
    params = {
        "q": query,
        "format": "json",
        "categories": "general",
        "language": "en",
        "safesearch": "0",
    }
    async with httpx.AsyncClient(timeout=_BACKEND_TIMEOUT,
                                 headers=headers,
                                 follow_redirects=True) as client:
        for inst in instances[:4]:  # try up to 4
            try:
                resp = await client.get(f"{inst}/search", params=params)
            except Exception as e:
                log.debug("searxng %s unreachable: %s", inst, e)
                continue
            if resp.status_code >= 400:
                log.debug("searxng %s -> %d", inst, resp.status_code)
                continue
            try:
                data = resp.json()
            except Exception:
                continue
            hits = data.get("results") or []
            out = [
                {
                    "title": h.get("title", "") or "",
                    "url": h.get("url", "") or "",
                    "snippet": h.get("content", "") or "",
                }
                for h in hits[:max_results]
                if h.get("url")
            ]
            if out:
                return {"backend": "searxng", "instance": inst, "results": out}
    return {"error": "searxng_empty"}


# ── Backend 4: Wikipedia opensearch (always-up fallback) ──────────────────

async def _wikipedia_search(query: str, max_results: int) -> dict[str, Any]:
    import httpx
    headers = {"User-Agent": _USER_AGENT}
    params = {
        "action": "opensearch",
        "search": query,
        "limit": max_results,
        "namespace": 0,
        "format": "json",
    }
    try:
        async with httpx.AsyncClient(timeout=_BACKEND_TIMEOUT,
                                     headers=headers,
                                     follow_redirects=True) as client:
            resp = await client.get("https://en.wikipedia.org/w/api.php", params=params)
        if resp.status_code >= 400:
            return {"error": f"wikipedia_http_{resp.status_code}"}
        data = resp.json()
        # opensearch returns [query, [titles], [descriptions], [urls]]
        if not isinstance(data, list) or len(data) < 4:
            return {"error": "wikipedia_bad_payload"}
        titles, descs, urls = data[1], data[2], data[3]
        out = []
        for t, d, u in zip(titles, descs, urls):
            if u:
                out.append({"title": t or "", "url": u, "snippet": d or ""})
        if out:
            return {"backend": "wikipedia", "results": out}
        return {"error": "wikipedia_empty"}
    except Exception as e:
        log.info("wikipedia search failed: %s", e)
        return {"error": "wikipedia_failed", "_detail": str(e)}


# ── The chain ─────────────────────────────────────────────────────────────

BackendFn = Callable[[str, int], Awaitable[dict[str, Any]]]
_BACKENDS: tuple[BackendFn, ...] = (
    _gemini_grounding,        # opt-in; silent no-op unless LLM_PROVIDER=gemini
    _tavily_search,           # opt-in; silent no-op when TAVILY_API_KEY is unset
    _ddg_search,
    _googlesearch_search,
    _searxng_search,
    _wikipedia_search,
)


@tool(
    name="web_search",
    description="Search the open web for current best practices, papers, model cards. Use freely when uncertain about recipes.",
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            # Optional fields accept null because some providers (Groq's
            # llama-3.3-70b) send null instead of omitting the key.
            "max_results": {"type": ["integer", "null"]},
        },
        "required": ["query"],
    },
    side_effect="external",
    cost_class="cheap",
)
async def web_search(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    query = (args.get("query") or "").strip()
    if not query:
        return {"error": "query is required"}
    max_results = int(args.get("max_results") or 5)
    max_results = max(1, min(10, max_results))

    # In-session TTL cache: cuts redundant traffic when several agents
    # ask the same question (e.g. strategy and intake both web-search
    # "Llama-3 fine-tuning best practices" within seconds of each other).
    cache_key = f"{max_results}|{query.lower()}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return {**cached, "cached": True}

    last_internal_error = ""
    for backend in _BACKENDS:
        result = await backend(query, max_results)
        if result.get("results"):
            payload = {
                "query": query,
                "backend": result.get("backend", "unknown"),
                "results": result["results"],
            }
            _cache_put(cache_key, payload)
            return payload
        last_internal_error = result.get("_detail") or result.get("error") or last_internal_error
        log.debug("backend %s did not return results: %s",
                  backend.__name__, result.get("error"))

    log.warning("all web_search backends returned empty for query=%r last=%s",
                query, last_internal_error)
    return dict(_BUSY_RESPONSE)


# ── web_fetch (unchanged behavior, kept here for cohesion) ────────────────

_MAX_CHARS = 8000


@tool(
    name="web_fetch",
    description="Fetch a URL and return its main text (max 8000 chars). Use after web_search.",
    input_schema={
        "type": "object",
        "properties": {"url": {"type": "string"}},
        "required": ["url"],
    },
    side_effect="external",
    cost_class="cheap",
    # The body already caps the extracted text at 8000 chars and reports
    # ``truncated``; we don't want the registry middleware to chop again.
    supports_full_output=True,
)
async def web_fetch(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    import httpx
    url = (args.get("url") or "").strip()
    if not url.startswith("http://") and not url.startswith("https://"):
        return {"error": "url must be absolute http(s)"}

    try:
        async with httpx.AsyncClient(
            timeout=15,
            follow_redirects=True,
            max_redirects=5,
            headers={"User-Agent": _USER_AGENT},
        ) as client:
            resp = await client.get(url)
    except Exception as e:
        return {"error": f"fetch failed: {e}"}

    if resp.status_code >= 400:
        return {"error": f"http {resp.status_code}", "url": str(resp.url)}

    raw_html = resp.text
    title = ""
    text = ""
    try:
        import trafilatura  # type: ignore
        text = trafilatura.extract(raw_html, include_comments=False,
                                   include_tables=True) or ""
        m = re.search(r"<title[^>]*>([^<]+)</title>", raw_html, re.IGNORECASE)
        if m:
            title = m.group(1).strip()
    except ImportError:
        try:
            from bs4 import BeautifulSoup  # type: ignore
            soup = BeautifulSoup(raw_html, "html.parser")
            for tag in soup(["script", "style", "noscript", "nav", "header",
                             "footer", "aside"]):
                tag.decompose()
            title = (soup.title.string.strip() if soup.title and soup.title.string else "")
            text = soup.get_text(separator="\n")
            text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
        except Exception as e:
            return {"error": f"text extraction failed: {e}"}

    if not text.strip():
        return {"error": "no extractable text", "url": str(resp.url),
                "title": title}

    truncated = len(text) > _MAX_CHARS
    return {
        "url": str(resp.url),
        "title": title,
        "text": text[:_MAX_CHARS],
        "truncated": truncated,
        "length": len(text),
    }
