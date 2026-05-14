"""Web grounding tools: search the open web, fetch pages.

These let the AgenticLoop look up current best practices, model cards, and
recipes when the user's hardware or task is unusual. The model decides
when to call them; we keep them cheap and safe by default.

Backends:
    web_search  - DuckDuckGo (no API key) by default. Tavily if
                  TAVILY_API_KEY is set in the environment.
    web_fetch   - httpx + trafilatura. Caps output to 8000 chars after
                  main-text extraction. No JS execution.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from app.tools.registry import ToolContext, tool


log = logging.getLogger("finetune-studio.tools.web")


# ── Search ────────────────────────────────────────────────────────────────

async def _ddg_search(query: str, max_results: int) -> dict[str, Any]:
    """DuckDuckGo via the duckduckgo-search library. Pure Python, no key.

    The lib is sync; we run it in a thread.
    """
    import asyncio
    try:
        from duckduckgo_search import DDGS  # type: ignore
    except ImportError:
        return {"error": "duckduckgo-search not installed",
                "advice": "pip install duckduckgo-search"}

    def _do() -> list[dict[str, Any]]:
        results = []
        with DDGS() as ddgs:
            for hit in ddgs.text(query, max_results=max_results, region="wt-wt"):
                results.append({
                    "title": hit.get("title", ""),
                    "url": hit.get("href", ""),
                    "snippet": hit.get("body", ""),
                })
        return results

    try:
        results = await asyncio.to_thread(_do)
        return {"backend": "duckduckgo", "query": query, "results": results}
    except Exception as e:
        msg = str(e).lower()
        if "rate" in msg or "limit" in msg or "ratelimit" in msg:
            return {"error": "rate_limited", "backend": "duckduckgo",
                    "advice": "wait ~10s or set TAVILY_API_KEY for higher limits"}
        log.warning("ddg search failed: %s", e)
        return {"error": str(e), "backend": "duckduckgo"}


async def _tavily_search(query: str, max_results: int) -> dict[str, Any]:
    """Tavily API. Requires TAVILY_API_KEY."""
    import httpx
    api_key = os.environ.get("TAVILY_API_KEY", "")
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": api_key,
                    "query": query,
                    "max_results": max_results,
                    "search_depth": "basic",
                },
            )
        data = resp.json()
        if resp.status_code != 200:
            return {"error": data.get("error") or f"http {resp.status_code}",
                    "backend": "tavily"}
        results = [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("content", ""),
            }
            for r in (data.get("results") or [])
        ]
        return {"backend": "tavily", "query": query, "results": results}
    except Exception as e:
        log.warning("tavily search failed: %s", e)
        return {"error": str(e), "backend": "tavily"}


@tool(
    name="web_search",
    description="Search the web (DDG or Tavily). Use for current best practices, model cards, papers.",
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "max_results": {"type": "integer"},
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

    if os.environ.get("TAVILY_API_KEY"):
        return await _tavily_search(query, max_results)
    return await _ddg_search(query, max_results)


# ── Fetch ─────────────────────────────────────────────────────────────────

_MAX_CHARS = 8000
_USER_AGENT = "FineTuneStudio/1.0 (+https://github.com)"


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
        # trafilatura does not give us the title directly; pull it from HTML.
        import re
        m = re.search(r"<title[^>]*>([^<]+)</title>", raw_html, re.IGNORECASE)
        if m:
            title = m.group(1).strip()
    except ImportError:
        # Fallback: bs4 only.
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
