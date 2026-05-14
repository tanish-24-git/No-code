"""LLM provider implementations.

Two real engines are shipped:

    stream_anthropic   uses the official `anthropic` SDK
    stream_openai      uses the official `openai` SDK; works for OpenAI cloud
                       and any OpenAI-compatible server (Gemini's compat
                       endpoint, Groq, Grok, DeepSeek, Mistral, Together,
                       Fireworks, OpenRouter, Perplexity, Cohere's compat
                       endpoint, HF Router, Ollama, LM Studio, vLLM, ...).

The dispatcher `stream_chat` looks up the configured provider in
`registry.PROVIDERS`, picks the right engine and base URL, and runs it.
Adding a provider is therefore a one-entry change in registry.py - no
new code here.

This module exposes two layers:
    stream_chat / stream_anthropic / stream_openai
        Sync generators yielding plain text deltas. Used by the legacy
        per-agent call_llm path.
    stream_chat_async
        Async generator yielding typed chunks (text / thinking /
        tool_call / stop). Used by the AgenticLoop. Supports a tool-use
        protocol so the loop can drive the model with arbitrary registered
        tools, not just the legacy hardcoded TOOLS list.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator, Iterator, Optional

from app.agents.registry import resolve_base_url, resolve_engine
from app.agents.tools import SYSTEM_PROMPT, TOOLS, run_tool


log = logging.getLogger("finetune-studio.providers")
# Dedicated channel for diagnostic dumps when a provider rejects a
# function/tool call. Quiet by default; grep for "failed_generation" in
# docker logs after a Groq function-call validation error.
failed_gen_log = logging.getLogger("finetune-studio.providers.failed_generation")


def stream_anthropic(
    *,
    api_key: str,
    model: str,
    base_url: str,
    system: str,
    messages: list[dict[str, Any]],
    use_tools: bool = True,
) -> Iterator[str]:
    """Anthropic Messages API.

    Two paths:
        * ``use_tools=False``  - pure text stream. We do NOT register tools
          and we do NOT engage the tool-call multi-hop loop, so a model
          that wants to use built-in tools cannot derail the call.
        * ``use_tools=True``   - the legacy multi-hop tool-use path used
          by the legacy /api/agent/chat endpoint.
    """
    from anthropic import Anthropic

    client = Anthropic(api_key=api_key, base_url=base_url or None)
    history = list(messages)

    if not use_tools:
        # Pure text-stream. No tool registration, no hops, no tool_choice.
        with client.messages.stream(
            model=model,
            max_tokens=2048,
            system=system,
            messages=history,
        ) as stream:
            for chunk in stream.text_stream:
                yield chunk
        return

    tools = [
        {"name": t.name, "description": t.description, "input_schema": t.input_schema}
        for t in TOOLS
    ]

    for _hop in range(8):
        with client.messages.stream(
            model=model,
            max_tokens=2048,
            system=system,
            tools=tools,
            messages=history,
        ) as stream:
            for chunk in stream.text_stream:
                yield chunk
            final = stream.get_final_message()

        history.append({"role": "assistant", "content": final.content})
        tool_uses = [b for b in final.content if getattr(b, "type", None) == "tool_use"]
        if not tool_uses:
            return

        results: list[dict[str, Any]] = []
        for tu in tool_uses:
            content, is_err = run_tool(tu.name, tu.input or {})
            results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": content,
                "is_error": is_err,
            })
            yield f"\n\n[tool: {tu.name}]\n"
        history.append({"role": "user", "content": results})

    yield "\n\n[agent reached max tool-use hops]\n"


def stream_openai(
    *,
    api_key: str,
    model: str,
    base_url: str,
    system: str,
    messages: list[dict[str, Any]],
    use_tools: bool = True,
) -> Iterator[str]:
    """OpenAI Chat Completions API. base_url lets this provider drive
    Ollama, LM Studio, vLLM, OpenRouter, Together, Groq, etc.

    Two execution paths:

        * ``use_tools=False``  - pure text stream. We do NOT pass `tools`
          or `tool_choice`, and we DO NOT walk the tool-handling loop.
          If the underlying model emits tool calls anyway (Groq's
          ``gpt-oss-120b`` is a known offender - it has built-in
          browser_search and python tools that fire even when no tool
          list is provided), we silently drop the tool_call deltas and
          keep the text. This is what every agent's LLM call wants.

        * ``use_tools=True``   - legacy multi-hop tool-use path used by
          the legacy /api/agent/chat endpoint.
    """
    from openai import OpenAI

    client = OpenAI(api_key=api_key or "sk-not-needed", base_url=base_url or None)

    history: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for m in messages:
        if isinstance(m.get("content"), str):
            history.append({"role": m["role"], "content": m["content"]})
        else:
            history.append(m)

    if not use_tools:
        # Pure text path. Strategy:
        #   1. Streaming with no tools field. Yield text deltas; drop any
        #      tool_call deltas the provider emits anyway. If the stream
        #      raises mid-flight (e.g. Groq's "Tool choice is none, but
        #      model called a tool" on gpt-oss reasoning models), we yield
        #      whatever text we accumulated and return cleanly instead of
        #      bubbling the exception.
        #   2. If we collected zero text, fall back to a NON-streaming
        #      request with explicit ``tools=[]`` + ``tool_choice="none"``.
        #      Groq validates the response atomically when not streaming,
        #      which usually surfaces clean text or a clean error rather
        #      than the mid-stream rejection.
        collected = ""

        def _try_stream(extra_kwargs: dict[str, Any]) -> bool:
            """Yield text from a streaming call; True if any text arrived."""
            nonlocal collected
            got_any = False
            try:
                stream = client.chat.completions.create(
                    model=model,
                    messages=history,
                    stream=True,
                    **extra_kwargs,
                )
                for chunk in stream:
                    choice = chunk.choices[0] if chunk.choices else None
                    if not choice:
                        continue
                    delta = choice.delta
                    if delta and delta.content:
                        collected += delta.content
                        got_any = True
                        yield delta.content
            except Exception as e:
                err = str(e).lower()
                if "tool" in err and (
                    "choice" in err or "called a tool" in err or "called" in err
                ):
                    return  # swallow; caller will try non-streaming
                raise
            return got_any

        # Phase 1 - streaming, no tools field at all.
        for piece in _try_stream({}):
            yield piece

        # Phase 2 - if nothing arrived, try non-streaming with explicit
        # tool_choice="none". Many Groq reasoning models comply better
        # when the request is non-streaming.
        if not collected.strip():
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=history,
                    tools=[],
                    tool_choice="none",
                    stream=False,
                )
                content = ""
                if resp.choices:
                    msg = resp.choices[0].message
                    content = (msg.content or "") if msg else ""
                if content:
                    yield content
                    collected += content
            except Exception:
                # Final fallback - try once more non-streaming, no tools field.
                try:
                    resp = client.chat.completions.create(
                        model=model,
                        messages=history,
                        stream=False,
                    )
                    if resp.choices:
                        msg = resp.choices[0].message
                        content = (msg.content or "") if msg else ""
                        if content:
                            yield content
                except Exception:
                    pass
        return

    # ── Tool-use multi-hop path ───────────────────────────────────────
    tools = [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
            },
        }
        for t in TOOLS
    ]

    for _hop in range(8):
        stream = client.chat.completions.create(
            model=model,
            messages=history,
            tools=tools,
            stream=True,
        )

        assistant_text = ""
        tool_calls: dict[int, dict[str, Any]] = {}
        for chunk in stream:
            choice = chunk.choices[0] if chunk.choices else None
            if not choice:
                continue
            delta = choice.delta
            if delta and delta.content:
                assistant_text += delta.content
                yield delta.content
            if delta and delta.tool_calls:
                for tc in delta.tool_calls:
                    slot = tool_calls.setdefault(tc.index, {"id": "", "name": "", "args": ""})
                    if tc.id:
                        slot["id"] = tc.id
                    if tc.function and tc.function.name:
                        slot["name"] = tc.function.name
                    if tc.function and tc.function.arguments:
                        slot["args"] += tc.function.arguments

        if not tool_calls:
            return

        history.append({
            "role": "assistant",
            "content": assistant_text or None,
            "tool_calls": [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["args"] or "{}"},
                }
                for tc in tool_calls.values()
            ],
        })

        for tc in tool_calls.values():
            try:
                args = json.loads(tc["args"] or "{}")
            except json.JSONDecodeError:
                args = {}
            content, _is_err = run_tool(tc["name"], args)
            history.append({"role": "tool", "tool_call_id": tc["id"], "content": content})
            yield f"\n\n[tool: {tc['name']}]\n"

    yield "\n\n[agent reached max tool-use hops]\n"


def stream_chat(
    *,
    provider: str,
    api_key: str,
    model: str,
    base_url: str,
    messages: list[dict[str, Any]],
    use_tools: bool = True,
    extra_system: str = "",
) -> Iterator[str]:
    """Look up the provider in the registry, resolve engine + base URL,
    and run the right streaming function. Tools and system prompt are
    shared across engines."""
    system = SYSTEM_PROMPT if not extra_system else SYSTEM_PROMPT + "\n\n" + extra_system
    engine = resolve_engine(provider)
    resolved_base = resolve_base_url(provider, base_url or None)

    if engine == "anthropic":
        yield from stream_anthropic(
            api_key=api_key,
            model=model,
            base_url=resolved_base,
            system=system,
            messages=messages,
            use_tools=use_tools,
        )
        return
    if engine == "openai":
        yield from stream_openai(
            api_key=api_key,
            model=model,
            base_url=resolved_base,
            system=system,
            messages=messages,
            use_tools=use_tools,
        )
        return
    raise ValueError(f"Unknown engine for provider {provider!r}: {engine}")


# ══════════════════════════════════════════════════════════════════════════
# Async typed-chunk streaming (used by AgenticLoop)
# ══════════════════════════════════════════════════════════════════════════
#
# Yields dicts of shape:
#   {"kind": "text", "delta": str}          - assistant text token
#   {"kind": "thinking", "delta": str}      - extended-thinking token
#   {"kind": "tool_call", "id": str,        - one full tool call after args
#       "name": str, "args": dict}            JSON has been buffered
#   {"kind": "stop", "reason": str}         - terminal event
#   {"kind": "error", "error": str}         - non-fatal error; caller decides
#
# Anthropic native thinking is honored when the caller passes thinking=...
# OpenAI providers do not stream thinking; we surface only text + tool_calls.

async def stream_anthropic_async(
    *,
    api_key: str,
    model: str,
    base_url: str,
    system: str,
    messages: list[dict[str, Any]],
    tools: Optional[list[dict[str, Any]]] = None,
    thinking_budget: Optional[int] = None,
    max_tokens: int = 4096,
) -> AsyncIterator[dict[str, Any]]:
    """Async stream over the Anthropic Messages API with typed events."""
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic(api_key=api_key, base_url=base_url or None)
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": messages,
    }
    if tools:
        kwargs["tools"] = tools
    if thinking_budget and thinking_budget > 0:
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}

    try:
        async with client.messages.stream(**kwargs) as stream:
            # In-progress tool-use blocks: index -> {"id", "name", "args_buf"}
            tool_blocks: dict[int, dict[str, Any]] = {}
            async for event in stream:
                etype = getattr(event, "type", "")
                if etype == "content_block_start":
                    block = getattr(event, "content_block", None)
                    btype = getattr(block, "type", "")
                    if btype == "tool_use":
                        idx = getattr(event, "index", 0)
                        tool_blocks[idx] = {
                            "id": getattr(block, "id", ""),
                            "name": getattr(block, "name", ""),
                            "args_buf": "",
                        }
                elif etype == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    dtype = getattr(delta, "type", "")
                    if dtype == "text_delta":
                        yield {"kind": "text",
                               "delta": getattr(delta, "text", "")}
                    elif dtype == "thinking_delta":
                        yield {"kind": "thinking",
                               "delta": getattr(delta, "thinking", "")}
                    elif dtype == "input_json_delta":
                        idx = getattr(event, "index", 0)
                        slot = tool_blocks.get(idx)
                        if slot is not None:
                            slot["args_buf"] += getattr(delta, "partial_json", "")
                elif etype == "content_block_stop":
                    idx = getattr(event, "index", 0)
                    if idx in tool_blocks:
                        slot = tool_blocks.pop(idx)
                        try:
                            args = json.loads(slot["args_buf"] or "{}")
                        except json.JSONDecodeError:
                            args = {}
                        yield {"kind": "tool_call", "id": slot["id"],
                               "name": slot["name"], "args": args}
                elif etype == "message_stop":
                    pass
            final = await stream.get_final_message()
            yield {"kind": "stop",
                   "reason": getattr(final, "stop_reason", "end_turn") or "end_turn"}
    except Exception as e:
        log.warning("anthropic async stream failed: %s", e)
        yield {"kind": "error", "error": str(e)}
        yield {"kind": "stop", "reason": "error"}


async def stream_openai_async(
    *,
    api_key: str,
    model: str,
    base_url: str,
    system: str,
    messages: list[dict[str, Any]],
    tools: Optional[list[dict[str, Any]]] = None,
    max_tokens: int = 4096,
) -> AsyncIterator[dict[str, Any]]:
    """Async stream over the OpenAI Chat Completions API (and any compatible
    endpoint). Buffers tool-call deltas and emits each tool call atomically
    when its block stops."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key or "sk-not-needed",
                         base_url=base_url or None)

    history: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for m in messages:
        history.append(m)

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": history,
        "stream": True,
        "max_tokens": max_tokens,
    }
    if tools:
        kwargs["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema") or {"type": "object"},
                },
            }
            for t in tools
        ]

    # Hoist these out of the try block so the failed_generation diagnostic
    # below can always reference them even if the very first await throws.
    tool_calls: dict[int, dict[str, Any]] = {}
    text_buf = ""
    stop_reason = "stop"
    try:
        stream = await client.chat.completions.create(**kwargs)
        async for chunk in stream:
            choice = chunk.choices[0] if chunk.choices else None
            if not choice:
                continue
            delta = choice.delta
            if delta and delta.content:
                text_buf += delta.content
                yield {"kind": "text", "delta": delta.content}
            if delta and delta.tool_calls:
                for tc in delta.tool_calls:
                    slot = tool_calls.setdefault(tc.index, {"id": "", "name": "", "args_buf": ""})
                    if tc.id:
                        slot["id"] = tc.id
                    if tc.function and tc.function.name:
                        slot["name"] = tc.function.name
                    if tc.function and tc.function.arguments:
                        slot["args_buf"] += tc.function.arguments
            fr = getattr(choice, "finish_reason", None)
            if fr:
                stop_reason = fr

        # Flush buffered tool calls. OpenAI does not have an explicit
        # per-call stop event in the delta stream; we treat finish_reason
        # as the boundary.
        for slot in tool_calls.values():
            try:
                args = json.loads(slot["args_buf"] or "{}")
            except json.JSONDecodeError:
                args = {}
            yield {"kind": "tool_call", "id": slot["id"],
                   "name": slot["name"], "args": args}

        # If the model was asked for tool use but emitted a JSON tool blob
        # as text (weaker models do this), synthesize a tool call.
        if tools and not tool_calls and text_buf.strip():
            blob = _try_extract_text_tool_call(text_buf)
            if blob is not None:
                yield {"kind": "tool_call", "id": blob.get("id", "synth-0"),
                       "name": blob["name"], "args": blob.get("args") or {}}

        yield {"kind": "stop", "reason": stop_reason}
    except Exception as e:
        err_text = str(e)
        log.warning("openai async stream failed: %s", err_text)
        # When the provider rejects the model's tool call (Groq's
        # llama-3.3-70b is the known offender), dump what we had buffered
        # so the operator can see what the model emitted.
        lowered = err_text.lower()
        if ("function call" in lowered
                or "function_call" in lowered
                or "failed_generation" in lowered
                or "tool call" in lowered):
            pending_tools = [
                {"name": s.get("name"), "args_raw": (s.get("args_buf") or "")[:500]}
                for s in tool_calls.values()
            ]
            last_user_content = ""
            if history:
                last = history[-1].get("content")
                if isinstance(last, str):
                    last_user_content = last[:500]
            failed_gen_log.warning(
                "model=%s rejected_tool_call err=%r assistant_text=%r pending_tools=%r last_user=%r",
                model, err_text, (text_buf or "")[:500], pending_tools, last_user_content,
            )
        yield {"kind": "error", "error": err_text}
        yield {"kind": "stop", "reason": "error"}


def _try_extract_text_tool_call(text: str) -> Optional[dict[str, Any]]:
    """If the model emitted ``{"tool": "...", "args": {...}}`` as text
    instead of via the structured tool-call channel, parse and return it.
    Helps with weaker local models that ignore the tool_use channel."""
    import re
    # Look for a fenced or bare JSON object that has both "tool" (or "name")
    # and an "args" / "input" field.
    candidates: list[str] = []
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates.extend(fenced)
    # Bare top-level object as a fallback.
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        candidates.append(stripped)
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        name = obj.get("tool") or obj.get("name")
        args = obj.get("args") or obj.get("input") or obj.get("arguments") or {}
        if isinstance(name, str) and name:
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            if isinstance(args, dict):
                return {"id": obj.get("id", "synth-0"), "name": name, "args": args}
    return None


async def stream_chat_async(
    *,
    provider: str,
    api_key: str,
    model: str,
    base_url: str,
    system: str,
    messages: list[dict[str, Any]],
    tools: Optional[list[dict[str, Any]]] = None,
    thinking_budget: Optional[int] = None,
    max_tokens: int = 4096,
) -> AsyncIterator[dict[str, Any]]:
    """Unified async streaming with typed chunks.

    Unlike the sync stream_chat above, this does NOT prepend the legacy
    SYSTEM_PROMPT - the caller (AgenticLoop) owns the full system prompt
    because tool taxonomy differs.
    """
    engine = resolve_engine(provider)
    resolved_base = resolve_base_url(provider, base_url or None)

    if engine == "anthropic":
        async for chunk in stream_anthropic_async(
            api_key=api_key,
            model=model,
            base_url=resolved_base,
            system=system,
            messages=messages,
            tools=tools,
            thinking_budget=thinking_budget,
            max_tokens=max_tokens,
        ):
            yield chunk
        return
    if engine == "openai":
        async for chunk in stream_openai_async(
            api_key=api_key,
            model=model,
            base_url=resolved_base,
            system=system,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
        ):
            yield chunk
        return
    yield {"kind": "error", "error": f"Unknown engine for provider {provider!r}: {engine}"}
    yield {"kind": "stop", "reason": "error"}
