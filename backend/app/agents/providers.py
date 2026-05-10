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
Adding a provider is therefore a one-entry change in registry.py — no
new code here.
"""
from __future__ import annotations

import json
from typing import Any, Iterator

from app.agents.registry import resolve_base_url, resolve_engine
from app.agents.tools import SYSTEM_PROMPT, TOOLS, run_tool


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
