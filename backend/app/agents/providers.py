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
    """Anthropic Messages API with tool use."""
    from anthropic import Anthropic

    client = Anthropic(api_key=api_key, base_url=base_url or None)
    tools = [
        {"name": t.name, "description": t.description, "input_schema": t.input_schema}
        for t in TOOLS
    ] if use_tools else []
    history = list(messages)

    for _hop in range(8):
        # We need to filter tool-result messages to ensure they follow tool-use messages.
        # Anthropic is very strict about this.
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
    """OpenAI Chat Completions API with tool use. base_url lets this provider
    drive Ollama, LM Studio, vLLM, OpenRouter, Together, Groq, etc."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key or "sk-not-needed", base_url=base_url or None)
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
    ] if use_tools else []

    history: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for m in messages:
        if isinstance(m.get("content"), str):
            history.append({"role": m["role"], "content": m["content"]})
        else:
            history.append(m)

    for _hop in range(8):
        stream = client.chat.completions.create(
            model=model,
            messages=history,
            tools=tools if tools else None,
            stream=True,
        )

        # Reassemble streamed deltas. Tool call deltas arrive piecemeal too.
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

        # Append assistant turn as one message including the tool_calls.
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

        # Execute each tool, append a tool message per call.
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
