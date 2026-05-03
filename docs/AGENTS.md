# The agent

FineTune Studio's agent is a single LLM with seven server-side tools.
Its job is to read what the user actually has (hardware, datasets,
pipelines, inference endpoints) and produce concrete recommendations
that get written back to the database.

The agent is provider-agnostic. Pick from 17 providers in the setup
wizard or `backend/.env`; the agent code never branches on provider
name.

## File layout

```
backend/app/agents/
  agent.py        # public entry: run_chat(messages, ...)
  providers.py    # stream_anthropic, stream_openai, stream_chat dispatcher
  registry.py     # PROVIDERS dict: name -> engine, base_url, models, notes
  tools.py        # tool registry + impls + system prompt
```

## Provider registry

Every supported provider is one entry in `registry.py`:

```python
"groq": ProviderSpec(
    name="groq",
    engine="openai",
    label="Groq",
    base_url="https://api.groq.com/openai/v1",
    needs_key=True,
    sample_models=("llama-3.3-70b-versatile", "mixtral-8x7b-32768"),
),
```

The dispatcher `providers.stream_chat` looks up the engine, resolves
the base URL (UI override beats default), and runs the right streaming
function. Adding a provider is therefore a one-entry change in
`registry.py` plus a backend restart. The frontend pulls the new
provider through `GET /api/settings/providers` automatically; no
TypeScript changes needed.

## Two engines

### `stream_anthropic`
Uses the official `anthropic` SDK. Tool schemas are passed verbatim
because Anthropic's `input_schema` field already matches our internal
`Tool.input_schema` shape.

```python
with client.messages.stream(model=model, system=system, tools=tools, messages=history) as s:
    for chunk in s.text_stream:
        yield chunk
```

After the stream completes, we inspect `final.content` for `tool_use`
blocks. Each one becomes a server-side tool call; the result is appended
as a `tool_result` content block on a new user turn.

### `stream_openai`
Uses the official `openai` SDK with `stream=True`. Because the same SDK
talks to OpenAI cloud, Gemini's compat endpoint, Groq, Grok, DeepSeek,
Mistral, Together, Fireworks, OpenRouter, Perplexity, Cohere's compat
endpoint, HF Router, Ollama, LM Studio, vLLM, and any other server that
exposes `/v1/chat/completions`, this single provider doubles as our
local-server adapter.

Tool schemas wrap our internal shape:

```python
tools = [{
    "type": "function",
    "function": { "name": t.name, "description": t.description, "parameters": t.input_schema },
}]
```

OpenAI streams tool-call deltas piecemeal: a single `tool_calls[i]`
might be split across many chunks. We accumulate them into `tool_calls[i]`
and only execute once the stream ends.

## Supported providers

Source of truth: `backend/app/agents/registry.py`. As of writing:

```
anthropic    openai       gemini       groq         grok
deepseek     mistral      together     fireworks    openrouter
perplexity   cohere       huggingface  ollama       lmstudio
vllm         custom
```

See [docs/CONFIGURATION.md](CONFIGURATION.md) for default base URLs and
sample model ids.

## Tools

Each tool is a thin Python function with a JSON schema. The agent calls
them through the LLM's tool-use mechanism.

| Tool | What it does |
| --- | --- |
| `list_inferences` | List user's registered inference endpoints. |
| `get_inference` | Full details + last reachability probe for one endpoint. |
| `get_hardware` | Detect CPU/GPU/MPS, VRAM, CUDA version. |
| `get_dataset` | Schema, sample rows, and stats for a dataset. |
| `list_models` | Local registry (pulled base models, trained outputs). |
| `suggest_pipeline_config` | **Write a config patch back to a pipeline.** Records per-field reasoning. |
| `suggest_inference_metrics` | **Write generation metrics back to an inference endpoint.** Records reasoning. |

The two `suggest_*` tools are how the agent makes changes visible in
the UI without the user having to copy-paste anything.

### Tool example

User says:
> Look at my Ollama endpoint and tell me what generation metrics to set
> for short-form QA.

The agent typically does:

1. `list_inferences` -> sees the Ollama record id.
2. `get_inference` with that id -> gets base_url, default model,
   reachability.
3. `get_hardware` -> sees the user has a 12GB RTX 3060.
4. `suggest_inference_metrics` with `metrics: { max_tokens: 256,
   temperature: 0.2, top_p: 0.9, num_ctx: 4096, stop: ["\n\n", "Q:"] }`
   plus per-key reasoning.

The endpoint card on the Inference page now shows the metrics block
without any further user action.

## System prompt

The system prompt (in `tools.py`) tells the agent:

- It is FineTune Studio's pipeline + inference copilot.
- Use tools liberally. Always inspect what the user has before
  recommending.
- When suggesting config, justify each non-default value briefly.
- If a pipeline is active, write back via `suggest_pipeline_config`.
- Prefer short tables to walls of prose.

Per-request context (active pipeline id, focused inference id,
referenced dataset id) is appended after the static system prompt so
the agent does not have to ask "which one?" every turn.

## Streaming format

The chat endpoint emits SSE frames:

```
data: <text chunk>

data: <text chunk>

data: [tool: list_inferences]

data: <text chunk>

...

data: [DONE]
```

Tool markers are inserted by the provider after each tool call so the
UI can render them as small grey badges if it wants. The frontend
currently appends them as plain text into the assistant bubble.

## Error handling

- **Missing config.** If `LLM_PROVIDER` or `LLM_MODEL` is not set,
  `run_chat` raises immediately with a message pointing the user at
  `backend/.env` or Settings.
- **Missing key for a cloud provider.** The check `needs_key` from the
  registry catches it and reports which provider's key is missing.
- **Bad key.** Provider raises an HTTP error. The endpoint catches it,
  yields an `[error: ...]` line, and closes the stream cleanly.
- **Tool exception.** Caught inside `tools.run_tool`, returned as
  `{"error": "..."}` so the model can react ("the tool says X is
  missing - I will skip that step") instead of crashing.

## Limits

- 8 tool-use hops per request.
- 2048 max output tokens per turn.

Both are constants at the top of `providers.py`.
