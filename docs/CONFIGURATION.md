# Configuration

Configuration is layered:

1. Defaults baked into the Python code.
2. Values from `backend/.env` (loaded on backend startup).
3. Values saved through the Settings page in the UI (stored in
   `data/settings.json`).

The UI always wins. Clearing a UI override falls back to the env value;
clearing both falls back to defaults.

## LLM provider variables

These four env vars determine the agent's brain.

| Variable | Required | Description |
| --- | --- | --- |
| `LLM_PROVIDER` | yes | One of the 17 supported provider names (see table below). |
| `LLM_API_KEY` | depends | Required for cloud providers; not needed for local servers (`ollama`, `lmstudio`, `vllm`, `custom`). |
| `LLM_MODEL` | yes | Model id for the chosen provider. Free text. |
| `LLM_BASE_URL` | no | Override the API base URL. Each named provider already has a sensible default; only set this for `custom` or when you proxy a provider through your own gateway. |

## Provider matrix

The full list is loaded from the backend at runtime via
`GET /api/settings/providers`. Source of truth:
[`backend/app/agents/registry.py`](../backend/app/agents/registry.py).

| `LLM_PROVIDER` | Engine | Default `LLM_BASE_URL` | API key |
| --- | --- | --- | --- |
| `anthropic` | Anthropic SDK | `https://api.anthropic.com` | required |
| `openai` | OpenAI SDK | `https://api.openai.com/v1` | required |
| `gemini` | OpenAI SDK | `https://generativelanguage.googleapis.com/v1beta/openai` | required |
| `groq` | OpenAI SDK | `https://api.groq.com/openai/v1` | required |
| `grok` | OpenAI SDK | `https://api.x.ai/v1` | required |
| `deepseek` | OpenAI SDK | `https://api.deepseek.com/v1` | required |
| `mistral` | OpenAI SDK | `https://api.mistral.ai/v1` | required |
| `together` | OpenAI SDK | `https://api.together.xyz/v1` | required |
| `fireworks` | OpenAI SDK | `https://api.fireworks.ai/inference/v1` | required |
| `openrouter` | OpenAI SDK | `https://openrouter.ai/api/v1` | required |
| `perplexity` | OpenAI SDK | `https://api.perplexity.ai` | required |
| `cohere` | OpenAI SDK | `https://api.cohere.ai/compatibility/v1` | required |
| `huggingface` | OpenAI SDK | `https://router.huggingface.co/v1` | required (HF token) |
| `ollama` | OpenAI SDK | `http://localhost:11434/v1` | not needed |
| `lmstudio` | OpenAI SDK | `http://localhost:1234/v1` | not needed |
| `vllm` | OpenAI SDK | `http://localhost:8000/v1` | not needed |
| `custom` | OpenAI SDK | (must set `LLM_BASE_URL`) | optional |

Two engines, fifteen aliases plus `custom`. The OpenAI engine speaks
the OpenAI chat-completions wire format with tool calling; every entry
above except `anthropic` works through it.

### Model id suggestions

Pick anything your provider serves. Suggestions appear in the UI as
chips on the setup wizard.

| Provider | Examples |
| --- | --- |
| `anthropic` | `claude-sonnet-4-5`, `claude-opus-4-5`, `claude-haiku-4-5` |
| `openai` | `gpt-4o`, `gpt-4o-mini`, `gpt-4.1` |
| `gemini` | `gemini-2.5-flash`, `gemini-2.0-flash`, `gemini-1.5-pro` |
| `groq` | `llama-3.3-70b-versatile`, `mixtral-8x7b-32768`, `gemma2-9b-it` |
| `grok` | `grok-2-latest`, `grok-2-mini` |
| `deepseek` | `deepseek-chat`, `deepseek-reasoner` |
| `mistral` | `mistral-large-latest`, `mistral-small-latest` |
| `together` | `meta-llama/Llama-3.3-70B-Instruct-Turbo` |
| `fireworks` | `accounts/fireworks/models/llama-v3p1-70b-instruct` |
| `openrouter` | `anthropic/claude-3.5-sonnet`, `openai/gpt-4o` |
| `perplexity` | `sonar`, `sonar-pro`, `sonar-reasoning` |
| `cohere` | `command-r-plus`, `command-r` |
| `huggingface` | `meta-llama/Llama-3.1-8B-Instruct` |
| `ollama` | `llama3.1:8b`, `qwen2.5:7b`, `mistral:7b`, `gemma2:9b` |
| `lmstudio` | (whatever you have loaded in LM Studio) |
| `vllm` | (whatever your vLLM server serves) |

### Adding a new provider

Edit `backend/app/agents/registry.py` and add an entry to `PROVIDERS`.
The frontend dropdown updates automatically because it reads the
registry over HTTP. Most new providers need only one entry; pick
`engine="openai"` and supply the right `base_url`.

### How resolution works

```
if UI has provider, model, and (key or provider doesn't need a key):
    return UI values, source = "ui"
if env has provider and model and (key or provider doesn't need a key):
    return env values, source = "env"
return unset, source = "unset"
```

The `/health` endpoint and Settings page both display the source so it
is always clear which layer is in effect.

## Hugging Face token

| Variable | Required | Description |
| --- | --- | --- |
| `HF_TOKEN` | no | Read token for pulling base models, read+write for pushing adapters. |

The token is needed only for `/api/models/pull` and
`/api/models/{id}/push-hub`. The agent does not need it.

## Server settings

| Variable | Default | Description |
| --- | --- | --- |
| `API_HOST` | `0.0.0.0` | Bind address for uvicorn. |
| `API_PORT` | `8000` | Backend port. |
| `DEBUG` | `true` | Enables uvicorn `--reload` when running `app.main` directly. |
| `LOG_LEVEL` | `INFO` | One of `DEBUG`, `INFO`, `WARNING`, `ERROR`. |

## On-disk paths

| Variable | Default | Description |
| --- | --- | --- |
| `DATA_DIR` | `./data` | JSON state (settings, pipelines, jobs, datasets, inferences, models). |
| `UPLOAD_DIR` | `./uploads` | Raw dataset files. |
| `MODELS_DIR` | `./models` | Pulled base models and trained outputs. |

All three directories are created on first run. Paths are resolved
relative to the directory you run uvicorn from (typically `backend/`).

## Security

| Variable | Default | Description |
| --- | --- | --- |
| `ENCRYPTION_KEY` | auto-generated | Fernet key used to encrypt API keys saved through the UI. Persisted under `data/.encryption_key` if blank. |
| `CORS_ORIGINS_RAW` | `http://localhost:3000,http://127.0.0.1:3000` | Comma-separated list of frontend origins. |

If you delete `data/.encryption_key`, all UI-stored API keys become
unreadable; the next startup will issue a fresh key.

## Frontend

The frontend is mostly self-configuring.

| Variable | Default | Description |
| --- | --- | --- |
| `BACKEND_URL` | `http://localhost:8000` | Server-only. Where Next.js proxies `/api/*` and `/health` from. Set to `http://backend:8000` inside Docker (already in `docker-compose.yml`). |
| `NEXT_PUBLIC_API_BASE` | `http://localhost:8000` | Legacy fallback used when `BACKEND_URL` is unset. |

## Behaviour flags (UI only)

These live in `data/settings.json` and are toggled from the Settings page.

| Flag | Default | Effect |
| --- | --- | --- |
| `auto_config_on_upload` | `true` | When you upload a dataset, the agent is automatically asked to suggest a config. |
| `show_agent_reasoning` | `true` | Show the agent's per-field reasoning in the inspector and chat. |
