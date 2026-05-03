# Inference endpoints

The Inference page lets you register the inference servers you actually
run, so the agent can recommend generation metrics tuned to each one.
The agent reads this list as a tool (`list_inferences`).

## Supported kinds

| `kind` | Server | Probe endpoint | Auth |
| --- | --- | --- | --- |
| `ollama` | Ollama | `GET /api/tags` | none |
| `openai_compat` | OpenAI cloud, vLLM, LM Studio, OpenRouter, Together, Groq, ... | `GET /v1/models` | bearer if set |
| `huggingface_inference` | HF Inference API | `GET /` | bearer if set |
| `anthropic` | Anthropic API | `GET /v1/models` | x-api-key |

## Adding an endpoint

Open `/inference` and click "+ add endpoint". You need:

- **name** - human-friendly label.
- **kind** - one of the four above.
- **base URL** - examples below.
- **api key** - optional for local servers.
- **default model** - what to use when no model is specified.

### Base URL examples

| Kind | URL |
| --- | --- |
| `ollama` | `http://localhost:11434` |
| `openai_compat` (LM Studio) | `http://localhost:1234` |
| `openai_compat` (vLLM) | `http://localhost:8000` |
| `openai_compat` (OpenAI) | `https://api.openai.com` |
| `openai_compat` (OpenRouter) | `https://openrouter.ai/api` |
| `huggingface_inference` | `https://api-inference.huggingface.co` |
| `anthropic` | `https://api.anthropic.com` |

Note: the inference registry uses the **bare host** form, not the `/v1`
suffix, because we add the right path per kind. (The LLM provider for
the agent itself does include `/v1` in `LLM_BASE_URL` for OpenAI-compat
servers, because the OpenAI SDK expects it.)

## Probing

Click "probe" on any endpoint card. The backend hits the kind-specific
list endpoint and stores:

- `reachable` - boolean.
- `latency_ms` - one round trip.
- `models[]` - what the server reports.
- `detail` - error text if reachable was false.

Probe results live on the record and are visible to the agent through
`get_inference` and `list_inferences`.

## Generate (test from the API)

For one-off testing without going through the agent:

```
POST /api/inferences/generate
{
  "inference_id": "<id>",
  "prompt": "ping",
  "model": null,
  "max_tokens": 128,
  "temperature": 0.2
}
```

The backend dispatches to the right wire format per kind. There is no
UI for this yet; use the Swagger docs or curl.

## Agent-suggested metrics

When the agent calls `suggest_inference_metrics`, the metrics dict is
saved on the endpoint record. The Inference page renders it as a small
table on the endpoint card:

```
max_tokens     256
temperature    0.2
top_p          0.9
num_ctx        4096
stop           ["\n\n", "Q:"]
```

The agent picks these based on:

- The endpoint's reported model and quantization (where probeable).
- The user's hardware (CPU vs GPU, VRAM).
- The conversation context (e.g. "short-form QA" vs "long-form summary").

You can edit them by hand later by re-running the suggestion or by
asking the agent for a different optimisation goal ("optimise for
latency", "I want deterministic output", ...).

## Note on the agent's brain vs registered endpoints

These two are independent:

- **`LLM_PROVIDER` / `LLM_BASE_URL`** = which model powers the *agent*.
  The agent talks to *one* LLM at a time, and that LLM cannot be Ollama
  unless you set `LLM_PROVIDER=openai` and `LLM_BASE_URL=http://localhost:11434/v1`.
- **Inference endpoints** = inference servers the *user* runs in
  production. The agent can read them, suggest configs, and benchmark
  them. The agent does not chat through them.

You can register an Ollama endpoint in the Inference page even if the
agent itself runs on Anthropic. They serve different purposes.
