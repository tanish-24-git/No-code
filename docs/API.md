# HTTP API reference

All routes are mounted under `/api` except `/health` and `/`. The
backend ships with auto-generated Swagger UI at `/docs` and ReDoc at
`/redoc`.

Base URL during development: `http://localhost:8000`.

## Health

### `GET /health`

Returns server status, hardware info, and a config-state summary used
by the frontend's first-run wizard.

```json
{
  "status": "ok",
  "version": "2.0.0",
  "hardware": {
    "device": "cuda",
    "gpu_name": "NVIDIA GeForce RTX 3060",
    "vram_gb": 12.0,
    "cuda_version": "12.1",
    "platform": "Windows-11-...",
    "python": "3.11.9",
    "recommended_trainer": "lora"
  },
  "llm": {
    "configured": true,
    "provider": "anthropic",
    "model": "claude-sonnet-4-5",
    "base_url": null,
    "source": "ui"
  },
  "hf": { "configured": true, "source": "ui" }
}
```

## Settings

### `GET /api/settings`
Returns the current resolved settings (UI overrides env). API keys are
masked.

### `POST /api/settings/llm`
Save LLM provider config. Omit `api_key` to keep the existing one.
```json
{ "provider": "openai", "model": "gpt-4o-mini", "base_url": null, "api_key": "sk-..." }
```

### `DELETE /api/settings/llm`
Clear the UI-stored LLM config. Falls back to env.

### `POST /api/settings/hf-token`
```json
{ "token": "hf_..." }
```

### `DELETE /api/settings/hf-token`

### `PUT /api/settings`
Update behaviour flags only.
```json
{ "auto_config_on_upload": true, "show_agent_reasoning": false }
```

### `POST /api/settings/verify-llm`
Probes the configured provider's `/v1/models` endpoint.
```json
{ "valid": true, "detail": "anthropic reachable", "models": ["claude-sonnet-4-5", ...] }
```

### `POST /api/settings/verify-hf`
Calls `https://huggingface.co/api/whoami-v2` with the saved token.
```json
{ "valid": true, "username": "alice", "detail": "HF token OK" }
```

## Datasets

### `POST /api/datasets/upload`
Multipart form. Field name: `file`. Accepts `.csv`, `.json`, `.jsonl`.
Returns the parsed dataset record (schema, sample rows, stats).

### `GET /api/datasets`
List dataset records, newest first.

### `GET /api/datasets/{id}`

### `DELETE /api/datasets/{id}`
Removes both the record and the underlying file.

## Pipelines

### `POST /api/pipelines`
```json
{ "name": "qa pipeline", "description": "...", "dataset_id": "abc..." }
```
Creates a pipeline with a default 5-node DAG (dataset, preprocess, train,
evaluate, export).

### `GET /api/pipelines`

### `GET /api/pipelines/{id}`

### `PUT /api/pipelines/{id}`
Patch shape. Send only the fields you want to change.
```json
{ "name": "renamed", "config": { "epochs": 5, "lora_rank": 32 }, "node_graph": { ... } }
```

### `DELETE /api/pipelines/{id}`

## Jobs

### `POST /api/jobs/start`
```json
{ "pipeline_id": "..." }
```
Returns immediately with `{ job_id, status: "queued" }`. Execution
happens on a background thread.

### `GET /api/jobs`

### `GET /api/jobs/{id}`
Includes progress, current epoch, and a list of metric points.

### `POST /api/jobs/{id}/stop`
Sets the cooperative stop flag. The worker checks between nodes and at
each training step.

### `DELETE /api/jobs/{id}`
Returns 409 if the job is still running.

### `GET /api/jobs/{id}/logs` (Server-Sent Events)
Replays buffered logs first, then streams live. Sends `data: [DONE]`
when the job reaches a terminal state.

```
data: [INFO] Job ... starting for pipeline 'qa pipeline'

data: [STEP] (1/5) running 'dataset'

data: [METRIC] epoch=1 step=1 loss=2.5

...

data: [DONE]
```

## Inference endpoints

### `POST /api/inferences`
Register an endpoint.
```json
{
  "name": "local llama 3.1",
  "kind": "ollama",
  "base_url": "http://localhost:11434",
  "api_key": null,
  "default_model": "llama3.1:8b",
  "notes": "8B Q4 quant"
}
```

`kind` is one of `ollama`, `openai_compat`, `huggingface_inference`,
`anthropic`.

### `GET /api/inferences` and `GET /api/inferences/{id}`

### `PUT /api/inferences/{id}`

### `DELETE /api/inferences/{id}`

### `POST /api/inferences/{id}/probe`
Probes the endpoint and saves the result on the record.
```json
{ "reachable": true, "latency_ms": 23.4, "models": ["llama3.1:8b", "qwen2.5:7b"], "detail": null }
```

### `POST /api/inferences/generate`
Quick generate request against any registered endpoint. Useful for
testing.
```json
{ "inference_id": "...", "prompt": "hello", "model": null, "max_tokens": 256, "temperature": 0.7 }
```

## Models

### `GET /api/models`
List local entries (pulled base models and trained outputs).

### `GET /api/models/{id}`

### `POST /api/models/pull`
Pull a base model from the Hugging Face Hub. Uses the saved HF token.
```json
{ "repo_id": "meta-llama/Llama-3.1-8B-Instruct" }
```
Returns immediately. Poll `/api/models/pull-status/{repo_id}` for
progress.

### `GET /api/models/pull-status/{repo_id}` (path supports slashes)

### `POST /api/models/{id}/push-hub`
```json
{ "repo_id": "your-username/your-model" }
```

### `GET /api/models/{id}/push-status`

## Agent

### `POST /api/agent/chat` (Server-Sent Events)
```json
{
  "messages": [
    { "role": "user", "content": "configure this pipeline for my hardware" }
  ],
  "pipeline_id": "...",
  "inference_id": null,
  "dataset_id": null
}
```
Streams the agent's response as SSE. Tool calls happen on the server
and surface in the stream as `[tool: tool_name]` markers.

### `POST /api/agent/apply-config`
Convenience endpoint for the UI to persist a config the agent suggested.
```json
{
  "pipeline_id": "...",
  "config": { "epochs": 3, "lora_rank": 16, "base_model": "..." },
  "reasoning": { "lora_rank": "small dataset, low rank reduces overfit" }
}
```

## Error format

FastAPI default. Validation errors:
```json
{ "detail": [ { "loc": [...], "msg": "...", "type": "..." } ] }
```
Application errors:
```json
{ "detail": "Pipeline not found" }
```
