# Architecture

## Layout

```
.
+-- backend/                 FastAPI service. Single process, single thread per job.
|   +-- app/
|   |   +-- agents/          Agent runner + provider implementations + tools.
|   |   |   +-- agent.py     Public entry point (run_chat).
|   |   |   +-- providers.py Anthropic and OpenAI streaming with tool use.
|   |   |   +-- tools.py     Tool registry + impls + system prompt.
|   |   +-- api/
|   |   |   +-- routes/      One file per endpoint group.
|   |   |   +-- schemas/     Pydantic request/response shapes.
|   |   +-- services/        Business logic (datasets, pipelines, jobs, inference, hf).
|   |   +-- storage/         JSON-on-disk store (atomic writes, listing, singletons).
|   |   +-- utils/           config (env loader), crypto (Fernet), hardware probe.
|   |   +-- main.py          App factory + startup logging.
|   +-- requirements.txt
|   +-- .env.example
+-- frontend/                Next.js 14 App Router.
|   +-- src/
|   |   +-- app/
|   |   |   +-- page.tsx              Home.
|   |   |   +-- setup/page.tsx        First-run wizard.
|   |   |   +-- settings/page.tsx     Settings page.
|   |   |   +-- inference/page.tsx    Inference endpoints.
|   |   |   +-- models/page.tsx       Local + Hub models.
|   |   |   +-- playground/page.tsx   Canvas + inspector + chat + log panel.
|   |   +-- components/      Nav, ConfigBanner, AgentChat, PipelineCanvas, ...
|   |   +-- lib/             api client, types, classnames helper.
|   +-- next.config.mjs       Rewrites /api/* and /health to the backend.
+-- docs/                    This documentation.
+-- data/      uploads/      models/   Created at runtime.
```

## Process model

There is exactly one server process. FastAPI handles HTTP requests on
the asyncio event loop. Long-running work uses one of two patterns:

- **Background tasks.** `BackgroundTasks` from FastAPI fires off
  short-lived work (under a second) after the response is returned.
- **Worker threads.** For real jobs, the API enqueues a thread (one per
  job). The thread reads the pipeline's node graph, topologically
  sorts it, and runs node handlers in order. Logs append to an
  in-memory ring buffer plus a per-job file at `data/jobs/<id>.log`.

This means the project scales vertically (one machine) but not
horizontally. That is the right tradeoff for a local tool. If you want
multi-node training, the trainer of your choice (Accelerate, Ray, etc.)
would be invoked from inside the train node handler.

## Storage layer

JSON files on disk. One file per record under `data/<collection>/<id>.json`.

| Collection | Purpose |
| --- | --- |
| `datasets/` | Uploaded dataset metadata + parsed schema and sample rows. |
| `pipelines/` | Node graphs and 22-field training configs. |
| `jobs/` | Job records + per-job append-only `.log` file. |
| `inferences/` | User-registered inference endpoints (Ollama, OpenAI-compat, ...). |
| `models/` | Pulled base models and trained outputs. |
| `settings.json` | Singleton: encrypted API keys + behaviour flags. |
| `.encryption_key` | Fernet key for encrypting API keys at rest. |

Writes are atomic: the store writes to a temp file and renames it,
which on POSIX and modern Windows guarantees readers never see a partial
file.

## Request flow: chat with the agent

```
Browser (AgentChat.tsx)
  |
  | POST /api/agent/chat  { messages, pipeline_id, inference_id, dataset_id }
  v
Next.js dev rewrites  /api/agent/chat  ->  http://localhost:8000/api/agent/chat
  |
  v
backend/app/api/routes/agent.py
  |
  | spawns worker thread that calls agent.run_chat(...)
  v
backend/app/agents/agent.py
  |
  | reads LLM config (UI > env), builds augmented system prompt
  v
backend/app/agents/providers.py
  |
  | streams text from Anthropic OR OpenAI-compatible endpoint
  | for each tool_use, invokes tools.run_tool() and feeds the result back
  v
Worker thread enqueues each chunk to an asyncio Queue
  |
  v
Endpoint reads chunks, formats as SSE frames, yields them through
StreamingResponse to the browser
  |
  v
AgentChat.tsx parses `data: ...` lines, appends to message bubble
```

The Anthropic SDK is synchronous, so we run the producer on a thread
and bridge chunks back to asyncio with `run_coroutine_threadsafe`. The
OpenAI SDK supports both, but we use the same bridge pattern for
symmetry.

## Request flow: run a pipeline

```
Browser (Playground)  POST /api/jobs/start { pipeline_id }
  |
  v
backend/app/api/routes/jobs.py
  | services.job_service.create()  -> JobRecord with status="queued"
  | BackgroundTasks.add_task(job_service.run_in_background, job_id)
  | returns 201 { job_id, status: "queued" }
  v
Browser opens GET /api/jobs/{job_id}/logs (SSE)
  |
  v
backend/app/services/job_service.py
  | run_in_background spawns a daemon thread
  v
Worker thread:
  1. _topo_sort(node_graph) -> list of nodes in execution order
  2. for each node, run _HANDLERS[node.type] with shared ctx
  3. each step calls log() -> append to in-memory deque + .log file +
     fan out to subscribed asyncio.Queue instances
  4. on finish: status=completed/failed/stopped, persist final job record
  |
  v
SSE endpoint:
  - Replays buffered log lines first
  - Subscribes to live queue
  - Sends [DONE] frame when job is terminal
  - Closes the stream
```

## Frontend rendering

- The Playground keeps the active pipeline in SWR state. When the
  agent applies a config via tool use, the chat completion handler
  triggers `mutate('/api/pipelines/{id}')`, and the Inspector and
  Canvas re-render against the new config.
- React Flow holds its own ephemeral graph state. Node-position changes
  are debounced into PUTs against `/api/pipelines/{id}` so the
  on-disk graph stays in sync.
- Job logs use the browser's native `EventSource`. Each frame becomes
  a single line in the log panel; a `[DONE]` frame closes the source.

## Design decisions

- **No DB.** A single-user local tool does not need transactions. JSON
  files are atomic-renamed and easy to inspect. If you ever outgrow
  this, swap `app/storage/store.py` for a SQLite-backed module without
  touching anything upstream.
- **No queue.** Same logic. One job per thread, capped by the number
  of jobs the user starts. If you want a real queue later, add Celery
  or RQ behind the same `job_service` interface.
- **Two providers, not five.** The OpenAI provider, with its
  `base_url` switch, covers every popular local server already
  (Ollama, LM Studio, vLLM) plus most aggregators (OpenRouter,
  Together, Groq). Maintaining bespoke clients per provider is
  not worth the cost.
- **SSE over WebSockets.** SSE is a subset of HTTP, survives the dev
  proxy, and matches the unidirectional nature of "agent streams a
  response" and "job streams logs". WebSockets buy nothing here.
