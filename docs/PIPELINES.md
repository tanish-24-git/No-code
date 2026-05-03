# Pipelines and jobs

A pipeline is a node graph plus a flat training config. A job is a
single execution of a pipeline.

## Pipeline shape

```
{
  "id": "...",
  "name": "qa pipeline",
  "node_graph": {
    "nodes": [
      { "id": "dataset",    "type": "dataset",    "position": {"x": 40,   "y": 80}, "data": { "dataset_id": "..." } },
      { "id": "preprocess", "type": "preprocess", "position": {"x": 360,  "y": 80}, "data": {} },
      { "id": "train",      "type": "train",      "position": {"x": 700,  "y": 60}, "data": {} },
      { "id": "evaluate",   "type": "evaluate",   "position": {"x": 1040, "y": 80}, "data": {} },
      { "id": "export",     "type": "export",     "position": {"x": 1380, "y": 80}, "data": {} }
    ],
    "edges": [
      { "id": "e1", "source": "dataset",    "target": "preprocess" },
      { "id": "e2", "source": "preprocess", "target": "train" },
      { "id": "e3", "source": "train",      "target": "evaluate" },
      { "id": "e4", "source": "evaluate",   "target": "export" }
    ],
    "viewport": { "x": 0, "y": 0, "zoom": 1 }
  },
  "config": { ...22 fields... },
  "is_agent_configured": false,
  "reasoning": {}
}
```

When you create a pipeline, the backend seeds it with the default
5-node DAG above. Nodes are draggable; edges can be added or removed in
React Flow.

## Built-in node types

| Type | What the handler does today |
| --- | --- |
| `dataset` | Loads the attached dataset record into the run context. |
| `preprocess` | Placeholder. Wire your cleaning/dedup/templating logic here. |
| `train` | Stub trainer. Streams fake metrics and writes a placeholder artifact. |
| `evaluate` | Placeholder. |
| `export` | Logs the artifact path. |

To wire real training, edit
`backend/app/services/job_service.py::_handler_train`. The function
signature is:

```python
def _handler_train(job, config, node, ctx, log, stop_flag):
    ...
```

`config` is the typed `PipelineConfig`. `log("...")` appends a line to
the SSE stream. `ctx` is a dict shared across handlers; set
`ctx["model_output_path"]` when training produces an artifact so the
export node can pick it up.

## Adding a custom node type

1. Add a handler function with the signature above.
2. Register it in the `_HANDLERS` dict at the bottom of
   `job_service.py`.
3. The frontend's `PipelineCanvas` and `Inspector` already render any
   node type generically; if you want a custom card, add a key to
   `nodeTypes` in `PipelineCanvas.tsx`.

## The 22-field config

Lives in `backend/app/api/schemas/pipeline.py` as `PipelineConfig`.

| Field | Type | Default |
| --- | --- | --- |
| `project_name` | str | `"untitled"` |
| `description` | str? | none |
| `tags` | list[str] | `[]` |
| `dataset_id` | str? | none |
| `target_column` | str? | none |
| `input_columns` | list[str] | `[]` |
| `split_ratio` | float [0.5, 0.95] | `0.8` |
| `task_type` | enum | `"Chat"` |
| `output_type` | enum | `"text"` |
| `domain` | enum | `"General"` |
| `language` | str | `"en"` |
| `training_mode` | enum | `"balanced"` |
| `training_method` | enum | `"lora"` |
| `base_model` | str | `"TinyLlama/TinyLlama-1.1B-Chat-v1.0"` |
| `epochs` | int [1, 100] | `3` |
| `batch_size` | int [1, 128] | `4` |
| `learning_rate` | float [1e-6, 1e-1] | `2e-4` |
| `max_seq_len` | int [64, 8192] | `512` |
| `lora_rank` | enum (8, 16, 32, 64) | `16` |
| `gradient_accumulation` | int [1, 64] | `4` |
| `precision` | enum | `"fp16"` |
| `early_stopping` | bool | `true` |
| `class_balancing` | bool | `false` |
| `data_augmentation` | bool | `false` |
| `resume_checkpoint` | str? | none |

The agent can patch any subset of these via `suggest_pipeline_config`.

## Jobs

Starting a job:

```
POST /api/jobs/start { "pipeline_id": "..." }
-> { "job_id": "...", "status": "queued" }
```

The pipeline is loaded, a JobRecord is written with `status=queued`,
and a daemon thread is spawned. The endpoint returns immediately.

Inside the worker:

1. `_topo_sort(node_graph)` produces an execution order. Cycles raise.
2. Each node's handler runs in sequence, sharing a `ctx` dict.
3. Each handler can call `log(line)` to emit an SSE-visible line. The
   `train` handler also appends `JobMetric` entries to the job record
   so the inspector can plot loss curves later.
4. The `stop_flag` (a `threading.Event`) is checked between nodes and
   inside the train handler's inner loop. `POST /api/jobs/{id}/stop`
   sets it.
5. Final status is `completed`, `failed`, or `stopped`.

## Logs

Streamed at `GET /api/jobs/{id}/logs` over SSE. The endpoint:

- Replays buffered lines first, so a late subscriber catches up.
- Subscribes to a live `asyncio.Queue` for new lines.
- Sends `data: [DONE]` when the job reaches a terminal state.
- Sends `: keepalive` comments every 15 seconds so reverse proxies do
  not drop the connection.

The log tags conventionally used by the built-in handlers:

| Prefix | Meaning |
| --- | --- |
| `[INFO]` | Informational. |
| `[WARN]` | Soft problem. |
| `[ERROR]` / `[FATAL]` | Hard failure. |
| `[STEP]` | Node boundary. |
| `[METRIC]` | Training metric (epoch, step, loss, lr). |
| `[DONE]` | Terminal frame. UI closes the EventSource. |

The frontend colors lines by prefix in `LogPanel.tsx`.
