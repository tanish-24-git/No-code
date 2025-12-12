# API Specification

## Base URL
```
http://localhost:8000/api/v1
```

## Authentication
Currently no authentication required (to be implemented).

---

## Endpoints

### Health & Status

#### GET /health
Check system health status.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "services": {
    "redis": "healthy",
    "api": "healthy"
  }
}
```

---

### Jobs Management

#### POST /api/v1/jobs
Submit a new pipeline job.

**Request Body:**
```json
{
  "pipeline_config": {
    "run_id": "abc123",
    "nodes": [
      {
        "agent_name": "dataset",
        "agent_class": "DatasetAgent",
        "config": {
          "file_path": "s3://datasets/training_data.csv",
          "format": "csv"
        }
      },
      {
        "agent_name": "validation",
        "agent_class": "ValidationAgent",
        "config": {}
      },
      {
        "agent_name": "preprocessing",
        "agent_class": "PreprocessingAgent",
        "config": {
          "base_model": "meta-llama/Llama-2-7b-hf",
          "clean": true,
          "dedup": true,
          "template": "alpaca",
          "template_mapping": {
            "instruction": "question",
            "output": "answer"
          }
        }
      }
    ],
    "edges": [
      {"from_agent": "dataset", "to_agent": "validation"},
      {"from_agent": "validation", "to_agent": "preprocessing"}
    ],
    "global_config": {
      "max_retries": 3
    }
  }
}
```

**Response:**
```json
{
  "job_id": "job_abc123",
  "status": "pending",
  "created_at": "2025-12-13T01:00:00Z"
}
```

#### GET /api/v1/jobs/{job_id}
Get job status and results.

**Response:**
```json
{
  "job_id": "job_abc123",
  "status": "running",
  "created_at": "2025-12-13T01:00:00Z",
  "started_at": "2025-12-13T01:00:05Z",
  "progress": {
    "completed_agents": ["dataset", "validation"],
    "current_agent": "preprocessing",
    "total_agents": 3
  },
  "result": null
}
```

**Status Values:**
- `pending` - Job queued, not started
- `running` - Job in progress
- `completed` - Job finished successfully
- `failed` - Job failed with errors

#### DELETE /api/v1/jobs/{job_id}
Cancel a running job.

**Response:**
```json
{
  "job_id": "job_abc123",
  "status": "cancelled"
}
```

---

### Datasets

#### POST /api/v1/datasets/upload
Upload a dataset file.

**Request (multipart/form-data):**
- `file`: Dataset file (CSV, JSON, JSONL, Parquet, TXT)
- `format`: File format (optional, auto-detected)

**Response:**
```json
{
  "dataset_id": "ds_xyz789",
  "filename": "training_data.csv",
  "size_bytes": 1048576,
  "s3_path": "s3://datasets/ds_xyz789/training_data.csv",
  "analysis": {
    "rows": 10000,
    "columns": ["text", "label"],
    "text_column": "text",
    "stats": {
      "avg_length": 256.5,
      "min_length": 10,
      "max_length": 1024
    }
  }
}
```

#### GET /api/v1/datasets/{dataset_id}
Get dataset information.

**Response:**
```json
{
  "dataset_id": "ds_xyz789",
  "filename": "training_data.csv",
  "format": "csv",
  "rows": 10000,
  "columns": ["text", "label"],
  "created_at": "2025-12-13T01:00:00Z",
  "s3_path": "s3://datasets/ds_xyz789/training_data.csv"
}
```

#### GET /api/v1/datasets/{dataset_id}/preview
Preview dataset samples.

**Query Parameters:**
- `limit`: Number of samples (default: 10, max: 100)

**Response:**
```json
{
  "dataset_id": "ds_xyz789",
  "samples": [
    {"text": "Sample text 1", "label": "positive"},
    {"text": "Sample text 2", "label": "negative"}
  ],
  "total_rows": 10000
}
```

---

### Models

#### GET /api/v1/models
List all trained models.

**Query Parameters:**
- `limit`: Number of models (default: 20)
- `offset`: Pagination offset (default: 0)
- `base_model`: Filter by base model name

**Response:**
```json
{
  "models": [
    {
      "model_id": "model_abc123",
      "run_id": "abc123",
      "model_name": "llama2-finetuned",
      "base_model": "meta-llama/Llama-2-7b-hf",
      "created_at": "2025-12-13T01:00:00Z",
      "metrics": {
        "accuracy": 0.92,
        "f1": 0.89
      },
      "exports": ["adapter", "merged"]
    }
  ],
  "total": 1,
  "limit": 20,
  "offset": 0
}
```

#### GET /api/v1/models/{model_id}
Get model details.

**Response:**
```json
{
  "model_id": "model_abc123",
  "run_id": "abc123",
  "model_name": "llama2-finetuned",
  "base_model": "meta-llama/Llama-2-7b-hf",
  "dataset_id": "ds_xyz789",
  "training_config": {
    "method": "qlora",
    "lora_r": 16,
    "lora_alpha": 32,
    "epochs": 3,
    "batch_size": 4,
    "learning_rate": 2e-4
  },
  "metrics": {
    "accuracy": 0.92,
    "f1": 0.89,
    "final_loss": 0.42
  },
  "exports": {
    "adapter": "s3://artifacts/abc123/adapter",
    "merged": "s3://artifacts/abc123/merged"
  },
  "created_at": "2025-12-13T01:00:00Z"
}
```

#### GET /api/v1/models/{model_id}/download
Download model artifacts.

**Query Parameters:**
- `format`: Export format (`adapter`, `merged`, `gguf`)

**Response:**
- Binary file download or presigned S3 URL

#### GET /api/v1/models/{model_id}/card
Get model card (Markdown).

**Response:**
```markdown
# llama2-finetuned

## Model Description
This model was fine-tuned from `meta-llama/Llama-2-7b-hf`...

## Training Configuration
...

## Evaluation Metrics
...
```

---

### Logs & Monitoring

#### GET /api/v1/logs/stream/{run_id}
Stream real-time logs (Server-Sent Events).

**Response (SSE):**
```
data: {"run_id":"abc123","timestamp":"2025-12-13T01:00:00Z","agent":"DatasetAgent","level":"INFO","message":"Starting dataset ingestion"}

data: {"run_id":"abc123","timestamp":"2025-12-13T01:00:05Z","agent":"DatasetAgent","level":"INFO","message":"Dataset loaded: 10000 rows"}

data: {"run_id":"abc123","timestamp":"2025-12-13T01:00:10Z","agent":"TrainingAgent","level":"METRIC","message":"Training step completed","step":100,"epoch":1,"loss":1.85}
```

**Event Types:**
- `INFO` - Informational messages
- `WARN` - Warnings
- `ERROR` - Errors
- `METRIC` - Training metrics

#### GET /api/v1/logs/history/{run_id}
Get historical logs.

**Query Parameters:**
- `limit`: Number of log entries (default: 100, max: 1000)

**Response:**
```json
{
  "run_id": "abc123",
  "logs": [
    {
      "timestamp": "2025-12-13T01:00:00Z",
      "agent": "DatasetAgent",
      "level": "INFO",
      "message": "Starting dataset ingestion"
    },
    {
      "timestamp": "2025-12-13T01:00:05Z",
      "agent": "DatasetAgent",
      "level": "INFO",
      "message": "Dataset loaded: 10000 rows"
    }
  ],
  "total": 2
}
```

---

## Error Responses

All errors follow this format:

```json
{
  "error": "ERROR_CODE",
  "message": "Human-readable error message",
  "metadata": {
    "field": "additional context"
  }
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `VALIDATION_ERROR` | 400 | Input validation failed |
| `DATASET_ERROR` | 400 | Dataset processing error |
| `TRAINING_ERROR` | 500 | Training execution error |
| `STORAGE_ERROR` | 500 | Object storage error |
| `CONFIG_ERROR` | 400 | Invalid configuration |
| `NOT_FOUND` | 404 | Resource not found |
| `INTERNAL_ERROR` | 500 | Unexpected server error |

### Example Error Response

```json
{
  "error": "VALIDATION_ERROR",
  "message": "Input validation failed for DatasetAgent",
  "metadata": {
    "errors": [
      {
        "field": "file_path",
        "message": "field required"
      }
    ]
  }
}
```

---

## Rate Limiting

(To be implemented)

- **Default**: 100 requests per minute per IP
- **Authenticated**: 1000 requests per minute per user

**Rate Limit Headers:**
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1702425600
```

---

## Pagination

List endpoints support pagination:

**Query Parameters:**
- `limit`: Items per page (default: 20, max: 100)
- `offset`: Number of items to skip (default: 0)

**Response:**
```json
{
  "items": [...],
  "total": 150,
  "limit": 20,
  "offset": 40,
  "has_more": true
}
```

---

## WebSocket Support

(Planned for future release)

Real-time bidirectional communication for:
- Live training updates
- Interactive pipeline control
- Multi-user collaboration

---

## SDK Examples

### Python SDK (Example)

```python
from llm_platform import Client

# Initialize client
client = Client(base_url="http://localhost:8000/api/v1")

# Upload dataset
dataset = client.datasets.upload("training_data.csv")
print(f"Dataset ID: {dataset.dataset_id}")

# Submit pipeline job
job = client.jobs.create({
    "pipeline_config": {
        "nodes": [...],
        "edges": [...]
    }
})

# Stream logs
for log in client.logs.stream(job.run_id):
    print(f"[{log.level}] {log.message}")

# Get trained model
model = client.models.get(job.model_id)
model.download(format="adapter", path="./adapter")
```

### JavaScript SDK (Example)

```javascript
import { LLMPlatformClient } from '@llm-platform/sdk';

const client = new LLMPlatformClient({
  baseUrl: 'http://localhost:8000/api/v1'
});

// Upload dataset
const dataset = await client.datasets.upload(file);

// Submit job
const job = await client.jobs.create({
  pipelineConfig: { ... }
});

// Stream logs (SSE)
const eventSource = client.logs.stream(job.runId);
eventSource.onmessage = (event) => {
  const log = JSON.parse(event.data);
  console.log(`[${log.level}] ${log.message}`);
};
```

---

## Changelog

### v1.0.0 (Current)
- Initial release
- Core agent framework
- Dataset, validation, preprocessing agents
- Real-time log streaming
- Docker deployment

### Planned Features
- Training agents (LoRA, QLoRA, Full)
- Evaluation and comparison
- Model export (adapters, merged, GGUF)
- Authentication & authorization
- Rate limiting
- WebSocket support
- Python & JavaScript SDKs
