# LLM Fine-Tuning Platform - Backend

An industry-scale, agent-based backend system for executing drag-and-drop LLM fine-tuning pipelines with real-time log streaming.

## 🎯 Overview

This platform enables users to:
- Upload and validate datasets
- Configure fine-tuning pipelines visually (drag-and-drop)
- Train LLMs using LoRA, QLoRA, or full fine-tuning
- Monitor training progress with real-time logs
- Evaluate and compare models
- Export trained models in multiple formats

## 🏗️ Architecture

### Core Principles
- **Agent-Based Execution**: Modular, stateless agents for each pipeline step
- **Asynchronous & Fault-Tolerant**: Background workers with retry logic
- **Real-Time Observability**: Structured logs streamed via Redis Streams
- **Production-Ready**: Docker-based deployment with GPU support

### Technology Stack
- **API**: FastAPI (async, high-performance)
- **Task Queue**: Redis + RQ (background job execution)
- **Database**: PostgreSQL (metadata), Redis (state/logs)
- **Object Storage**: MinIO (S3-compatible)
- **ML Libraries**: PyTorch, Transformers, PEFT, bitsandbytes
- **Deployment**: Docker Compose (K8s-ready architecture)

## 📁 Project Structure

```
app/
├── agents/              # Agent implementations
│   ├── base_agent.py           # Abstract base class
│   ├── orchestrator.py         # DAG execution engine
│   ├── dataset_agent.py        # Dataset ingestion
│   ├── validation_agent.py     # Data validation
│   └── preprocessing_agent.py  # LLM-native preprocessing
│
├── training/            # Training modules (to be implemented)
│   ├── lora.py                 # LoRA training
│   ├── qlora.py                # QLoRA training
│   └── full_finetune.py        # Full fine-tuning
│
├── preprocessing/       # Preprocessing utilities
│   ├── cleaning.py             # Text normalization
│   ├── dedup.py                # Deduplication
│   ├── chunking.py             # Token-aware chunking
│   ├── tokenization.py         # Tokenizer integration
│   └── prompt_formatting.py    # Template application
│
├── evaluation/          # Evaluation metrics (to be implemented)
│   ├── metrics.py              # Task-specific metrics
│   └── comparison.py           # Model comparison
│
├── export/              # Model export (to be implemented)
│   ├── adapter_export.py       # LoRA adapter export
│   ├── merged_export.py        # Merged model export
│   └── gguf_export.py          # GGUF quantization
│
├── infra/               # Infrastructure layer
│   ├── redis.py                # Redis client
│   ├── queue.py                # Task queue abstraction
│   ├── logging_stream.py       # Real-time log streaming
│   └── gpu_manager.py          # GPU resource management
│
├── storage/             # Storage layer
│   ├── object_store.py         # MinIO/S3 abstraction
│   └── model_registry.py       # Model metadata tracking
│
├── api/                 # API routes (to be implemented)
│   └── routes/
│       ├── jobs.py             # Job management
│       ├── datasets.py         # Dataset endpoints
│       ├── models.py           # Model registry
│       └── logs.py             # Log streaming (SSE)
│
├── utils/               # Utilities
│   ├── config.py               # Configuration management
│   ├── exceptions.py           # Custom exceptions
│   └── logging.py              # Structured logging
│
└── main.py              # FastAPI application
```

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- NVIDIA GPU with CUDA support (for training)
- nvidia-docker2 (for GPU passthrough)

### 1. Environment Setup

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your configuration
# Key settings: GPU_ENABLED, CUDA_VISIBLE_DEVICES
```

### 2. Start Services

```bash
# Start all services
docker-compose up -d

# Check service health
docker-compose ps

# View logs
docker-compose logs -f api
docker-compose logs -f worker
```

### 3. Verify Installation

```bash
# Check API health
curl http://localhost:8000/health

# Access API documentation
open http://localhost:8000/docs

# Access MinIO console
open http://localhost:9001
# Login: minioadmin / minioadmin
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `API_HOST` | API host | `0.0.0.0` |
| `API_PORT` | API port | `8000` |
| `REDIS_URL` | Redis connection URL | `redis://redis:6379/0` |
| `DATABASE_URL` | PostgreSQL URL | See `.env.example` |
| `MINIO_ENDPOINT` | MinIO endpoint | `minio:9000` |
| `GPU_ENABLED` | Enable GPU support | `true` |
| `CUDA_VISIBLE_DEVICES` | GPU devices | `0` |
| `DEFAULT_LORA_R` | Default LoRA rank | `16` |
| `DEFAULT_LORA_ALPHA` | Default LoRA alpha | `32` |
| `DEFAULT_BATCH_SIZE` | Default batch size | `4` |

### Training Defaults

```python
# LoRA Configuration
lora_r = 16
lora_alpha = 32
lora_dropout = 0.05
target_modules = ["q_proj", "v_proj"]  # Attention layers

# QLoRA Configuration
quantization = "4bit-nf4"
compute_dtype = "bfloat16"
double_quantization = True
```

## 📊 Agent Pipeline Example

```python
# Example pipeline configuration
pipeline = {
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
                "clean": True,
                "dedup": True,
                "template": "alpaca"
            }
        }
    ],
    "edges": [
        {"from_agent": "dataset", "to_agent": "validation"},
        {"from_agent": "validation", "to_agent": "preprocessing"}
    ]
}
```

## 🔍 Real-Time Log Streaming

Logs are streamed via Redis Streams and can be consumed using Server-Sent Events (SSE):

```python
# Example: Consuming logs (client-side)
import requests

response = requests.get(
    "http://localhost:8000/api/v1/logs/stream/abc123",
    stream=True
)

for line in response.iter_lines():
    if line:
        event = json.loads(line.decode('utf-8'))
        print(f"[{event['level']}] {event['message']}")
```

### Log Event Schema

```json
{
  "run_id": "abc123",
  "timestamp": "2025-12-13T01:30:00Z",
  "agent": "TrainingAgent",
  "level": "METRIC",
  "message": "Training step completed",
  "step": 420,
  "epoch": 2,
  "loss": 1.73
}
```

## 🧪 Development

### Running Locally (without Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Start Redis
redis-server

# Start MinIO
minio server /data --console-address ":9001"

# Run API
python -m app.main

# Run worker
rq worker --url redis://localhost:6379/0 training evaluation orchestration default
```

### Running Tests

```bash
# Unit tests
pytest tests/unit -v

# Integration tests
pytest tests/integration -v --slow

# Coverage
pytest --cov=app --cov-report=html
```

## 📝 API Endpoints (Planned)

### Jobs
- `POST /api/v1/jobs` - Submit new pipeline
- `GET /api/v1/jobs/{job_id}` - Get job status
- `GET /api/v1/jobs/{job_id}/progress` - Get detailed progress
- `DELETE /api/v1/jobs/{job_id}` - Cancel job

### Datasets
- `POST /api/v1/datasets/upload` - Upload dataset
- `GET /api/v1/datasets/{dataset_id}` - Get dataset info
- `GET /api/v1/datasets/{dataset_id}/preview` - Preview samples

### Models
- `GET /api/v1/models` - List trained models
- `GET /api/v1/models/{model_id}` - Get model details
- `GET /api/v1/models/{model_id}/download` - Download model
- `GET /api/v1/models/{model_id}/card` - Get model card

### Logs
- `GET /api/v1/logs/stream/{run_id}` - Stream logs (SSE)
- `GET /api/v1/logs/history/{run_id}` - Get log history

## 🛡️ Security Considerations

- Input validation on all API endpoints
- File upload size limits (configurable)
- Sanitized user-provided model names
- Secure credential management (environment variables)
- Rate limiting (can be enabled with slowapi)

## 🚧 Roadmap

### Phase 1: Core Infrastructure ✅
- [x] Project structure
- [x] Redis integration
- [x] Object storage (MinIO)
- [x] GPU manager
- [x] Structured logging

### Phase 2: Agent Framework ✅
- [x] Base agent class
- [x] Orchestrator (DAG execution)
- [x] DatasetAgent
- [x] ValidationAgent
- [x] PreprocessingAgent

### Phase 3: Training (In Progress)
- [ ] TrainingAgent
- [ ] LoRA training module
- [ ] QLoRA training module
- [ ] Full fine-tuning module
- [ ] Training callbacks for log streaming

### Phase 4: Evaluation & Export
- [ ] EvaluationAgent
- [ ] ComparisonAgent
- [ ] ExportAgent
- [ ] Model card generation

### Phase 5: API & UI Integration
- [ ] Complete API routes
- [ ] WebSocket/SSE log streaming
- [ ] Frontend integration endpoints

### Phase 6: Production Features
- [ ] Kubernetes deployment
- [ ] Experiment tracking (MLflow)
- [ ] Cost estimation
- [ ] Training resumption
- [ ] A/B testing framework

## 📄 License

This project is licensed under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:
1. Fork the repository
2. Create a feature branch
3. Write tests for new features
4. Ensure all tests pass
5. Submit a pull request

## 📧 Support

For issues and questions:
- GitHub Issues: [Create an issue](https://github.com/your-repo/issues)
- Documentation: `/docs` endpoint when API is running

---

**Built with ❤️ for the ML community**
