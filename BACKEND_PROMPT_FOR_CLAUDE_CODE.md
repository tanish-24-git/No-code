# FineTune Studio — Backend Architecture Prompt for Claude Code

> Paste this entire prompt into Claude Code to build the complete backend.

---

## PROJECT CONTEXT

You are building **FineTune Studio** — a local-first, open-source LLM fine-tuning platform.
It runs entirely via Docker on the user's machine. Users upload datasets, an AI agent
auto-configures the training pipeline, training runs locally, and the final model can be
saved to disk or pushed to Hugging Face Hub.

This is a complete ground-up rewrite of an existing backend. Discard the old web-based
architecture. The new system is local-inference-first, Docker-native, and agent-driven.

---

## TECH STACK

| Layer | Technology |
|---|---|
| API Framework | FastAPI (Python 3.11) |
| Task Queue | Celery + Redis |
| Database | SQLite (default local) via SQLAlchemy |
| Object Storage | MinIO (S3-compatible, runs in Docker) |
| Training | HuggingFace Transformers + PEFT |
| Agent | LangChain + Anthropic/OpenAI API (user-supplied key) |
| Auth | Simple encrypted local key storage (no JWT needed) |
| Streaming | Server-Sent Events (SSE) for log streaming |
| Containerization | Docker + Docker Compose |

---

## COMPLETE DIRECTORY STRUCTURE

Generate every file listed below:

```
finetune-studio/
├── docker-compose.yml              # Single command: docker compose up
├── .env.example                    # All env vars documented
├── README.md                       # Setup instructions
│
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app/
│   │   ├── main.py                 # FastAPI app, CORS, router registration
│   │   ├── config.py               # Pydantic Settings, env loading
│   │   ├── database.py             # SQLAlchemy engine, session, Base
│   │   │
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── routes/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── health.py       # GET /health
│   │   │   │   ├── settings.py     # API key CRUD, HF token CRUD
│   │   │   │   ├── datasets.py     # Upload, preview, delete, analyze
│   │   │   │   ├── pipelines.py    # Pipeline CRUD + node graph CRUD
│   │   │   │   ├── jobs.py         # Start, stop, status, SSE logs
│   │   │   │   └── models.py       # Export local, push to HF Hub
│   │   │   └── schemas/
│   │   │       ├── __init__.py
│   │   │       ├── settings.py
│   │   │       ├── dataset.py
│   │   │       ├── pipeline.py
│   │   │       ├── job.py
│   │   │       └── model.py
│   │   │
│   │   ├── db/
│   │   │   ├── __init__.py
│   │   │   ├── models.py           # All SQLAlchemy ORM models
│   │   │   └── crud.py             # All DB operations
│   │   │
│   │   ├── agents/
│   │   │   ├── __init__.py
│   │   │   ├── pipeline_agent.py   # AI auto-config agent (CORE)
│   │   │   └── monitoring_agent.py # Real-time training monitor
│   │   │
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── dataset_service.py  # Upload, validate, analyze dataset
│   │   │   ├── training_service.py # Orchestrate training lifecycle
│   │   │   ├── storage_service.py  # MinIO read/write
│   │   │   ├── export_service.py   # Save model locally
│   │   │   └── hf_service.py       # HuggingFace Hub push
│   │   │
│   │   ├── training/
│   │   │   ├── __init__.py
│   │   │   ├── base.py             # Abstract BaseTrainer
│   │   │   ├── lora.py             # LoRA trainer wrapper
│   │   │   ├── qlora.py            # QLoRA trainer wrapper
│   │   │   └── full_finetune.py    # Full fine-tune trainer
│   │   │
│   │   ├── workers/
│   │   │   ├── __init__.py
│   │   │   └── training_worker.py  # Celery task for async training
│   │   │
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── crypto.py           # Fernet encryption for API keys
│   │       ├── gpu.py              # GPU/CPU detection utility
│   │       ├── logger.py           # Structured logging setup
│   │       └── validators.py       # Input validation helpers
│   │
│   └── tests/
│       ├── test_agent.py
│       ├── test_training.py
│       └── test_api.py
│
└── frontend/
    ├── Dockerfile
    └── index.html                  # Static frontend (already built)
```

---

## FILE-BY-FILE IMPLEMENTATION SPEC

### `docker-compose.yml`

```yaml
version: '3.9'
services:

  backend:
    build: ./backend
    ports: ["8000:8000"]
    environment:
      - DATABASE_URL=sqlite:////data/finetune.db
      - MINIO_ENDPOINT=minio:9000
      - MINIO_ACCESS_KEY=minioadmin
      - MINIO_SECRET_KEY=minioadmin
      - REDIS_URL=redis://redis:6379/0
      - MODEL_OUTPUT_DIR=/models
    volumes:
      - ./data:/data
      - ./models:/models
      - ./uploads:/uploads
    depends_on: [redis, minio]

  frontend:
    build: ./frontend
    ports: ["3000:80"]

  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]

  minio:
    image: minio/minio
    ports: ["9000:9000", "9001:9001"]
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    command: server /data --console-address ":9001"
    volumes:
      - minio_data:/data

volumes:
  minio_data:
```

---

### `backend/app/main.py`

- Create FastAPI app with title "FineTune Studio API"
- Register all routers with prefix `/api`
- Add CORS middleware allowing all origins (local use)
- Add startup event: initialize DB tables, create MinIO bucket
- Add `/health` endpoint returning `{ status: "ok", gpu: bool, version: str }`

---

### `backend/app/config.py`

Use `pydantic_settings.BaseSettings`. Include:
```python
DATABASE_URL: str = "sqlite:///./finetune.db"
MINIO_ENDPOINT: str = "localhost:9000"
MINIO_ACCESS_KEY: str = "minioadmin"
MINIO_SECRET_KEY: str = "minioadmin"
MINIO_BUCKET: str = "finetune-datasets"
REDIS_URL: str = "redis://localhost:6379/0"
MODEL_OUTPUT_DIR: str = "./models"
UPLOAD_DIR: str = "./uploads"
ENCRYPTION_KEY: str = ""   # auto-generated if empty
LOG_LEVEL: str = "INFO"
```

---

### `backend/app/db/models.py`

Define these SQLAlchemy models with full relationships:

```python
class UserSettings(Base):
    id: int (PK)
    agent_api_key: str (encrypted)
    agent_api_provider: str  # "anthropic" | "openai" | "local"
    agent_model: str         # e.g. "claude-sonnet-4-20250514"
    hf_token: str (encrypted)
    hf_username: str
    auto_config_on_upload: bool = True
    allow_agent_overrides: bool = True
    show_agent_reasoning: bool = False
    suggest_base_model: bool = True
    created_at: datetime
    updated_at: datetime

class Dataset(Base):
    id: str (UUID PK)
    name: str
    file_path: str           # local path or MinIO key
    file_type: str           # "csv" | "json" | "jsonl"
    row_count: int
    column_names: str        # JSON serialized list
    schema_info: str         # JSON: column types, sample values
    size_bytes: int
    analysis_result: str     # JSON: agent analysis output
    is_analyzed: bool = False
    created_at: datetime

class Pipeline(Base):
    id: str (UUID PK)
    name: str
    description: str
    dataset_id: str (FK -> Dataset)
    node_graph: str          # JSON: nodes + connections for UI
    config: str              # JSON: all 22 training fields
    is_agent_configured: bool = False
    created_at: datetime
    updated_at: datetime

class TrainingJob(Base):
    id: str (UUID PK)
    pipeline_id: str (FK -> Pipeline)
    status: str  # "queued"|"running"|"completed"|"failed"|"stopped"
    celery_task_id: str
    current_epoch: int = 0
    total_epochs: int = 0
    current_loss: float
    val_loss: float
    progress_pct: float = 0.0
    logs: str = ""           # append-only log text
    error_message: str
    model_output_path: str
    started_at: datetime
    completed_at: datetime
    created_at: datetime

class TrainedModel(Base):
    id: str (UUID PK)
    job_id: str (FK -> TrainingJob)
    local_path: str
    hf_repo_id: str          # null until pushed
    is_pushed_to_hub: bool = False
    push_status: str         # "pending"|"pushing"|"done"|"failed"
    base_model: str
    training_method: str     # "lora"|"qlora"|"full"
    created_at: datetime
```

---

### `backend/app/api/routes/settings.py`

Endpoints:
```
GET  /api/settings              → Return settings (mask keys: show last 4 chars)
POST /api/settings/agent-key    → Store encrypted agent API key
POST /api/settings/hf-token     → Store encrypted HF token
POST /api/settings/verify-agent → Call LLM API with key, return {valid: bool}
POST /api/settings/verify-hf    → Call HF whoami API, return {valid: bool, username: str}
PUT  /api/settings              → Update all non-key settings
```

---

### `backend/app/api/routes/datasets.py`

Endpoints:
```
POST   /api/datasets/upload      → Accept multipart file (CSV/JSON/JSONL)
                                   Save to /uploads/, store metadata in DB
                                   If auto_config_on_upload=True: trigger agent async
GET    /api/datasets             → List all datasets
GET    /api/datasets/{id}        → Dataset metadata + first 10 rows preview
POST   /api/datasets/{id}/analyze → Manually trigger AI agent analysis
DELETE /api/datasets/{id}        → Delete file + DB record
```

Upload logic:
1. Validate file type (csv, json, jsonl only)
2. Save to `UPLOAD_DIR/{uuid}_{filename}`
3. Parse file, count rows, extract column names and types
4. Store schema as JSON in DB
5. If auto_config enabled: call `PipelineAutoConfigAgent.analyze_dataset()` in background

---

### `backend/app/api/routes/pipelines.py`

Endpoints:
```
POST   /api/pipelines              → Create pipeline with dataset_id + name
GET    /api/pipelines              → List all pipelines
GET    /api/pipelines/{id}         → Full pipeline with node_graph + config
PUT    /api/pipelines/{id}         → Update node_graph and/or config (manual override)
DELETE /api/pipelines/{id}         → Delete pipeline
POST   /api/pipelines/{id}/validate → Validate all 22 fields, return errors list
```

The `node_graph` field stores the visual node positions and connections as JSON for the frontend canvas. The `config` field stores the flat 22-field training configuration.

---

### `backend/app/api/routes/jobs.py`

Endpoints:
```
POST /api/jobs/start          → Validate pipeline → dispatch Celery task → return job_id
GET  /api/jobs                → List all jobs with status
GET  /api/jobs/{id}           → Job details: status, metrics, progress
GET  /api/jobs/{id}/logs      → SSE stream of training logs (text/event-stream)
POST /api/jobs/{id}/stop      → Revoke Celery task, set status="stopped"
DELETE /api/jobs/{id}         → Delete job record (only if not running)
```

SSE log streaming implementation:
```python
@router.get("/{job_id}/logs")
async def stream_logs(job_id: str, db: Session = Depends(get_db)):
    async def event_generator():
        last_len = 0
        while True:
            job = crud.get_job(db, job_id)
            if not job:
                break
            logs = job.logs or ""
            if len(logs) > last_len:
                new_lines = logs[last_len:]
                last_len = len(logs)
                yield f"data: {new_lines}\n\n"
            if job.status in ("completed", "failed", "stopped"):
                yield "data: [DONE]\n\n"
                break
            await asyncio.sleep(0.5)
    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

---

### `backend/app/api/routes/models.py`

Endpoints:
```
GET  /api/models               → List all trained models
GET  /api/models/{id}          → Model details
POST /api/models/{id}/export   → Copy model to final local path, create zip
POST /api/models/{id}/push-hub → Push to HF Hub using stored HF token
GET  /api/models/{id}/push-status → Poll push progress
```

---

### `backend/app/agents/pipeline_agent.py`

This is the CORE agent. Implement fully:

```python
class PipelineAutoConfigAgent:
    """
    Analyzes an uploaded dataset and recommends optimal values
    for all 22 pipeline configuration fields.
    
    Uses LangChain with the user's configured LLM (Anthropic/OpenAI/local).
    """
    
    def __init__(self, api_key: str, provider: str, model: str):
        # Initialize LangChain LLM based on provider
        # anthropic → ChatAnthropic(model=model, api_key=api_key)
        # openai → ChatOpenAI(model=model, api_key=api_key)
        # local → ChatOllama(model=model)
        pass

    def analyze_dataset(self, dataset_path: str, schema_info: dict) -> DatasetAnalysis:
        """
        Step 1: Analyze dataset characteristics.
        Returns: row_count, avg_text_length, column_types, 
                 inferred_task_type, language, domain, data_quality_score
        """
        pass

    def recommend_config(self, analysis: DatasetAnalysis) -> PipelineConfig:
        """
        Step 2: Given dataset analysis, call LLM to recommend all 22 fields.
        
        System prompt instructs LLM to act as an ML expert.
        LLM receives: dataset stats, inferred task, hardware context.
        LLM returns: structured JSON with all 22 fields + reasoning for each.
        
        Parse JSON response into PipelineConfig Pydantic model.
        """
        pass

    def run(self, dataset_id: str, db: Session) -> PipelineConfig:
        """
        Full pipeline: load dataset → analyze → recommend → save to pipeline.
        Updates job logs at each step for SSE streaming.
        """
        pass
```

System prompt for the LLM (use exactly):
```
You are an expert ML engineer specializing in LLM fine-tuning.
Given a dataset analysis, recommend optimal training configuration.

Dataset info: {dataset_stats}
Hardware: {hardware_info}

Return ONLY valid JSON with these exact fields:
{
  "project_name": string,
  "task_type": "Classification"|"Chat"|"QA"|"Extraction",
  "output_type": "JSON"|"text"|"label"|"multi-label",
  "domain": "General"|"Finance"|"Medical"|"Legal"|"Code",
  "language": string,
  "training_mode": "fast"|"balanced"|"high_quality",
  "base_model": string (HuggingFace model ID),
  "epochs": integer,
  "batch_size": integer,
  "learning_rate": float,
  "max_seq_len": integer,
  "lora_rank": 8|16|32|64,
  "gradient_accumulation": integer,
  "precision": "bf16"|"fp16"|"float32",
  "early_stopping": boolean,
  "class_balancing": boolean,
  "data_augmentation": boolean,
  "split_ratio": float,
  "reasoning": {
    "base_model": "why this model was chosen",
    "lora_rank": "why this rank",
    "epochs": "why this count"
  }
}
Rules:
- If row_count < 1000: epochs=5, lora_rank=8, batch_size=4
- If row_count 1000-10000: epochs=3, lora_rank=16, batch_size=8
- If row_count > 10000: epochs=2, lora_rank=32, batch_size=16
- If no GPU: base_model=TinyLlama-1.1B, precision=float32, batch_size=2
- If GPU with <8GB VRAM: use QLoRA, precision=fp16
- If GPU with >=8GB VRAM: use LoRA, precision=bf16
```

---

### `backend/app/training/base.py`

```python
from abc import ABC, abstractmethod

class BaseTrainer(ABC):
    def __init__(self, config: PipelineConfig, job_id: str, log_callback):
        self.config = config
        self.job_id = job_id
        self.log = log_callback  # function(msg: str) to append to job logs
    
    @abstractmethod
    def load_model(self): pass
    
    @abstractmethod
    def load_dataset(self): pass
    
    @abstractmethod
    def train(self) -> TrainingResult: pass
    
    @abstractmethod
    def save(self, output_dir: str): pass
    
    def detect_hardware(self) -> dict:
        # Return: {device: "cuda"|"cpu"|"mps", vram_gb: float|None}
        pass
```

---

### `backend/app/training/lora.py`

Implement `LoRATrainer(BaseTrainer)`:
- Load base model with `AutoModelForCausalLM.from_pretrained()`
- Apply PEFT `LoraConfig` with `r=config.lora_rank`, `target_modules=["q_proj","v_proj"]`
- Use `get_peft_model()` to wrap
- Configure `TrainingArguments` from all 22 config fields
- Use HuggingFace `Trainer` with `DataCollatorForLanguageModeling`
- On GPU: use FP16. On CPU: use FP32
- Call `self.log()` at each epoch with loss metrics
- Save with `model.save_pretrained(output_dir)` and `tokenizer.save_pretrained(output_dir)`

---

### `backend/app/training/qlora.py`

Implement `QLoRATrainer(BaseTrainer)`:
- Check GPU availability. If CPU: fall back to `LoRATrainer` (log the fallback reason)
- Load model with `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)`
- Apply `prepare_model_for_kbit_training()`
- Apply LoRA config same as above
- Use `"paged_adamw_8bit"` optimizer
- Otherwise same training loop as LoRATrainer

---

### `backend/app/training/full_finetune.py`

Implement `FullFineTuneTrainer(BaseTrainer)`:
- Load with `AutoModelForCausalLM.from_pretrained()`
- No PEFT — train all parameters
- Standard `TrainingArguments` + `Trainer`
- Use gradient accumulation from config
- Warn in logs if model > 7B params on CPU

---

### `backend/app/workers/training_worker.py`

```python
@celery_app.task(bind=True)
def run_training_job(self, job_id: str, pipeline_config: dict):
    """
    Celery task that:
    1. Updates job status to "running"
    2. Selects trainer based on config.training_method (lora/qlora/full)
    3. Runs trainer.train()
    4. On success: updates status="completed", saves model path
    5. On failure: updates status="failed", saves error_message
    6. Streams logs by appending to job.logs in DB every 500ms
    """
```

---

### `backend/app/services/hf_service.py`

```python
class HuggingFaceService:
    def verify_token(self, token: str) -> dict:
        # Call https://huggingface.co/api/whoami
        # Return {valid: bool, username: str}
    
    def push_model(self, local_path: str, repo_id: str, token: str, job_id: str):
        # Use huggingface_hub.upload_folder()
        # Update TrainedModel.push_status in DB
        # Log progress to job logs
```

---

### `backend/app/utils/crypto.py`

```python
from cryptography.fernet import Fernet

class CryptoService:
    # Use Fernet symmetric encryption
    # Key stored in .env as ENCRYPTION_KEY
    # Auto-generate and save if not set
    
    def encrypt(self, plaintext: str) -> str: ...
    def decrypt(self, ciphertext: str) -> str: ...
    def mask(self, value: str) -> str:
        # Return "****" + last 4 chars
```

---

### `backend/app/utils/gpu.py`

```python
def detect_hardware() -> dict:
    """
    Returns:
    {
      "device": "cuda" | "mps" | "cpu",
      "gpu_name": str | None,
      "vram_gb": float | None,
      "cuda_version": str | None,
      "recommended_trainer": "qlora" | "lora" | "full"
    }
    """
    # Check torch.cuda.is_available()
    # Check torch.backends.mps.is_available() for Apple Silicon
    # If CUDA: get device name and total memory
    # Recommend trainer based on VRAM:
    #   < 6GB → qlora
    #   6-16GB → lora  
    #   > 16GB → full or lora
```

---

### `backend/requirements.txt`

```
fastapi==0.115.0
uvicorn[standard]==0.30.0
pydantic==2.7.0
pydantic-settings==2.3.0
sqlalchemy==2.0.30
alembic==1.13.1
celery==5.4.0
redis==5.0.4
python-multipart==0.0.9
httpx==0.27.0
cryptography==42.0.8
minio==7.2.7

# ML
torch==2.3.0
transformers==4.41.0
peft==0.11.0
datasets==2.19.0
accelerate==0.30.0
bitsandbytes==0.43.1
trl==0.8.6
huggingface-hub==0.23.0

# Agent
langchain==0.2.0
langchain-anthropic==0.1.13
langchain-openai==0.1.8
langchain-community==0.2.0

# Utils
python-dotenv==1.0.1
rich==13.7.1
pandas==2.2.2
numpy==1.26.4
```

---

### `backend/Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
```

---

### `frontend/Dockerfile`

```dockerfile
FROM nginx:alpine
COPY index.html /usr/share/nginx/html/index.html
EXPOSE 80
```

---

### `.env.example`

```env
# Database
DATABASE_URL=sqlite:////data/finetune.db

# MinIO
MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=finetune-datasets

# Redis (for Celery)
REDIS_URL=redis://redis:6379/0

# Storage paths (Docker volumes)
MODEL_OUTPUT_DIR=/models
UPLOAD_DIR=/uploads

# Encryption (auto-generated if blank)
ENCRYPTION_KEY=

# Logging
LOG_LEVEL=INFO
```

---

### `README.md`

Include:
1. Prerequisites: Docker + Docker Compose
2. Quick start:
   ```bash
   git clone https://github.com/yourname/finetune-studio
   cd finetune-studio
   cp .env.example .env
   docker compose up
   # Frontend: http://localhost:3000
   # Backend API: http://localhost:8000
   # MinIO Console: http://localhost:9001
   ```
3. Usage flow:
   - Go to Settings → add Agent API key + HF token
   - Open Playground → upload dataset
   - Agent auto-configures pipeline nodes
   - Override any values manually if needed
   - Click Run Pipeline
   - Monitor logs in real-time
   - Export model locally or push to HF Hub
4. GPU support note (CUDA passthrough for Docker)
5. Local dev setup (without Docker)

---

## API RESPONSE FORMAT CONVENTIONS

All responses follow:
```json
{
  "success": true,
  "data": { ... },
  "message": "optional human-readable string"
}
```

Errors:
```json
{
  "success": false,
  "error": "SHORT_ERROR_CODE",
  "message": "Human readable explanation"
}
```

HTTP status codes:
- 200: success
- 201: created
- 400: validation error
- 404: not found
- 422: unprocessable entity (Pydantic)
- 500: internal server error

---

## VALIDATION RULES (22 FIELDS)

Implement in `app/api/schemas/pipeline.py`:

```python
class PipelineConfig(BaseModel):
    # Screen 1
    project_name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    tags: List[str] = Field(default_factory=list)
    
    # Screen 2
    dataset_name: str
    target_column: str
    input_columns: List[str] = Field(..., min_items=1)
    split_ratio: float = Field(0.8, ge=0.5, le=0.95)
    
    # Screen 3
    task_type: Literal["Classification","Chat","QA","Extraction"]
    output_type: Literal["JSON","text","label","multi-label"]
    domain: Literal["General","Finance","Medical","Legal","Code"]
    language: str = "en"
    
    # Screen 4
    training_mode: Literal["fast","balanced","high_quality"] = "balanced"
    base_model: str
    epochs: int = Field(3, ge=1, le=100)
    batch_size: int = Field(4, ge=1, le=128)
    learning_rate: float = Field(2e-4, ge=1e-6, le=1e-1)
    max_seq_len: int = Field(512, ge=64, le=8192)
    lora_rank: Literal[8,16,32,64] = 16
    
    # Screen 5
    gradient_accumulation: int = Field(4, ge=1, le=64)
    precision: Literal["bf16","fp16","float32"] = "fp16"
    early_stopping: bool = True
    class_balancing: bool = False
    data_augmentation: bool = False
    resume_checkpoint: Optional[str] = None
    
    # Meta (set by agent)
    training_method: Literal["lora","qlora","full"] = "lora"
```

---

## IMPORTANT IMPLEMENTATION NOTES

1. **Never store raw API keys** — always encrypt with Fernet before saving to SQLite
2. **Training runs async** — always dispatch to Celery, never block the API thread
3. **SSE logs** — append to `job.logs` string in DB every ~500ms during training; the SSE endpoint polls DB
4. **GPU fallback** — if user requests QLoRA but no GPU, silently fall back to LoRA + log the reason
5. **Agent is optional** — if no API key is set, all pipeline fields default to sensible values; agent section just stays disabled
6. **HF push is optional** — only enabled if HF token is verified
7. **Local model path** — always save to `MODEL_OUTPUT_DIR/{job_id}/` so Docker volume persists it
8. **CORS** — allow all origins (this is a local tool, security is not a concern)
9. **No authentication** — this is a single-user local tool; no login/JWT needed
10. **Dataset preview** — when returning dataset details, include first 10 rows as JSON array

---

## TESTING

Write tests in `backend/tests/`:

`test_agent.py`:
- Test `analyze_dataset()` with a sample CSV
- Test `recommend_config()` with mock LLM response
- Test config validation against all 22 fields

`test_training.py`:
- Test `LoRATrainer` with TinyLlama on a 100-row mock dataset
- Test `QLoRATrainer` CPU fallback behavior
- Test model save/load roundtrip

`test_api.py`:
- Test all endpoints with `TestClient`
- Test dataset upload + analysis flow
- Test job start → status → logs → complete flow

---

## DELIVERABLE CHECKLIST

After generating all files, confirm:
- [ ] `docker compose up` starts all 4 services (backend, frontend, redis, minio)
- [ ] `GET /health` returns 200 with hardware info
- [ ] Dataset upload endpoint accepts CSV and returns preview
- [ ] Agent runs and returns structured 22-field config
- [ ] Training job dispatches to Celery and streams logs via SSE
- [ ] Model saves to `/models/{job_id}/` on completion
- [ ] HF push works with valid token
- [ ] All API keys encrypted at rest in SQLite
- [ ] Frontend served at `http://localhost:3000`
- [ ] Backend API docs at `http://localhost:8000/docs`
