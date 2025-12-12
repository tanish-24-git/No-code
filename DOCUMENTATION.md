# LLM Fine-Tuning Platform - Complete Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Component Details](#component-details)
4. [Data Flow](#data-flow)
5. [API Reference](#api-reference)
6. [Deployment Guide](#deployment-guide)
7. [Development Guide](#development-guide)

---

## System Overview

### What is This Platform?

An **industry-scale, agent-based backend system** that enables users to fine-tune Large Language Models (LLMs) through visual drag-and-drop pipelines with real-time monitoring.

### Key Features

- 🎯 **Drag-and-Drop Pipelines**: Visual configuration of training workflows
- 🔄 **Agent-Based Architecture**: Modular, stateless execution units
- 📊 **Real-Time Monitoring**: Live training logs via Server-Sent Events
- 🚀 **GPU-Accelerated**: Automatic GPU detection with CPU fallback
- 💾 **Object Storage**: S3-compatible storage for datasets and models
- 🔁 **Fault Tolerant**: Automatic retries with exponential backoff
- 📦 **Docker-Ready**: Complete containerized deployment

### Use Cases

1. **Fine-tune LLMs** on custom datasets (LoRA, QLoRA, Full)
2. **Validate datasets** for quality and PII detection
3. **Preprocess text** with LLM-native cleaning and formatting
4. **Evaluate models** with task-specific metrics
5. **Export models** in multiple formats (adapters, merged, GGUF)

---

## Architecture

### High-Level Architecture

```mermaid
graph TB
    subgraph "Client Layer"
        UI[Web UI / API Client]
    end
    
    subgraph "API Layer"
        API[FastAPI Application]
        SSE[SSE Endpoint]
    end
    
    subgraph "Orchestration Layer"
        ORC[Orchestrator Agent]
        QUEUE[Task Queue - RQ]
    end
    
    subgraph "Agent Layer"
        DA[Dataset Agent]
        VA[Validation Agent]
        PA[Preprocessing Agent]
        TA[Training Agent]
        EA[Evaluation Agent]
        EXA[Export Agent]
    end
    
    subgraph "Infrastructure Layer"
        REDIS[(Redis)]
        POSTGRES[(PostgreSQL)]
        MINIO[(MinIO - S3)]
        GPU[GPU Manager]
    end
    
    subgraph "Worker Layer"
        W1[Worker 1 - GPU]
        W2[Worker 2 - GPU]
        W3[Worker N - GPU]
    end
    
    UI --> API
    API --> QUEUE
    API --> SSE
    SSE --> REDIS
    
    QUEUE --> ORC
    ORC --> DA
    ORC --> VA
    ORC --> PA
    ORC --> TA
    ORC --> EA
    ORC --> EXA
    
    DA --> MINIO
    VA --> MINIO
    PA --> MINIO
    TA --> MINIO
    TA --> GPU
    EA --> MINIO
    EXA --> MINIO
    
    W1 --> QUEUE
    W2 --> QUEUE
    W3 --> QUEUE
    
    API --> REDIS
    API --> POSTGRES
    ORC --> REDIS
    DA --> REDIS
    VA --> REDIS
    PA --> REDIS
    TA --> REDIS
    EA --> REDIS
    EXA --> REDIS
    
    style UI fill:#e1f5ff
    style API fill:#fff3e0
    style ORC fill:#f3e5f5
    style REDIS fill:#ffebee
    style MINIO fill:#e8f5e9
    style GPU fill:#fff9c4
```

### System Components

#### 1. **Client Layer**
- Web UI or API clients
- Submits pipeline configurations
- Consumes real-time logs via SSE

#### 2. **API Layer**
- **FastAPI Application**: REST API endpoints
- **SSE Endpoint**: Real-time log streaming
- **CORS Middleware**: Cross-origin support
- **Exception Handlers**: Structured error responses

#### 3. **Orchestration Layer**
- **Orchestrator Agent**: DAG execution engine
- **Task Queue (RQ)**: Background job management
- **Redis**: Job state and message queue

#### 4. **Agent Layer**
- **Dataset Agent**: File ingestion and analysis
- **Validation Agent**: Data quality checks
- **Preprocessing Agent**: LLM-native text processing
- **Training Agent**: Model fine-tuning
- **Evaluation Agent**: Metric computation
- **Export Agent**: Model export in multiple formats

#### 5. **Infrastructure Layer**
- **Redis**: State, queue, log streaming
- **PostgreSQL**: Metadata persistence
- **MinIO**: S3-compatible object storage
- **GPU Manager**: Device allocation and monitoring

#### 6. **Worker Layer**
- **GPU Workers**: Execute training jobs
- **CPU Workers**: Execute non-GPU tasks
- **Auto-scaling**: Add workers as needed

---

## Component Details

### Agent Framework

#### Base Agent Architecture

```mermaid
classDiagram
    class BaseAgent {
        <<abstract>>
        +agent_name: str
        +logger: StructuredLogger
        +log_stream: LogStream
        +execute(input_data)* Dict
        +run(input_data) Dict
        +validate_input(data, schema) BaseModel
        -_emit_log(run_id, level, message)
        -_set_state(run_id, state)
        -_get_state(run_id) Dict
    }
    
    class DatasetAgent {
        +execute(input_data) Dict
        -_get_local_path(file_path) Path
        -_infer_format(file_path) str
        -_load_dataset(path, format) DataFrame
        -_analyze_dataset(df) Dict
    }
    
    class ValidationAgent {
        +pii_patterns: Dict
        +execute(input_data) Dict
        -_check_missing_values(df) Dict
        -_check_duplicates(df) Dict
        -_check_pii(df) Dict
        -_check_text_quality(df) Dict
    }
    
    class PreprocessingAgent {
        +execute(input_data) Dict
        -_apply_template(df, config) DataFrame
        -_chunk_dataset(df, config) DataFrame
        -_save_processed_dataset(df) str
    }
    
    class OrchestratorAgent {
        +agent_registry: Dict
        +execute(input_data) Dict
        +register_agent(agent_class, name)
        -_execute_agent(node, input) Dict
        -_build_graph(nodes, edges) Dict
        -_topological_sort(graph) List
    }
    
    BaseAgent <|-- DatasetAgent
    BaseAgent <|-- ValidationAgent
    BaseAgent <|-- PreprocessingAgent
    BaseAgent <|-- OrchestratorAgent
```

#### Agent Lifecycle

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant Queue
    participant Worker
    participant Agent
    participant Redis
    participant MinIO
    
    Client->>API: Submit Pipeline
    API->>Queue: Enqueue Job
    API-->>Client: Job ID
    
    Queue->>Worker: Assign Job
    Worker->>Agent: Initialize
    Agent->>Redis: Set State (running)
    Agent->>Redis: Publish Log (started)
    
    Agent->>MinIO: Download Data
    Agent->>Agent: Execute Logic
    Agent->>Redis: Publish Logs (progress)
    
    Agent->>MinIO: Upload Results
    Agent->>Redis: Set State (completed)
    Agent->>Redis: Publish Log (finished)
    
    Agent-->>Worker: Return Output
    Worker-->>Queue: Job Complete
    
    Client->>API: Stream Logs (SSE)
    API->>Redis: Read Stream
    Redis-->>API: Log Events
    API-->>Client: SSE Events
```

---

## Data Flow

### Complete Pipeline Flow

```mermaid
flowchart TD
    START([User Uploads Dataset]) --> UPLOAD[Upload to MinIO]
    UPLOAD --> DATASET[Dataset Agent]
    
    DATASET --> |Analyze| DATASET_OUT{Dataset Valid?}
    DATASET_OUT -->|Yes| VALIDATE[Validation Agent]
    DATASET_OUT -->|No| ERROR1[Return Error]
    
    VALIDATE --> |Check Quality| VALIDATE_OUT{Passes Validation?}
    VALIDATE_OUT -->|Yes| PREPROCESS[Preprocessing Agent]
    VALIDATE_OUT -->|No| WARN[Return Warnings]
    
    PREPROCESS --> |Clean & Format| PREPROCESS_OUT[Processed Dataset]
    PREPROCESS_OUT --> TRAIN[Training Agent]
    
    TRAIN --> |Fine-tune| TRAIN_OUT{Training Success?}
    TRAIN_OUT -->|Yes| EVAL[Evaluation Agent]
    TRAIN_OUT -->|No| ERROR2[Return Error]
    
    EVAL --> |Compute Metrics| EVAL_OUT[Evaluation Results]
    EVAL_OUT --> COMPARE[Comparison Agent]
    
    COMPARE --> |Base vs Fine-tuned| COMPARE_OUT[Comparison Report]
    COMPARE_OUT --> EXPORT[Export Agent]
    
    EXPORT --> |Generate Artifacts| EXPORT_OUT[Exported Models]
    EXPORT_OUT --> END([Pipeline Complete])
    
    ERROR1 --> END
    WARN --> PREPROCESS
    ERROR2 --> END
    
    style START fill:#e1f5ff
    style END fill:#c8e6c9
    style ERROR1 fill:#ffcdd2
    style ERROR2 fill:#ffcdd2
    style WARN fill:#fff9c4
```

### Dataset Agent Flow

```mermaid
flowchart LR
    INPUT[Input: file_path, format] --> CHECK{S3 Path?}
    CHECK -->|Yes| DOWNLOAD[Download from MinIO]
    CHECK -->|No| LOCAL[Use Local Path]
    
    DOWNLOAD --> INFER[Infer Format]
    LOCAL --> INFER
    
    INFER --> LOAD[Load Dataset]
    LOAD --> ANALYZE[Analyze Schema]
    
    ANALYZE --> DETECT[Detect Text Columns]
    DETECT --> STATS[Compute Statistics]
    
    STATS --> OUTPUT[Output: dataset_id, stats, columns]
    
    style INPUT fill:#e3f2fd
    style OUTPUT fill:#c8e6c9
```

### Validation Agent Flow

```mermaid
flowchart TD
    INPUT[Input: dataset_id, local_path] --> LOAD[Load Dataset]
    
    LOAD --> CHECK1[Check Missing Values]
    CHECK1 --> CHECK2[Check Duplicates]
    CHECK2 --> CHECK3[Check PII]
    CHECK3 --> CHECK4[Check Text Quality]
    
    CHECK1 --> |Missing Found| WARN1[Add Warning]
    CHECK2 --> |Duplicates Found| WARN2[Add Warning]
    CHECK3 --> |PII Found| WARN3[Add Warning]
    CHECK4 --> |Low Quality| WARN4[Add Warning]
    
    CHECK4 --> DECIDE{Has Errors?}
    DECIDE -->|Yes| INVALID[valid: false]
    DECIDE -->|No| VALID[valid: true]
    
    WARN1 --> COLLECT[Collect Warnings]
    WARN2 --> COLLECT
    WARN3 --> COLLECT
    WARN4 --> COLLECT
    
    INVALID --> OUTPUT[Output: validation_report]
    VALID --> OUTPUT
    COLLECT --> OUTPUT
    
    style INPUT fill:#e3f2fd
    style OUTPUT fill:#c8e6c9
    style INVALID fill:#ffcdd2
    style VALID fill:#c8e6c9
```

### Preprocessing Agent Flow

```mermaid
flowchart TD
    INPUT[Input: dataset, config] --> LOAD[Load Dataset]
    
    LOAD --> CLEAN{Clean Enabled?}
    CLEAN -->|Yes| DO_CLEAN[Clean Text]
    CLEAN -->|No| DEDUP
    DO_CLEAN --> DEDUP
    
    DEDUP{Dedup Enabled?}
    DEDUP -->|Yes| DO_DEDUP[Remove Duplicates]
    DEDUP -->|No| TEMPLATE
    DO_DEDUP --> TEMPLATE
    
    TEMPLATE{Template Specified?}
    TEMPLATE -->|Yes| APPLY_TEMPLATE[Apply Prompt Template]
    TEMPLATE -->|No| CHUNK
    APPLY_TEMPLATE --> CHUNK
    
    CHUNK{Chunk Enabled?}
    CHUNK -->|Yes| DO_CHUNK[Chunk Long Texts]
    CHUNK -->|No| TOKENIZE
    DO_CHUNK --> TOKENIZE
    
    TOKENIZE[Count Tokens] --> SAVE[Save to MinIO]
    SAVE --> OUTPUT[Output: processed_path, stats]
    
    style INPUT fill:#e3f2fd
    style OUTPUT fill:#c8e6c9
```

### Real-Time Log Streaming Flow

```mermaid
sequenceDiagram
    participant Agent
    participant Redis
    participant API
    participant Client
    
    Note over Agent,Client: Training in Progress
    
    loop Every Training Step
        Agent->>Redis: XADD logs:run_123 {step, loss, ...}
        Redis-->>Agent: ACK
    end
    
    Client->>API: GET /logs/stream/run_123
    API->>Redis: XREAD logs:run_123
    
    loop Stream Events
        Redis-->>API: Log Event
        API-->>Client: data: {event}\n\n
        
        Agent->>Redis: XADD logs:run_123 {new event}
        Redis-->>API: New Event
        API-->>Client: data: {new event}\n\n
    end
    
    Note over Agent: Training Complete
    Agent->>Redis: XADD logs:run_123 {status: completed}
    Redis-->>API: Final Event
    API-->>Client: data: {completed}\n\n
    Client->>API: Close Connection
```

---

## Infrastructure Details

### Redis Architecture

```mermaid
graph TB
    subgraph "Redis Usage"
        QUEUE[Task Queue - RQ]
        STATE[Job & Agent State]
        LOGS[Log Streams]
        CACHE[Tokenizer Cache]
    end
    
    subgraph "Data Structures"
        HASH[Hashes - job:*, agent:*]
        STREAM[Streams - logs:*]
        LIST[Lists - rq:queue:*]
    end
    
    QUEUE --> LIST
    STATE --> HASH
    LOGS --> STREAM
    CACHE --> HASH
    
    style QUEUE fill:#ffebee
    style STATE fill:#e8f5e9
    style LOGS fill:#e3f2fd
    style CACHE fill:#fff9c4
```

### Object Storage (MinIO) Structure

```mermaid
graph TB
    MINIO[MinIO Object Storage]
    
    MINIO --> DATASETS[Bucket: datasets]
    MINIO --> MODELS[Bucket: models]
    MINIO --> CHECKPOINTS[Bucket: checkpoints]
    MINIO --> ARTIFACTS[Bucket: artifacts]
    
    DATASETS --> DS1[run_123/dataset.csv]
    DATASETS --> DS2[run_123/processed_dataset.jsonl]
    
    MODELS --> M1[run_123/final/]
    MODELS --> M2[run_123/model_metadata.json]
    
    CHECKPOINTS --> C1[run_123/checkpoint-1000/]
    CHECKPOINTS --> C2[run_123/checkpoint-2000/]
    
    ARTIFACTS --> A1[run_123/adapter/]
    ARTIFACTS --> A2[run_123/merged/]
    ARTIFACTS --> A3[run_123/model.gguf]
    
    style MINIO fill:#e8f5e9
    style DATASETS fill:#e3f2fd
    style MODELS fill:#fff9c4
    style CHECKPOINTS fill:#f3e5f5
    style ARTIFACTS fill:#ffe0b2
```

### GPU Management Flow

```mermaid
flowchart TD
    START[GPU Manager Init] --> DETECT{CUDA Available?}
    
    DETECT -->|Yes| COUNT[Count GPUs]
    DETECT -->|No| CPU[Set CPU Mode]
    
    COUNT --> LOG[Log GPU Info]
    LOG --> READY[GPU Ready]
    
    CPU --> READY
    
    READY --> REQUEST[Training Request]
    REQUEST --> ALLOCATE{GPU Available?}
    
    ALLOCATE -->|Yes| ASSIGN[Assign GPU Device]
    ALLOCATE -->|No| QUEUE_GPU[Queue for GPU]
    
    ASSIGN --> TRAIN[Start Training]
    QUEUE_GPU --> WAIT[Wait for GPU]
    WAIT --> ALLOCATE
    
    TRAIN --> MONITOR[Monitor Memory]
    MONITOR --> COMPLETE{Training Done?}
    
    COMPLETE -->|No| MONITOR
    COMPLETE -->|Yes| RELEASE[Release GPU]
    
    RELEASE --> CLEAR[Clear Cache]
    CLEAR --> END[GPU Available]
    
    style START fill:#e3f2fd
    style READY fill:#c8e6c9
    style TRAIN fill:#fff9c4
    style END fill:#c8e6c9
```

---

## API Reference

### Endpoint Structure

```mermaid
graph LR
    ROOT[/] --> HEALTH[/health]
    ROOT --> API[/api/v1]
    
    API --> JOBS[/jobs]
    API --> DATASETS[/datasets]
    API --> MODELS[/models]
    API --> LOGS[/logs]
    
    JOBS --> J1[POST /jobs]
    JOBS --> J2[GET /jobs/:id]
    JOBS --> J3[DELETE /jobs/:id]
    
    DATASETS --> D1[POST /datasets/upload]
    DATASETS --> D2[GET /datasets/:id]
    
    MODELS --> M1[GET /models]
    MODELS --> M2[GET /models/:id]
    MODELS --> M3[GET /models/:id/download]
    
    LOGS --> L1[GET /logs/stream/:run_id]
    LOGS --> L2[GET /logs/history/:run_id]
    
    style ROOT fill:#e3f2fd
    style API fill:#fff9c4
```

### Request/Response Flow

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant Validator
    participant Queue
    participant Redis
    
    Client->>API: POST /api/v1/jobs
    Note over Client,API: {pipeline_config}
    
    API->>Validator: Validate Schema
    Validator-->>API: Valid ✓
    
    API->>Queue: Enqueue Job
    Queue-->>API: job_id
    
    API->>Redis: Set Job State
    Redis-->>API: OK
    
    API-->>Client: 202 Accepted
    Note over API,Client: {job_id, status: pending}
    
    Client->>API: GET /api/v1/jobs/:id
    API->>Redis: Get Job State
    Redis-->>API: {status, progress}
    API-->>Client: 200 OK
```

---

## Deployment Guide

### Docker Compose Deployment

```mermaid
graph TB
    subgraph "Docker Host"
        subgraph "llm-platform Network"
            REDIS_C[redis:6379]
            POSTGRES_C[postgres:5432]
            MINIO_C[minio:9000/9001]
            API_C[api:8000]
            WORKER_C[worker]
        end
        
        subgraph "Volumes"
            V1[redis_data]
            V2[postgres_data]
            V3[minio_data]
        end
        
        subgraph "GPU"
            GPU_DEV[/dev/nvidia0]
        end
    end
    
    REDIS_C --> V1
    POSTGRES_C --> V2
    MINIO_C --> V3
    WORKER_C --> GPU_DEV
    
    API_C --> REDIS_C
    API_C --> POSTGRES_C
    API_C --> MINIO_C
    
    WORKER_C --> REDIS_C
    WORKER_C --> POSTGRES_C
    WORKER_C --> MINIO_C
    
    style REDIS_C fill:#ffebee
    style POSTGRES_C fill:#e3f2fd
    style MINIO_C fill:#e8f5e9
    style API_C fill:#fff9c4
    style WORKER_C fill:#f3e5f5
```

### Deployment Steps

```mermaid
flowchart TD
    START([Start Deployment]) --> ENV[Copy .env.example to .env]
    ENV --> CONFIG[Configure Environment Variables]
    CONFIG --> BUILD[docker-compose build]
    
    BUILD --> UP[docker-compose up -d]
    UP --> WAIT[Wait for Health Checks]
    
    WAIT --> CHECK{All Healthy?}
    CHECK -->|No| LOGS[Check Logs]
    CHECK -->|Yes| VERIFY[Verify Services]
    
    LOGS --> FIX[Fix Issues]
    FIX --> UP
    
    VERIFY --> TEST1[curl /health]
    TEST1 --> TEST2[Access MinIO Console]
    TEST2 --> TEST3[Check API Docs]
    
    TEST3 --> READY([Deployment Ready])
    
    style START fill:#e3f2fd
    style READY fill:#c8e6c9
    style LOGS fill:#ffcdd2
```

---

## Development Guide

### Local Development Setup

```mermaid
flowchart TD
    START([Clone Repository]) --> VENV[Create Virtual Environment]
    VENV --> INSTALL[pip install -r requirements.txt]
    
    INSTALL --> REDIS[Start Redis]
    REDIS --> MINIO[Start MinIO]
    MINIO --> POSTGRES[Start PostgreSQL]
    
    POSTGRES --> ENV[Set Environment Variables]
    ENV --> API[Run API: python -m app.main]
    
    API --> WORKER[Run Worker: rq worker]
    
    WORKER --> DEV([Development Ready])
    
    style START fill:#e3f2fd
    style DEV fill:#c8e6c9
```

### Testing Strategy

```mermaid
graph TB
    subgraph "Test Pyramid"
        E2E[End-to-End Tests]
        INTEGRATION[Integration Tests]
        UNIT[Unit Tests]
    end
    
    UNIT --> U1[Agent Logic]
    UNIT --> U2[Preprocessing Utils]
    UNIT --> U3[Validation Logic]
    
    INTEGRATION --> I1[Agent Pipeline]
    INTEGRATION --> I2[Redis Integration]
    INTEGRATION --> I3[MinIO Integration]
    
    E2E --> E1[Complete Pipeline]
    E2E --> E2[API Endpoints]
    
    style UNIT fill:#c8e6c9
    style INTEGRATION fill:#fff9c4
    style E2E fill:#e3f2fd
```

---

## Performance & Scaling

### Horizontal Scaling

```mermaid
graph TB
    LB[Load Balancer] --> API1[API Instance 1]
    LB --> API2[API Instance 2]
    LB --> API3[API Instance N]
    
    API1 --> REDIS
    API2 --> REDIS
    API3 --> REDIS
    
    REDIS --> W1[Worker 1 - GPU 0]
    REDIS --> W2[Worker 2 - GPU 1]
    REDIS --> W3[Worker 3 - GPU 2]
    REDIS --> W4[Worker N - GPU N]
    
    W1 --> MINIO
    W2 --> MINIO
    W3 --> MINIO
    W4 --> MINIO
    
    style LB fill:#e3f2fd
    style REDIS fill:#ffebee
    style MINIO fill:#e8f5e9
```

### Performance Metrics

```mermaid
graph LR
    subgraph "Metrics to Monitor"
        M1[API Latency]
        M2[Queue Depth]
        M3[GPU Utilization]
        M4[Memory Usage]
        M5[Training Speed]
        M6[Log Throughput]
    end
    
    M1 --> TARGET1[< 100ms]
    M2 --> TARGET2[< 10 jobs]
    M3 --> TARGET3[> 80%]
    M4 --> TARGET4[< 90%]
    M5 --> TARGET5[tokens/sec]
    M6 --> TARGET6[events/sec]
    
    style M1 fill:#e3f2fd
    style M2 fill:#fff9c4
    style M3 fill:#c8e6c9
```

---

## Security Considerations

### Security Layers

```mermaid
graph TB
    subgraph "Security Measures"
        AUTH[Authentication]
        VALID[Input Validation]
        RATE[Rate Limiting]
        ENCRYPT[Encryption]
        AUDIT[Audit Logging]
    end
    
    AUTH --> A1[API Keys]
    AUTH --> A2[JWT Tokens]
    
    VALID --> V1[Pydantic Schemas]
    VALID --> V2[File Type Checks]
    VALID --> V3[Size Limits]
    
    RATE --> R1[Per-IP Limits]
    RATE --> R2[Per-User Limits]
    
    ENCRYPT --> E1[TLS/HTTPS]
    ENCRYPT --> E2[Encrypted Storage]
    
    AUDIT --> AU1[Request Logs]
    AUDIT --> AU2[Access Logs]
    
    style AUTH fill:#ffebee
    style VALID fill:#e3f2fd
    style RATE fill:#fff9c4
    style ENCRYPT fill:#e8f5e9
    style AUDIT fill:#f3e5f5
```

---

## Troubleshooting

### Common Issues Flow

```mermaid
flowchart TD
    ISSUE[Issue Reported] --> TYPE{Issue Type?}
    
    TYPE -->|Connection| CONN[Check Services]
    TYPE -->|Performance| PERF[Check Resources]
    TYPE -->|Error| ERR[Check Logs]
    
    CONN --> C1{Redis Up?}
    C1 -->|No| FIX1[Restart Redis]
    C1 -->|Yes| C2{MinIO Up?}
    C2 -->|No| FIX2[Restart MinIO]
    C2 -->|Yes| C3[Check Network]
    
    PERF --> P1{GPU Available?}
    P1 -->|No| FIX3[Check nvidia-docker]
    P1 -->|Yes| P2{Memory OK?}
    P2 -->|No| FIX4[Scale Workers]
    P2 -->|Yes| P3[Optimize Config]
    
    ERR --> E1[View Container Logs]
    E1 --> E2[Check Redis Logs]
    E2 --> E3[Check Worker Logs]
    E3 --> FIX5[Fix Root Cause]
    
    FIX1 --> VERIFY[Verify Fix]
    FIX2 --> VERIFY
    FIX3 --> VERIFY
    FIX4 --> VERIFY
    FIX5 --> VERIFY
    C3 --> VERIFY
    P3 --> VERIFY
    
    VERIFY --> RESOLVED([Issue Resolved])
    
    style ISSUE fill:#ffcdd2
    style RESOLVED fill:#c8e6c9
```

---

## Summary

This documentation provides a complete overview of the LLM Fine-Tuning Platform, including:

✅ **Architecture Diagrams** - High-level and component-level views  
✅ **Flow Diagrams** - Data flow, agent lifecycle, deployment  
✅ **Sequence Diagrams** - Request/response, log streaming  
✅ **Component Details** - Class diagrams, infrastructure  
✅ **API Reference** - Endpoint structure and flows  
✅ **Deployment Guide** - Docker setup and steps  
✅ **Development Guide** - Local setup and testing  
✅ **Performance & Scaling** - Horizontal scaling strategies  
✅ **Security** - Multi-layer security approach  
✅ **Troubleshooting** - Common issues and solutions

For more details, see:
- [README.md](./README.md) - Quick start guide
- [walkthrough.md](../walkthrough.md) - Implementation details
- [API Documentation](http://localhost:8000/docs) - Interactive API docs
