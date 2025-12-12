# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Prerequisites
- Docker & Docker Compose installed
- (Optional) NVIDIA GPU with nvidia-docker2 for training

---

## Step 1: Clone & Setup

```bash
# Navigate to project directory
cd NoCode-Back

# Copy environment template
cp .env.example .env

# (Optional) Edit .env for custom configuration
nano .env
```

---

## Step 2: Start Services

```bash
# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# Expected output:
# NAME                      STATUS
# llm-platform-api          Up (healthy)
# llm-platform-redis        Up (healthy)
# llm-platform-postgres     Up (healthy)
# llm-platform-minio        Up (healthy)
# llm-platform-worker       Up
```

---

## Step 3: Verify Installation

### Check API Health
```bash
curl http://localhost:8000/health
```

**Expected Response:**
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

### Access Services

| Service | URL | Credentials |
|---------|-----|-------------|
| API Docs | http://localhost:8000/docs | - |
| MinIO Console | http://localhost:9001 | minioadmin / minioadmin |
| API | http://localhost:8000 | - |

---

## Step 4: Test the System

### View API Documentation
Open your browser: http://localhost:8000/docs

You'll see interactive Swagger UI with all endpoints.

### Check MinIO Buckets
1. Open http://localhost:9001
2. Login: `minioadmin` / `minioadmin`
3. Verify buckets exist:
   - `datasets`
   - `models`
   - `checkpoints`
   - `artifacts`

---

## Step 5: Run Your First Pipeline (Coming Soon)

Once training agents are implemented:

```bash
# Upload dataset
curl -X POST http://localhost:8000/api/v1/datasets/upload \
  -F "file=@training_data.csv"

# Submit pipeline job
curl -X POST http://localhost:8000/api/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "pipeline_config": {
      "nodes": [...],
      "edges": [...]
    }
  }'

# Stream logs
curl http://localhost:8000/api/v1/logs/stream/{run_id}
```

---

## Common Commands

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api
docker-compose logs -f worker
docker-compose logs -f redis
```

### Restart Services
```bash
# Restart all
docker-compose restart

# Restart specific service
docker-compose restart api
docker-compose restart worker
```

### Stop Services
```bash
# Stop all
docker-compose down

# Stop and remove volumes (clean slate)
docker-compose down -v
```

### Rebuild After Code Changes
```bash
# Rebuild and restart
docker-compose up -d --build
```

---

## Troubleshooting

### Services Not Starting

**Check logs:**
```bash
docker-compose logs redis
docker-compose logs postgres
docker-compose logs minio
```

**Common fixes:**
```bash
# Restart Docker daemon
sudo systemctl restart docker

# Clean up and restart
docker-compose down -v
docker-compose up -d
```

### GPU Not Detected

**Check nvidia-docker:**
```bash
# Test GPU access
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

**If fails:**
```bash
# Install nvidia-docker2
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### Port Already in Use

**Change ports in docker-compose.yml:**
```yaml
services:
  api:
    ports:
      - "8001:8000"  # Changed from 8000:8000
```

### MinIO Access Denied

**Reset credentials:**
```bash
# Edit .env
MINIO_ACCESS_KEY=your_access_key
MINIO_SECRET_KEY=your_secret_key

# Restart
docker-compose down -v
docker-compose up -d
```

---

## Next Steps

1. **Read Documentation**: [DOCUMENTATION.md](./DOCUMENTATION.md)
2. **API Reference**: [API_SPEC.md](./API_SPEC.md)
3. **Implementation Details**: [walkthrough.md](../walkthrough.md)
4. **Contribute**: See development guide in README.md

---

## Development Mode

### Run Locally (without Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Start Redis
redis-server

# Start MinIO
minio server /data --console-address ":9001"

# Start PostgreSQL
# (Use Docker or local installation)

# Set environment variables
export REDIS_URL=redis://localhost:6379/0
export MINIO_ENDPOINT=localhost:9000
# ... other vars from .env

# Run API
python -m app.main

# Run worker (separate terminal)
rq worker --url redis://localhost:6379/0 training evaluation orchestration default
```

---

## Configuration Options

### Environment Variables

Edit `.env` file:

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=true

# GPU Configuration
GPU_ENABLED=true
CUDA_VISIBLE_DEVICES=0

# Training Defaults
DEFAULT_LORA_R=16
DEFAULT_LORA_ALPHA=32
DEFAULT_BATCH_SIZE=4
DEFAULT_LEARNING_RATE=2e-4

# Redis
REDIS_URL=redis://redis:6379/0

# MinIO
MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
```

---

## Health Checks

### Check All Services

```bash
# API
curl http://localhost:8000/health

# Redis
docker exec llm-platform-redis redis-cli ping

# PostgreSQL
docker exec llm-platform-postgres pg_isready

# MinIO
curl http://localhost:9000/minio/health/live
```

---

## Support

- **Documentation**: See [DOCUMENTATION.md](./DOCUMENTATION.md)
- **Issues**: Create GitHub issue
- **Questions**: Check API docs at `/docs`

---

**You're all set! 🎉**

The platform is ready for development. Next priority is implementing training agents.
