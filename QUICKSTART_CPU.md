# Quick Start Guide - Fixed for CPU

## ✅ Issue Fixed

The NVIDIA GPU error has been resolved! The docker-compose.yml has been updated to work on CPU-only systems.

## Starting the Platform

### 1. Start Backend Services

```bash
cd d:\Projects\NoCode-Back
docker-compose up -d
```

**Services Started**:
- ✅ Redis (port 6379)
- ✅ MinIO (ports 9000, 9001)
- ✅ API (port 8000)
- ✅ Worker (CPU mode)

### 2. Verify Services

```bash
docker-compose ps
```

All containers should show "Up" status.

### 3. Test API

Open browser to: **http://localhost:8000/docs**

Or test health endpoint:
```bash
# PowerShell
Invoke-WebRequest http://localhost:8000/health

# Or use browser
http://localhost:8000/health
```

### 4. Start Frontend

```bash
cd llm-platform-ui
npm install
npm run dev
```

Opens at: **http://localhost:3000**

## What Changed

**Fixed GPU Error**: Commented out NVIDIA GPU requirements in `docker-compose.yml`:

```yaml
# GPU support disabled for CPU-only systems
# Uncomment below if you have NVIDIA GPU available:
# deploy:
#   resources:
#     reservations:
#       devices:
#         - driver: nvidia
#           count: all
#           capabilities: [gpu]
```

## Training on CPU

All training code automatically detects CPU and adjusts:
- Uses `float32` precision (instead of `float16`)
- Disables FP16 training
- QLoRA falls back to regular LoRA
- Training will be slower but fully functional

## Next Steps

1. ✅ Backend running on CPU
2. 🚀 Start frontend: `cd llm-platform-ui && npm run dev`
3. 📊 Open http://localhost:3000
4. 🎨 Build your first pipeline in Playground
5. 🔍 Monitor jobs in real-time

## Troubleshooting

**If containers won't start**:
```bash
docker-compose down
docker-compose up -d
```

**Check logs**:
```bash
docker logs llm-platform-api
docker logs llm-platform-worker
```

**Reset everything**:
```bash
docker-compose down -v
docker-compose up -d
```
