# FineTune Studio - backend (FastAPI)
# Single-stage. The image is large because torch is large; that is the
# price of including the optional training stack. Trim requirements.txt
# if you only want the API + agent and will not run jobs in-container.

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System packages: build tools for native deps, curl for healthcheck, git for HF cache.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install --default-timeout=1000 -r requirements.txt

COPY app ./app

# Runtime data directories. They are mounted as volumes in docker-compose.
RUN mkdir -p /app/data /app/uploads /app/models

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# Reload mode is off in container; for live-reload dev use docker compose
# with the `dev` overrides described in docs/QUICKSTART.md.
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
