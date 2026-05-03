# Running with Docker

The project ships with two Dockerfiles and a `docker-compose.yml` at
the repo root. One command brings up the entire stack:

```
docker compose up --build
```

| Service | Image | Port | Source |
| --- | --- | --- | --- |
| `backend` | `finetune-studio-backend` | 8000 | `backend/Dockerfile` |
| `frontend` | `finetune-studio-frontend` | 3000 | `frontend/Dockerfile` |

Both images are built locally; nothing is pushed to a registry by
default.

## Before you start

The backend expects an LLM provider configured via `backend/.env`. Copy
the template once:

```
cp backend/.env.example backend/.env
```

Open `backend/.env` and at minimum set the LLM section. Pick any of the
17 supported providers - the comments at the top of `.env.example`
list every option. Examples:

```
# Anthropic Claude
LLM_PROVIDER=anthropic
LLM_API_KEY=sk-ant-...
LLM_MODEL=claude-sonnet-4-5
```

```
# Google Gemini (free tier available)
LLM_PROVIDER=gemini
LLM_API_KEY=AIzaSy...
LLM_MODEL=gemini-2.5-flash
```

```
# Groq (very fast, free tier)
LLM_PROVIDER=groq
LLM_API_KEY=gsk_...
LLM_MODEL=llama-3.3-70b-versatile
```

```
# Local Ollama (no API key)
LLM_PROVIDER=ollama
LLM_MODEL=llama3.1:8b
# Optional: only set if Ollama is on a non-default port or another host.
LLM_BASE_URL=
```

You can also leave the LLM block blank and configure everything later
through the Settings page in the UI - the wizard at
`http://localhost:3000/setup` walks you through it.

## Architecture inside Docker

```
[browser localhost:3000]
        |
        v
+---------------+         +---------------+
|   frontend    |  --->   |    backend    |
|  Next.js 14   |         |   FastAPI     |
| port 3000     |         | port 8000     |
+---------------+         +---------------+
                                  |
                                  v
                          +---------------+
                          |  named volumes|
                          | data uploads  |
                          | models        |
                          +---------------+
```

The frontend's Next.js server proxies `/api/*` and `/health` to the
backend over the Docker network at `http://backend:8000`. The browser
only ever talks to `http://localhost:3000`. CORS therefore does not
come into play; we still set `CORS_ORIGINS_RAW` for the case where
someone hits the backend directly on port 8000.

## Volumes

By default state lives in three named volumes:

```
backend_data       maps to  /app/data       in the backend container
backend_uploads    maps to  /app/uploads    in the backend container
backend_models     maps to  /app/models     in the backend container
```

To inspect state on the host instead, switch to bind mounts in
`docker-compose.yml`:

```yaml
volumes:
  - ./backend/data:/app/data
  - ./backend/uploads:/app/uploads
  - ./backend/models:/app/models
```

The compose file has these lines commented out next to the named
volumes; just swap which set is active.

## Useful commands

```
# Start (foreground)
docker compose up --build

# Start (detached)
docker compose up --build -d

# Tail logs
docker compose logs -f

# Tail one service
docker compose logs -f backend

# Stop and keep volumes (state is preserved)
docker compose down

# Stop and DELETE all state
docker compose down -v

# Rebuild a single service after code changes
docker compose up --build backend

# Shell into the backend container
docker compose exec backend bash

# Shell into the frontend container
docker compose exec frontend sh
```

## GPU access

The base backend image is CPU-only because `torch` from PyPI in this
configuration installs the CPU build. If you want CUDA inside the
container you have two clean options:

1. **Run training on the host.** Keep the backend in Docker for the
   API and agent layer, then override `_handler_train` in
   `app/services/job_service.py` to shell out to a host-side trainer.
2. **Build a CUDA image.** Replace `python:3.11-slim` in
   `backend/Dockerfile` with `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04`,
   add Python via apt, and reinstall torch with the matching CUDA
   wheels. Then add `deploy.resources.reservations.devices` to the
   backend service in `docker-compose.yml`.

Most users running a local studio will never hit this; the agent and
inference flows do not need a GPU.

## Image sizes

Approximate, after first build:

| Image | Size |
| --- | --- |
| backend (CPU torch) | ~3-4 GB |
| frontend (standalone) | ~250 MB |

If you do not need the training stack inside the container, delete the
torch / transformers / peft / bitsandbytes lines from
`backend/requirements.txt` before building. The agent and pipelines
still work without them; only `_handler_train` needs them.

## Troubleshooting Docker

- **`backend.healthcheck` keeps failing.** The backend container shows
  the boot log via `docker compose logs backend`. Look for the
  `LLM is NOT configured` warning - the API still serves, but most
  endpoints return errors when the LLM is missing.
- **`docker compose up` exits with `Cannot find module ...`.** Delete
  `frontend/node_modules` on the host (the build stage installs them
  fresh inside the container, but a stale host copy can confuse the
  COPY layer). Running with the bind-mount overrides shown above will
  also help debug this.
- **HF push from inside the container 403s.** Make sure `HF_TOKEN` in
  `backend/.env` has write scope and that you have accepted the
  destination repo's terms on the Hub.
