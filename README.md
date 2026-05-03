# FineTune Studio

A local-first, open-source platform for LLM fine-tuning and inference
tuning, with a node-based pipeline editor and an LLM-powered agent that
can read your hardware, your datasets, and your inference endpoints, then
recommend exact configurations.

No Redis. No database. No Docker. No telemetry.

```
backend/    FastAPI, Python 3.11+, file-based JSON store, BackgroundTasks
frontend/   Next.js 14 App Router, TypeScript, Tailwind, React Flow
docs/       Detailed documentation (you are here)
data/       JSON state created on first run (settings, pipelines, jobs, ...)
uploads/    Raw dataset files
models/     Pulled base models and trained outputs
```

## Highlights

- **Node-based pipelines.** Drag-and-drop dataset, preprocess, train,
  evaluate, and export nodes. Wire them however you like.
- **Bring your own inference.** Register Ollama, OpenAI-compatible servers
  (vLLM, LM Studio, OpenRouter, Together, Groq), Hugging Face Inference,
  or Anthropic endpoints. The agent reads them as tools.
- **Bring your own LLM provider.** 17 providers supported out of the box:
  Anthropic, OpenAI, Google Gemini, Groq, xAI Grok, DeepSeek, Mistral,
  Together AI, Fireworks AI, OpenRouter, Perplexity, Cohere, Hugging
  Face Router, Ollama, LM Studio, vLLM, and a `custom` escape hatch.
  Pick one in the setup wizard, paste a key, done. The provider list is
  driven from a single Python file - adding a new one is a one-entry
  change.
- **First-run setup wizard.** Boot the project, open the UI, and a guided
  three-step flow takes you from "no config" to "agent verified".
- **Encrypted at rest.** API keys are Fernet-encrypted; the symmetric
  key auto-generates on first run.
- **Streamed everything.** Job logs and agent chat both arrive over SSE.

## Documentation

| Topic | File |
| --- | --- |
| Quick start | [docs/QUICKSTART.md](docs/QUICKSTART.md) |
| Run with Docker | [docs/DOCKER.md](docs/DOCKER.md) |
| All configuration variables | [docs/CONFIGURATION.md](docs/CONFIGURATION.md) |
| Architecture and request flow | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| HTTP API reference | [docs/API.md](docs/API.md) |
| Agent, tools, and providers | [docs/AGENTS.md](docs/AGENTS.md) |
| Inference endpoints | [docs/INFERENCE.md](docs/INFERENCE.md) |
| Pipelines and jobs | [docs/PIPELINES.md](docs/PIPELINES.md) |
| Troubleshooting | [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) |

## Five-minute tour

### Option A: Docker (one command)

```
cp backend/.env.example backend/.env
# (optional) edit backend/.env and paste your LLM_API_KEY
docker compose up --build
```

Frontend at `http://localhost:3000`, backend at `http://localhost:8000`.
Full Docker notes in [docs/DOCKER.md](docs/DOCKER.md).

### Option B: Local Python + Node

1. Install Python 3.11+ and Node 18+.
2. Start the backend:
   ```
   cd backend
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   cp .env.example .env
   python -m uvicorn app.main:app --reload
   ```
3. Start the frontend in another terminal:
   ```
   cd frontend
   npm install
   npm run dev
   ```

### Either way

4. Open `http://localhost:3000`. If you have not configured an LLM, the
   warning banner routes you to `/setup`. Walk through the three steps
   (provider, model, API key), and the agent is ready.
5. Open the playground, drop a CSV, and ask the agent in the side panel:
   `"look at my hardware and configure this pipeline"`.

## License

Open source. Choose MIT or Apache 2.0 to taste before publishing.
