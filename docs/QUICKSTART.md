# Quick start

This guide walks through running the project locally for the first time.
You will need Python 3.11+ and Node 18+ on your machine.

## 1. Get the code

```
git clone <your-fork-url>
cd finetune-studio
```

## 2. Backend (FastAPI)

Open a terminal in the project root.

```
cd backend
python -m venv .venv
```

Activate the virtual environment.

| Shell | Command |
| --- | --- |
| Windows PowerShell | `.venv\Scripts\Activate.ps1` |
| Windows cmd | `.venv\Scripts\activate.bat` |
| macOS / Linux | `source .venv/bin/activate` |

Install dependencies and start the API server.

```
pip install -r requirements.txt
cp .env.example .env
python -m uvicorn app.main:app --reload
```

The server listens on `http://localhost:8000`. You can browse the
auto-generated docs at `http://localhost:8000/docs`.

If you have not yet edited `backend/.env`, the startup log prints a
warning along the lines of:

```
WARNING:finetune-studio:LLM is NOT configured. Either set LLM_PROVIDER,
LLM_API_KEY, and LLM_MODEL in backend/.env, or open the UI and visit Settings.
```

That is expected. You can configure the LLM through the UI in step 4
instead of editing the env file. Both options work, and the UI overrides
the env when both are set.

## 3. Frontend (Next.js)

Open a second terminal.

```
cd frontend
npm install
npm run dev
```

The dev server listens on `http://localhost:3000`. It proxies `/api/*`
and `/health` to the backend, so you do not need to think about CORS.

## 4. First-run setup wizard

Open `http://localhost:3000`. If the LLM is unconfigured, a yellow banner
appears across the top with a "run setup" button. Click it (or visit
`/setup` directly).

The wizard has three steps.

1. **Provider.** Pick from a tile grid. 17 providers ship out of the box:
   `anthropic`, `openai`, `gemini`, `groq`, `grok`, `deepseek`, `mistral`,
   `together`, `fireworks`, `openrouter`, `perplexity`, `cohere`,
   `huggingface`, `ollama`, `lmstudio`, `vllm`, and a `custom`
   OpenAI-compatible escape hatch. Each tile shows the default base URL
   and whether an API key is required.
2. **Model.** Type the model id. Suggestions for the active provider
   appear as click-to-fill chips (e.g. `claude-sonnet-4-5`,
   `gemini-2.5-flash`, `llama-3.3-70b-versatile`, `gpt-4o-mini`,
   `deepseek-reasoner`).
3. **API key.** Paste the key. For local providers (`ollama`,
   `lmstudio`, `vllm`) you can leave this blank.

Click **save and verify**. The backend immediately probes the provider's
`/models` endpoint to confirm credentials. On success the wizard shows
a green "setup complete" panel with three call-to-action buttons.

## 5. (Optional) Hugging Face token

If you plan to pull base models from the Hub or push trained adapters
back to it, open Settings and paste a Hugging Face token (read or
read+write). You can also set this in `backend/.env` as `HF_TOKEN=...`.

## 6. Run something

In the playground:

1. Click **+ new** to create a pipeline.
2. Click **upload dataset**, choose a CSV, JSON, or JSONL file. The
   pipeline auto-attaches it.
3. In the agent panel on the right, send a message such as:
   `"read my dataset and configure this pipeline for my hardware"`.
   The agent will call its `get_dataset`, `get_hardware`, and
   `suggest_pipeline_config` tools, then write the recommended values
   back to the pipeline (you will see the inspector panel update live).
4. Click **run pipeline**. The bottom panel streams logs in real time.

The shipped train node is a stub that simulates an epoch loop and writes
a placeholder artifact. Wire the real LoRA / QLoRA / full-fine-tune
trainer of your choice into `backend/app/services/job_service.py`
(`_handler_train`) when you are ready to do real training.

## What's next

- [docs/CONFIGURATION.md](CONFIGURATION.md) - every env var and setting.
- [docs/INFERENCE.md](INFERENCE.md) - register your local Ollama or
  vLLM server and let the agent suggest generation metrics.
- [docs/AGENTS.md](AGENTS.md) - how the agent works and what tools it
  has.
