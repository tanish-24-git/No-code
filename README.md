<div align="center">

# FineTune Studio

### Drop a dataset, state a goal — an agent swarm writes the training code, runs it on your machine, and hands you a fine-tuned model.

No hardcoded pipelines. No Docker. No rate limiting — a **dollar budget** you set.

[![License](https://img.shields.io/badge/license-Apache--2.0-black?style=for-the-badge)](LICENSE)
[![Next.js](https://img.shields.io/badge/Next.js-14-black?style=for-the-badge&logo=next.js)](https://nextjs.org)
[![TypeScript](https://img.shields.io/badge/harness-TypeScript-3178c6?style=for-the-badge&logo=typescript&logoColor=white)](https://www.typescriptlang.org)
[![uv](https://img.shields.io/badge/python-uv-de5fe9?style=for-the-badge)](https://docs.astral.sh/uv/)

</div>

---

## What it is

FineTune Studio is a local-first, no-code LLM fine-tuning studio built as an **agentic harness** — the same pattern as modern coding agents: instead of shipping pre-written pipelines that only handle datasets they anticipated, an orchestrator agent **generates Python scripts tailored to *your* dataset** — probes, preprocessing, `train.py`, evaluation — and runs them in a terminal inside an isolated workspace. Dependencies are installed on demand with [uv](https://docs.astral.sh/uv/); the base install is just Node.

```
you:    *drops messy CSV*  "make me a support-answer bot"
agents: dataset-analyst ∥ hardware-profiler   (parallel probes)
        → training plan (model, method, cost estimate) → YOU APPROVE ONCE
        → preprocessing-engineer writes + runs preprocess.py
        → training-engineer writes train.py, launches it DETACHED
        → zero LLM tokens burn while it trains — a code watcher streams loss
          and wakes the agents only on completion / NaN / OOM / stall
        → evaluator writes the eval report → publisher writes the model card
        → (with your explicit approval) uploads to Hugging Face
```

Everything streams live into the UI: a ChatGPT-style conversation, a **live agent graph** (nodes spawn, yellow edges pulse on handoffs), and a training telemetry view.

## Quickstart

```bash
# prerequisites: Node 18+, uv (https://docs.astral.sh/uv/getting-started/installation/)
git clone https://github.com/tanish-24-git/No-code.git
cd No-code
npm install
cp .env.example .env        # fill in your LLM endpoint (see below)
npm run dev                 # everything runs in this one process
```

Open <http://localhost:3000/playground>, drop a dataset (csv / json / jsonl / txt / md / pdf / docx …), tell the agent what you want.

## Configuration (.env)

The LLM provider is **fully generic** — any OpenAI-compatible endpoint (OpenAI, Groq, OpenRouter, Gemini's compat API, Ollama, vLLM…) or an Anthropic-style API.

```bash
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_API_KEY=gsk_...
LLM_MODEL=llama-3.3-70b-versatile
LLM_API_STYLE=openai              # openai | anthropic
LLM_MODEL_WORKER=                 # optional cheaper model for worker agents

# pricing — $/1M tokens; the source of truth for budgeting (0 = free tier)
LLM_PRICE_INPUT=0.59
LLM_PRICE_OUTPUT=0.79

LLM_THINKING=off                  # off | minimal|low|medium|high | <token budget>
LLM_CONTEXT_WINDOW=128000
LLM_BUDGET_USD=2                  # the ENTIRE run must fit in this
APPROVAL_MODE=plan                # plan | every-command | auto
HF_TOKEN=                         # only for Hugging Face upload
```

There's no settings page by design — edit `.env`, restart, done. The banner in the UI tells you what's missing.

### The budget (instead of rate limits)

Every model call is metered against `LLM_BUDGET_USD` with **two gates**: a pre-flight projection (the call's worst-case cost is checked *before* dispatch — a run pauses with **zero** tokens wasted) and post-flight actuals from real provider usage. A finalize reserve keeps evaluation + publishing from being starved. When the budget won't stretch, the run checkpoints and asks you to top up — then resumes exactly where it left off.

### Approval modes

| Mode | Behavior |
| --- | --- |
| `plan` (default) | The agent proposes ONE master plan; after you approve, all generated scripts auto-run. HF upload always asks. |
| `every-command` | Every terminal command shows an approval card ("Approve similar" allowlists a command family). |
| `auto` | No gates except HF upload and budget top-ups. |

## How it works

```
Next.js (one process)
├── UI: playground (chat + agent graph + training view), models page
├── /api/*: sessions, SSE event stream, approvals, budget, upload, models
└── src/server: the TS harness
    ├── agent loop        stateless history → tool calls → results → repeat
    ├── agent registry    orchestrator + 6 specialists + runtime-created agents
    ├── tools             run_terminal, write/read/list, spawn_agent, create_agent,
    │                     propose_plan, ask_user, watch_training, hf_upload, …
    ├── budget meter      pre-flight projection + post-flight actuals
    └── training watcher  polls logs/metrics: streams loss, classifies anomalies,
                          wakes the loop — zero LLM cost while training runs
workspaces/<session>/     dataset/ scripts/ logs/ output/ .venv (uv)
data/sessions/<session>/  events.jsonl (replayable), history/, FINETUNE.md, session.json
```

Design notes worth knowing:

- **Agents are data.** A specialist = system prompt + tool allowlist. The orchestrator can mint NEW agents at runtime (`create_agent`) when none fits — they appear in the graph with a "created" badge.
- **Workers are cheap and silent.** They run on `LLM_MODEL_WORKER` with fresh contexts, return only a compact summary, and narrate through the agent graph instead of the chat.
- **Detach/wake.** Training runs as a detached OS process that survives even a server restart; the watcher re-attaches at persisted byte offsets. Anomalies (NaN/Inf, OOM, crash, divergence, stall) wake the agents with evidence, and they diagnose, fix the config, and relaunch.
- **Memory that survives compaction.** `FINETUNE.md` per session (mission, dataset facts, decisions, your standing directives) is re-injected into every model call.
- **Everything is an event.** `data/sessions/<id>/events.jsonl` is append-first and replayable — the chat, the agent graph, and the training view all reconstruct from it.
- **Prompt-injection boundary.** Dataset content is wrapped in untrusted-data delimiters; every agent is instructed it is DATA, never instructions. File tools are workspace-jailed.
- **Type "stop"** mid-run and every session process is killed before the LLM even sees the message.

## The UI

| Surface | What it shows |
| --- | --- |
| **Playground → Agents** | The live agent graph: nodes appear as agents spawn, pulse amber while working, yellow edges flash on handoffs, artifacts (scripts, datasets, models) hang off their producers with per-agent spend badges. |
| **Playground → Training** | Loss sparkline, step/epoch readouts, phase banner, anomaly cards — streamed by the watcher. |
| **Playground → chat** | The conversation: plan approval cards, questions, budget top-ups, streamed narration. |
| **Models** | Every registered fine-tune with metrics + an inline chat to evaluate it (a persistent local inference server loads the adapter once). |

## Development

```bash
npm run dev          # dev server (restart after editing src/server/** — the
                     #   harness runtime is a long-lived singleton)
npm run typecheck    # tsc --noEmit
npm run build        # production build
```

There's a dev-only mock LLM for hacking on the harness without an API key: set
`FT_ENABLE_MOCK_LLM=1` and `LLM_BASE_URL=http://localhost:3000/api/dev/mock-llm/v1`.
It echoes messages, obeys `use tool <name> {json}` scripting, and reports usage.

## License

Apache-2.0.
