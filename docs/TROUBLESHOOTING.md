# Troubleshooting

## "Backend offline" pill in the navbar

The frontend pings `/health` on a 30-second interval. If the request
fails, the navbar shows a red dot and "backend offline".

Checks:

1. Is `python -m uvicorn app.main:app --reload` running and listening
   on port 8000?
2. Did `npm run dev` print "ready"? The frontend proxies `/api/*` to
   the backend, so the dev server must be up.
3. Open `http://localhost:8000/health` directly. If that loads, the
   issue is on the proxy side - check `frontend/next.config.mjs`.

## Yellow "Agent is not configured" banner

Means `LLM_PROVIDER` and `LLM_MODEL` are both unset. Open `/setup` or
edit `backend/.env`. After saving, the banner clears within 15 seconds
(SWR refresh interval) or on the next page load.

## "LLM is not configured" in chat

The agent saw a chat request but `get_llm_config()` resolved to the
unset state. Same fix as above.

## Verification fails

`POST /api/settings/verify-llm` reports `valid: false`.

| Symptom | Likely cause |
| --- | --- |
| `HTTP 401` | Wrong API key. |
| `HTTP 403` | Key valid but missing model access (Anthropic gates new models). |
| `HTTP 404` on Anthropic | Wrong base URL. Should be `https://api.anthropic.com` or blank. |
| Connection refused (Ollama) | Server not running. `ollama serve` and re-probe. |
| TLS errors against local servers | Some local servers default to plain HTTP. Use `http://localhost:...` not `https://`. |

## Agent runs in circles

If the assistant prints `[agent reached max tool-use hops]`, it called
tools too many times in one turn. This usually means the model is
unable to satisfy the request with the available tools. Either:

- Rephrase the user request to be more specific.
- Add a new tool to `backend/app/agents/tools.py` and register it in
  `TOOLS`.

## Logs stop streaming partway through

EventSource auto-reconnects, but if your reverse proxy buffers, SSE
breaks. Disable buffering:

- nginx: `proxy_buffering off;` and `proxy_cache off;` on the
  `/api/jobs/.../logs` location.
- Caddy: `flush_interval -1` on the reverse_proxy directive.

Locally, none of this should matter because Next.js dev does not
buffer.

## "Cannot find module 'reactflow'" or similar after pulling

Run `npm install` again in `frontend/`.

## Pulled HF model says "401" or "404"

`HF_TOKEN` is missing or lacks read scope. Generate a new token at
[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens),
paste it in Settings, click verify.

For gated repos (Llama, etc.), you also need to accept the license on
the Hub before the token can pull.

## "Encryption key invalid" errors after deleting data/

If you delete `data/` while keeping `data/settings.json` (e.g. via
selective rm), the new auto-generated `data/.encryption_key` will not
match the encrypted blobs already in `settings.json`. Either keep both
or delete both.

## Windows: `cp .env.example .env` fails

Use PowerShell `Copy-Item .env.example .env` or just paste the file
in Explorer. Windows cmd does have a `copy` command if you prefer.

## Train node finishes too fast

The shipped train handler is a stub. It runs five fake steps per epoch
and writes a README as the artifact. Wire a real trainer in
`backend/app/services/job_service.py::_handler_train` (see
`docs/PIPELINES.md`).

## Where to file issues

Open one in the repo with:

- The OS and Python/Node versions.
- The full backend startup log (it dumps the resolved LLM config).
- The browser console output for frontend issues.
- The contents of `data/jobs/<job_id>.log` if a job misbehaved.
