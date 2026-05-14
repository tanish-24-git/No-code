# FineTune Studio: Backend Stability & Orchestration Audit (2026-05-15)

This report documents the architectural "choking points" and runtime blockers identified during the deployment of the FineTune Studio v4.0 AgenticLoop.

## 1. Resolved Blockers
*   **Circular Import (Critical):** Fixed a dependency loop between `app.agents.base` and `app.tools.registry`. The `agent_tools` module (which depends on `BaseAgent`) was being imported during tool package initialization, which was triggered by `BaseAgent` itself. **Fix:** Removed `agent_tools` from `app/tools/__init__.py`.
*   **Gemini Compatibility (Critical):** Fixed a `400 Bad Request` where Gemini would fail if the `AgenticLoop` started with zero user history. **Fix:** Injected a synthetic "Please begin" user message for empty sessions.

## 2. Active Technical Issues
*   **Groq Tool-Call Fragility (High):** 
    *   **Error:** `Failed to call a function. Please adjust your prompt.`
    *   **Symptom:** Groq's `llama-3.3-70b` occasionally emits malformed tool calls or triggers server-side validation errors when processing complex plans (like searching and profiling in the same turn).
*   **Cloud Quota Exhaustion (High):**
    *   **Error:** `429 Too Many Requests` / `RESOURCE_EXHAUSTED`.
    *   **Symptom:** The autonomous `AgenticLoop` executes multiple turns in quick succession. Groq's 30 RPM and Gemini's 5 RPM limits are hit within the first minute of dataset intake.
*   **Lack of Turn Throttling (Medium):**
    *   The loop runs at the maximum speed allowed by inference. Without an artificial "thinking delay" (jitter), it bursts past API rate limits and makes the UI narration difficult for humans to follow.
*   **Stale Frontend State (Medium):**
    *   **Error:** `404 Not Found` for sessions.
    *   **Symptom:** After a `docker compose down -v` (volume wipe), the browser persists the old session ID in the URL, causing polling errors until the user manually refreshes or navigates back to Home.

## 3. Deployment Constraints
*   **Compute:** The backend currently relies on Cloud LLMs. For industrial-grade "zero-limit" testing, a local **Ollama** instance is required but not yet integrated into the `docker-compose` stack.
*   **Web Grounding:** DuckDuckGo search is limited. Higher-tier research requires a `TAVILY_API_KEY` which is currently missing from the production environment.

## 4. Pending Remediation
1.  Add `asyncio.sleep(3)` jitter to `AgenticLoop` turns to prevent 429s.
2.  Add a `failed_generation` logger to capture the raw JSON causing Groq's function call errors.
3.  Implement a more robust "synthetic user nudge" for all OpenAI-compatible providers.
