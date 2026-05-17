# FineTune Studio: Repository Stability & Performance Review (May 2026)

This report provides a "Maintainer-Level" audit of the FineTune Studio backend, focusing on industrial reliability and performance under "low-level" (free-tier/small) LLM constraints.

## 1. Executive Summary
The backend architecture is **highly robust**. The transition from a static Directed Acyclic Graph (DAG) to an **Event-Driven Agentic Loop** with a Federated Blackboard is a major leap in stability.

## 2. Architectural Strengths
- **Event-Driven Resilience:** Using an `EventBus` and SSE for streaming allows the UI and Backend to stay decoupled. Cancellations (UserMessage interrupts) are handled cleanly via `asyncio.shield` for critical jobs (Training/Export).
- **Proactive Guardrails:** The `LLMProbe` at upload-time prevents "silent failure" loops where an agent tries to think with an invalid API key.
- **Provider-Aware Throttling:** Turn-based sleep (e.g., 13s for Gemini) effectively mitigates 429 errors without complex infrastructure.
- **Fuzzy Orchestration:** `_coerce_phase` allows models to use natural language for planning without crashing the materialization logic.
- **Global Sanitization:** A central middleware in `registry.py` now automatically strips `null` hallucinations and enforces a 2000-char observation budget to prevent context window bloat.

## 3. Vulnerabilities & "Low-Level LLM" Choking Points

### 🚨 Critical: Quota Exhaustion (5 RPM & 15-20 RPD Hard Cap)
**Issue:** Gemini's free tier enforces a strict 5 requests-per-minute (RPM) limit and, in many regions, a severe **15-20 requests-per-day (RPD)** hard cap.
**Evidence:** `Quota exceeded for metric: ... GenerateRequestsPerDayPerProjectPerModel-FreeTier, limit: 20`.
**Impact:** An agentic loop consumes ~5-10 turns per pipeline run. With a 20 RPD limit, the application becomes unusable after just 2-3 sessions.
**Maintainer Advice:** For continuous development, **Groq** (with much higher free-tier RPD) or a paid **Gemini API** billing account is mandatory.



### 📉 Low: Stale Session State
**Issue:** Frontend 404s after volume wipes.
**Maintainer Advice:** The backend should return a `410 Gone` instead of `404` for sessions that existed in the event log but are missing from disk. This triggers a cleaner "Reset UI" state.

## 4. Maintenance Roadmap
1.  **[Med] Ollama Integration:** Provide a local LLM fallback for high-burst profiling tasks (Hardware/Dataset Profile) to preserve cloud credits for Strategy/Planning.
4.  **[Med] Decision Replay:** Improve the `audit` log to allow the LLM to "Review Past Failures" from the blackboard before attempting a retry.

---
**Reviewer:** Antigravity (Maintainer AI)
**Status:** 🟢 STABLE
