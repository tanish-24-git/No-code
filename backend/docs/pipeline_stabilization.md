# FineTune Studio: Pipeline Stabilization Report

This document summarizes the critical fixes applied to the Agentic Runtime to resolve the "nonsense" hallucinations, backend crashes, and Groq integration failures.

## 1. 🛑 Current Blocker: Groq Rate Limiting (429)
**Problem**: Your logs show a `Too Many Requests` error from Groq. 
- **Limit**: 100,000 tokens per day (Free Tier).
- **Current Status**: 97,019 tokens used.
- **Cooldown**: ~5 minutes.
- **Fix**: The system is now using `Tier Routing`. It will send mechanical tasks (intake, profiling) to the smaller `llama-3.1-8b-instant` model to preserve your 70B quota for high-level reasoning.

---

## 2. 🧠 Fix: Hallucination Prevention
**The "Nonsense" Problem**: When model search returned zero candidates (due to CPU/VRAM limits), the agent would hallucinate models like `t5-base` and try to build pipelines for them.
**The Fix**:
- **Removed Hardcoded Fallbacks**: Deleted the brittle list of tiny models from `hf_search.py`.
- **Transparent Errors**: The tool now returns the exact hardware budget (`budget_b`) it used, so the agent can explain *why* it found nothing instead of lying.
- **Web Grounding**: Enabled `web_search` and `web_fetch` for the `ModelSelectionAgent`. It can now find the latest models on the web if the API is stale.

---

## 3. 🛠️ Fix: Backend Crash Protection
**Problem**: The backend was crashing with `KeyError: 'job_id'` when an agent tried to export a model that didn't exist.
**The Fix**:
- **Tool Hardening**: Updated `export.save_local` and `run_training` to return structured `{"error": "..."}` dicts.
- **State Validation**: Added explicit checks to verify a `pipeline_id` and `job_id` exist before allowing the agent to proceed to the next node.

---

## 🔗 Fix: Groq Tool-Call Compliance
**Problem**: Groq was rejecting assistant messages with "Failed to call a function" because we were using synthetic "User" messages for tool results.
**The Fix**:
- **Role Alignment**: Refactored `AgenticLoop` to use the native `tool` role and `tool_calls` array. This satisfies Groq's strict schema requirements and prevents history corruption.

---

## 📋 Recommended Action Plan
1. **Wait 5 Minutes**: Allow the Groq TPD window to reset.
2. **Start a New Session**: This clears the corrupted history and starts fresh with the new web-grounding logic.
3. **Verify Tiering**: Check that mechanical tasks show up as using `llama-3.1-8b` in the logs to save tokens.
