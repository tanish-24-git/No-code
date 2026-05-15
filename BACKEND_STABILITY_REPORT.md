# FineTune Studio: Active Backend Issues & Blockers (2026-05-15)

This report tracks current runtime blockers preventing the FineTune Studio v4.0 AgenticLoop from completing autonomous training runs. 

> [!NOTE]
> All architectural blockers (Circular Imports) and start-up crashes (Gemini History) have been successfully resolved in the latest build.

## 1. Critical & High Priority Issues

*   **Groq Function Call Rejection (Critical):**
    *   **Error:** `Failed to call a function. Please adjust your prompt.`
    *   **Symptom:** Groq's validation layer rejects tool calls (especially `ask_user`) if the model's output deviates from the strict JSON schema. This stops the agent from communicating with the user when it needs help.
*   **HF Search Schema Validation (High):**
    *   **Error:** `parameters for tool search_hf_models did not match schema: errors: [/family: expected string, but got null]`.
    *   **Symptom:** Groq's `llama-3.3-70b` sends `null` for optional parameters instead of omitting them. The backend enforces strict types, causing the tool to fail.
*   **Phase Hallucination Risk (High):**
    *   **Error:** `ValueError: unknown phase: 'fine-tuning'` (or 'training').
    *   **Symptom:** The `propose_plan` tool crashes if the LLM uses natural language phase names instead of canonical ones (e.g., `execute`).
*   **Cloud Quota Exhaustion (High):**
    *   **Error:** `429 Too Many Requests` / `RESOURCE_EXHAUSTED`.
    *   **Symptom:** Even with the new 3-second turn throttle, free-tier accounts on Groq and Gemini hit RPM limits during the "Profile -> Task -> Model" burst sequence.

## 2. General Stability Issues

*   **Stale Frontend State (Medium):**
    *   **Error:** `404 Not Found` for sessions.
    *   **Symptom:** After a volume wipe, the browser tries to poll the previous session ID, leading to 404s until a manual hard refresh is performed.
*   **Web Grounding Limits (Low):**
    *   **Symptom:** DuckDuckGo search limits restrict the agent's ability to research deep model benchmarks without a `TAVILY_API_KEY`.

## 3. Pending Remediation
1.  **Schema Hardening:** Update `agent_tools.py` to handle `null` values or allow them in the JSON schema.
2.  **Phase Mapping:** Implement a synonym mapper in `phase_service.py` (re-applying the previously suggested fix).
3.  **Local Inference:** Integrate **Ollama** into the `docker-compose` stack to bypass cloud quotas entirely.
