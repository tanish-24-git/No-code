# Architectural Assessment: FineTune Studio (Agentic IDE Phase)

> [!IMPORTANT]
> This document is formatted as a **system prompt**. You can feed this into another agentic IDE or LLM to provide it with full context of our progress and the path forward.

## 1. Context & Mission
FineTune Studio has been pivoted from a deterministic, pipeline-based orchestrator to a **100% autonomous, dialogue-driven agentic swarm**. 

**The Vision:** An "Antigravity-style" IDE for fine-tuning where the system doesn't just execute a hardcoded graph, but dynamically reasons about the data, selects the model, and (eventually) writes the training code/tools necessary to achieve a user-defined goal.

---

## 2. Current State (The "Hardened" Swarm)
We have successfully decommissioned all heuristics. The architecture now relies on:
- **Federated Blackboard:** A shared cognitive workspace where agents post thoughts, plans, and decisions.
- **Global Directives Bus:** A persisted store for user intent (goals, constraints) that every agent reads before calling the LLM.
- **Goal-Gated Cascade:** The system now halts after dataset intake, waiting for a `UserMessage` to define the training goal before any task inference or model selection occurs.
- **Event-Driven Orchestration:** No central "controller" function; agents react to events (`IntakeCompleted`, `GoalCaptured`, `TaskInferred`) in a decoupled hive-mind.

---

## 3. Current Problems & Blockers
1. **LLM Hallucination Risk:** Without deterministic fallbacks, the system is highly sensitive to LLM output. If an agent fails to return valid JSON or makes an illogical inference, the pipeline stalls.
2. **Synchronicity Gaps:** While the `GoalCaptured` gate is implemented, some secondary agents (like `HardwareAnalysis`) still fire eagerly. While technically efficient, it can feel "un-conversational" to the user.
3. **Implicit vs. Explicit Tools:** We are currently using **pre-existing tools** (e.g., `model.search_hf`, `task.classify`). The vision requires the system to **generate tools or code** on the fly when a pre-existing tool is insufficient.
4. **State Traceability:** Debugging an event-driven swarm is harder than a linear script. We need better "Cognitive Logs" for the user to see *why* an agent made a decision.

---

## 4. Future Risks (Architecture Scale)
1. **Blackboard Bottleneck:** The current JSON-on-disk blackboard might experience latency or race conditions as the number of concurrent training sessions grows.
2. **Context Window Saturation:** As a session progresses, the history of "Thoughts" and "Directives" may exceed the LLM's prompt limit, requiring a RAG-based or distilled state management approach.
3. **Refinement Loops:** We lack a robust "Audit-Correct" loop. If the `AuditAgent` finds a flaw in the `PipelineBuilder's` graph, the system needs a standard protocol for "Re-Planning" without losing state.

---

## 5. Resolution Roadmap (Matching the IDE Vision)

### Phase A: The Synthetic Data Agent
**Goal:** If a dataset is too small or lacks instruction pairs, an agent must autonomously generate high-quality synthetic examples to "boost" the fine-tuning quality.

### Phase B: Tool/Code Generation Agent
**Goal:** Transition from "Choosing from a catalog" to "Generating the script."
- **Current:** `StrategyAgent` picks `LoRA`.
- **IDE Vision:** `AlchemistAgent` writes a custom PyTorch/Unsloth training script tailored specifically to the dataset's unique nuances.

### Phase C: Proactive Self-Correction
**Goal:** The `MonitoringAgent` should not just report errors but trigger the `RecoveryAgent` to modify the `TrainingStrategy` (e.g., lower learning rate) and resume the job without user intervention.

---

## 6. Execution Prompt for the Secondary Agent
**"You are being integrated into FineTune Studio. Your first task is to review the current `EventBus` and `Agent` implementations. Focus on transitioning the system from using static tools to a 'Generative IDE' model. Ensure all future agents you design are 'Dialogue-First'—they must wait for user directives and provide clear rationale on the Blackboard before acting. Your immediate priority is the implementation of the Synthetic Data Generation Agent."**

---

## 7. Assessment: Are we on track?
**Status: YES.**
By removing the "deterministic safety nets," we have forced the system into the "Pure Agentic" regime. The "Antigravity" feeling comes from the system *listening* to the user and *reasoning* in the open. The foundation is solid; the next 20% of work is transitioning from "Selecting" to "Creating."
