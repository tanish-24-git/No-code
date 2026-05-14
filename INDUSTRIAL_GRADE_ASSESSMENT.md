# Master Directive: Industrial-Grade FineTune Studio (v4.0)

## 🎯 The Vision: "Universal Data-to-Intelligence"
FineTune Studio must evolve from a "Model-Centric" toy into an **"Industrial Data Factory."** 
The core objective is to allow a user to upload **any amount of "crap"** (folders, disparate file types, unlabelled snippets, raw docs) and have the agent swarm autonomously formulate a high-fidelity, instruction-tuned dataset.

---

## 1. The "Data-First" Overhaul (Crap-to-Gold)

### 🚩 Problem: extension-Centric Hardcoding
The current `dataset_service.py` and `DataRestructurerAgent` are bound by hardcoded extension maps (`.pdf`, `.csv`, etc.). This is **UNACCEPTABLE** for industrial grade.

### 🛠️ Requirement: The Universal Intake Engine
- **Recursive Folder Scan:** Support uploading entire directories. The system must recursively scan and ingest every file, regardless of extension.
- **Extension-Agnostic Parsing:** Instead of `if suffix == ".pdf"`, use a **Probabilistic Parser**. The agent should inspect the file header/content and *infer* how to read it.
- **Cross-File Synthesis:** The agent must be able to correlate information across multiple files (e.g., a PDF manual + a folder of JSON logs) to generate complex, multi-turn instruction pairs.
- **Autonomous Data Cleaning:** Use LLM-driven "De-Noising." If the ingested "crap" contains boilerplate, headers, or irrelevant logs, the agent must autonomously filter them out before they hit the training pipeline.

---

## 2. Technical Stability (Solving the "Crash" Problem)

### 🚩 Problem: Blocking I/O & Concurrency Anti-Patterns
The backend currently uses `threading.Lock` and synchronous disk writes inside an `async` loop. This causes the "Parallel Process Lag" the user reported.

### 🛠️ Requirement: Non-Blocking Architecture
- **Asyncio.Lock Only:** Remove all `threading.Lock` from `session_service.py` and `blackboard.py`.
- **Aiofiles Integration:** Use `aiofiles` for every `json.dump` or file write to keep the event loop spinning.
- **Global Event De-Duplication:** Implement a buffer that prevents redundant agent triggers from slamming the session state simultaneously.

---

## 3. High-Fidelity Training (Modern ML Stack)

### 🛠️ Requirement: The "Efficiency-First" Kernel
- **Unsloth Integration:** 2x-5x faster training is mandatory.
- **GaLore / GaLore-Q:** Enable 7B+ parameter fine-tuning on consumer-grade hardware (8GB-16GB VRAM).
- **Liger-Kernel:** Implement Triton-optimized loss functions for maximum throughput.
- **Generative Training Scripts:** The system must **GENERATE** the `train.py` script on-the-fly, optimized for the specific "Crap-to-Gold" dataset it just synthesized.

---

## 4. Master Prompt for the Next Swarm Execution
> **[URGENT MANDATE]**
> "You are the Chief Architect of FineTune Studio. 
> 
> **TASK 1:** Refactor `dataset_service.py` to support recursive directory ingestion. Remove all hardcoded extension logic. Implement a `UniversalIngestAgent` that can read 'crap' and produce 'gold.'
> 
> **TASK 2:** Re-engineer `session_service.py` to be 100% async. Replace `threading.Lock` with `asyncio.Lock` and use `aiofiles` for all persistence. 
> 
> **TASK 3:** Update the `TrainingStrategyAgent` to prioritize **Unsloth** and **GaLore**. The goal is to train a 7B model on 12GB VRAM using the synthesized dataset.
> 
> **GOAL:** Zero deterministic fallbacks. If the user uploads a folder of messy logs, you must autonomously clean, structure, and train on it without being asked twice."

---

## 5. Architectural Verdict
**Foundation:** 🟢 (Fully async — asyncio.Lock + aiofiles ✅)
**Data Engine:** 🟢 (Universal Intake Engine — extension-agnostic, recursive, cross-file ✅)
**Training Engine:** 🟢 (Unsloth + GaLore + Liger integrated ✅)

**Status:** All three mandates from the Master Prompt have been executed.
