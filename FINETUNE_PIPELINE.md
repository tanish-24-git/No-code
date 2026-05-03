# Fine-Tuning Pipeline Implementation Details

This document outlines the architecture, backend logic, and required user inputs for the LLM fine-tuning pipeline implemented in the `No-code` platform.

## 1. Orchestration and Architecture

The fine-tuning pipeline is part of a Directed Acyclic Graph (DAG) orchestrated by the `OrchestratorAgent`.
- **Topological Execution:** The DAG nodes represent modular task agents (e.g., `DatasetAgent`, `PreprocessingAgent`, `TrainingAgent`, `ValidationAgent`) which run in topological order based on predefined dependencies.
- **Agent:** The main computational load is handled by `TrainingAgent` (located in `app/agents/training_agent.py`), which uses the `BaseAgent` abstraction and dynamically adapts to the underlying hardware context.

### Infrastructure integration
- **Storage:** Datasets and trained checkpoints are loaded and saved asynchronously using an S3-compatible `MinIO` object store.
- **Hardware Abstraction:** A central `gpu_manager` auto-detects CUDA GPU availability to map operations to CPU or GPU respectively.
- **Logging/Tracking:** A `model_registry` service securely registers fully trained models and associates them with unique IDs, evaluation metrics, and their base HuggingFace identifiers. Progress and logs are streamed over SSE.

## 2. Backend Training Logic

The backend supports three primary approaches through dedicated wrapper classes over the `HuggingFace Trainer` and `PEFT` ecosystems.

1. **LoRA (Low-Rank Adaptation) Trainer (`app/training/lora.py`)**
   - Implements parameter-efficient fine-tuning using HuggingFace's `PEFT` library.
   - Operates under standard FP16 conditions context when accelerated by GPU or falls-back gracefully to PF32 memory footprint on CPU clusters.
   - Configures weight updates strictly against configured injection layers (e.g., `q_proj`, `v_proj`).

2. **QLoRA Trainer (`app/training/qlora.py`)**
   - Designed for extreme memory frugality on single-GPU instances.
   - Utilizes HuggingFace `BitsAndBytesConfig` allowing 4-bit model quantization (`nf4` quant type and double quantization).
   - *Fallback Mechanism:* Explicitly falls back to regular 16-bit LoRA if assigned a CPU node, preventing compilation failures since 4-bit inference isn't CPU compatible natively.
   - Deploys the `"paged_adamw_8bit"` optimizer to curtail out-of-memory errors mapping paging directly back to system RAM dynamically.

3. **Full Fine-Tuning (`app/training/full_finetune.py`)**
   - Standard fine-tuning of all parameters based natively on `AutoModelForCausalLM`.

## 3. User Inputs (The 22 Field Configuration)

Our application configures these training algorithms based on user-supplied profiles grouped roughly into five logical screens managed and validated securely by `app/api/schemas/training.py`:

### Screen 1: Project Setup
- `project_name` (String): Display name of the project.
- `description` (String, Optional): Functional summary for internal search.
- `tags` (List[String]): Identification markers for model categorization. 

### Screen 2: Dataset Config
- `dataset_name` (String): Raw data alias.
- `target_column` (String): The inference label/summary column.
- `input_columns` (List[String]): Attributes concatenated to build context window.
- `split_ratio` (Float): Determines Train vs Evaluation breakdown ratio (e.g., `0.8`).

### Screen 3: Task Definition
- `task_type` (Enum): e.g., Classification, Chat, QA, Extraction.
- `output_type` (Enum): e.g., JSON, text, label, multi-label.
- `domain` (Enum): General, finance, medical, etc., tuning base behaviors.
- `language` (String): Targeted inference language.

### Screen 4: Training Specifics
- `training_mode` (Enum): Simplifier abstractions (presets: `fast`, `balanced`, `high_quality`).
- `base_model` (String): Target HuggingFace model weight path (e.g. `TinyLlama-1.1B`).
- `epochs` (Int): Total passes throughout the entire dataset.
- `batch_size` (Int): Active tensor sample groupings.
- `learning_rate` (Float): Size of optimizer steps.
- `max_seq_len` (Int): Prompt truncation size limits.
- `lora_rank` (Int): Dimensionality of the low-rank updates (typical powers of two: `8, 16, 32, 64`).

### Screen 5: Advanced & Edge Options
- `gradient_accumulation` (Int): Micro-batch accumulations before step optimizations.
- `precision` (Enum): Floating point standards e.g., `bf16`, `fp16`, `float32`.
- `early_stopping` (Bool): Auto-aborts processing loops on validation stagnation.
- `class_balancing` (Bool): Penalizes/Rewards minor classifications to prevent skewing.
- `data_augmentation` (Bool): Internal permutations added to sparse data contexts.
- `resume_checkpoint` (String, Optional): An ongoing run UUID string linked to MinIO.
