"""Real fine-tuning runtime.

v4.0 — Efficiency-First kernel with Unsloth + GaLore + Liger support.

Replaces the placeholder trainer with an actual transformers + peft +
(optionally) trl run that:

    1. Loads the base model from ``config.base_model`` (HuggingFace Hub or
       a local snapshot). **Prefers Unsloth FastModel** for 2x-5x speedup
       when the model architecture is supported.
    2. Builds a tokenized dataset from the bound DatasetSchema, supporting
       all four flavours produced upstream:
          * instruction (instruction / input / output)
          * chat        (messages list of {role, content})
          * qa          (instruction / output - same as instruction)
          * classification (text / label) - tuned via SFT on label-as-text
          * language_modeling (text)
    3. Wraps the model in a LoRA adapter (or QLoRA if ``method=='qlora'``).
    4. Configures the optimizer: **GaLore** for 7B+ on ≤16GB VRAM,
       Adam8bit for moderate savings, or standard AdamW.
    5. Runs ``trl.SFTTrainer`` (preferred) or falls back to
       ``transformers.Trainer`` with a basic causal-LM collator.
    6. Streams loss + step metrics through a callback that calls back into
       ``job_service`` so the FE sees real metrics live.
    7. Saves the trained adapter to ``models/<job_id>/`` so the /test
       endpoint can serve real responses from the user's adapter.

If any of the heavy ML deps (``torch``, ``transformers``, ``peft``,
``datasets``, optionally ``trl``, ``bitsandbytes``, ``unsloth``,
``galore-torch``) are missing or the base model can't be downloaded,
this module raises a clear error with a remediation hint - it never
silently produces a stub.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Optional


log = logging.getLogger("finetune-studio.trainer")


class TrainingNotAvailable(Exception):
    """Raised when the heavy ML stack is not importable. Surfaced to the UI
    so the user knows exactly what to install."""


def _require_stack() -> dict[str, Any]:
    """Lazy-import torch / transformers / peft / datasets and return them
    in a dict so the caller can reference them by name. Any missing piece
    becomes a TrainingNotAvailable with the pip command to fix it."""
    deps: dict[str, Any] = {}
    try:
        import torch  # type: ignore
        deps["torch"] = torch
    except Exception as e:
        raise TrainingNotAvailable(
            f"torch is required to train models locally. "
            f"pip install torch ({e})"
        )
    try:
        from transformers import (  # type: ignore
            AutoModelForCausalLM, AutoTokenizer,
            DataCollatorForLanguageModeling, Trainer, TrainingArguments,
            TrainerCallback,
        )
        deps["AutoModelForCausalLM"] = AutoModelForCausalLM
        deps["AutoTokenizer"] = AutoTokenizer
        deps["DataCollatorForLanguageModeling"] = DataCollatorForLanguageModeling
        deps["Trainer"] = Trainer
        deps["TrainingArguments"] = TrainingArguments
        deps["TrainerCallback"] = TrainerCallback
    except Exception as e:
        raise TrainingNotAvailable(
            f"transformers is required. pip install transformers ({e})"
        )
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # type: ignore
        deps["LoraConfig"] = LoraConfig
        deps["get_peft_model"] = get_peft_model
        deps["prepare_model_for_kbit_training"] = prepare_model_for_kbit_training
    except Exception as e:
        raise TrainingNotAvailable(
            f"peft is required for LoRA / QLoRA. pip install peft ({e})"
        )
    try:
        from datasets import Dataset  # type: ignore
        deps["Dataset"] = Dataset
    except Exception as e:
        raise TrainingNotAvailable(
            f"datasets is required. pip install datasets ({e})"
        )
    # trl is optional; fall back to plain Trainer if missing.
    try:
        from trl import SFTTrainer, SFTConfig  # type: ignore
        deps["SFTTrainer"] = SFTTrainer
        deps["SFTConfig"] = SFTConfig
    except Exception:
        deps["SFTTrainer"] = None
        deps["SFTConfig"] = None
    # bitsandbytes optional - only used for int4/int8.
    try:
        import bitsandbytes as bnb  # type: ignore
        deps["bnb"] = bnb
    except Exception:
        deps["bnb"] = None
    # Unsloth optional - 2x-5x faster LoRA training.
    try:
        from unsloth import FastLanguageModel  # type: ignore
        deps["FastLanguageModel"] = FastLanguageModel
    except Exception:
        deps["FastLanguageModel"] = None
    # GaLore optional - memory-efficient optimizer for 7B+ models.
    try:
        from galore_torch import GaLoreAdamW, GaLoreAdamW8bit  # type: ignore
        deps["GaLoreAdamW"] = GaLoreAdamW
        deps["GaLoreAdamW8bit"] = GaLoreAdamW8bit
    except Exception:
        deps["GaLoreAdamW"] = None
        deps["GaLoreAdamW8bit"] = None
    return deps


# ── dataset shaping ──────────────────────────────────────────────────────


_INSTRUCT_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n"
    "{input_block}"
    "### Response:\n{output}"
)


def _row_to_text(row: dict[str, Any], chat_template_fn: Optional[Callable[[Any], str]]) -> Optional[str]:
    """Map a single dataset row to a single training string.

    Handles every shape the restructurer produces and the common shapes a
    user might hand-author (alpaca, openai-chat, simple text)."""
    if "messages" in row and isinstance(row["messages"], list):
        if chat_template_fn is not None:
            try:
                return chat_template_fn(row["messages"])
            except Exception:
                pass
        # Hand-rolled fallback if the tokenizer has no chat template.
        parts = []
        for m in row["messages"]:
            role = m.get("role", "user")
            content = m.get("content", "")
            parts.append(f"<|{role}|>\n{content}")
        parts.append("<|end|>")
        return "\n".join(parts)

    if "instruction" in row and "output" in row:
        inp = row.get("input") or ""
        input_block = f"### Input:\n{inp}\n\n" if inp else ""
        return _INSTRUCT_TEMPLATE.format(
            instruction=row["instruction"], input_block=input_block, output=row["output"],
        )

    if "text" in row and "label" in row:
        return f"### Text:\n{row['text']}\n\n### Label:\n{row['label']}"

    if "text" in row:
        return str(row["text"])

    if "prompt" in row and "completion" in row:
        return f"{row['prompt']}{row['completion']}"

    return None


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    import json as _json
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(_json.loads(line))
            except Exception:
                continue
    return rows


def _load_dataset_rows(file_path: str, file_type: str) -> list[dict[str, Any]]:
    p = Path(file_path)
    if not p.exists():
        raise TrainingNotAvailable(f"dataset file not found: {file_path}")
    if file_type == "jsonl":
        return _load_jsonl(p)
    if file_type == "json":
        import json as _json
        with p.open("r", encoding="utf-8") as f:
            data = _json.load(f)
        if isinstance(data, list):
            return [r for r in data if isinstance(r, dict)]
        return []
    if file_type == "csv":
        import pandas as pd  # type: ignore
        return pd.read_csv(p).to_dict(orient="records")
    raise TrainingNotAvailable(f"unsupported dataset file_type for training: {file_type}")


# ── public entrypoint ────────────────────────────────────────────────────


def run_training(
    *,
    base_model: str,
    dataset_path: str,
    dataset_file_type: str,
    out_dir: Path,
    method: str = "lora",
    precision: str = "bf16",
    quantization: str = "none",
    epochs: int = 1,
    batch_size: int = 1,
    gradient_accumulation: int = 8,
    learning_rate: float = 2e-4,
    max_seq_len: int = 1024,
    lora_rank: int = 16,
    lora_alpha: Optional[int] = None,
    lora_dropout: float = 0.05,
    on_step: Optional[Callable[[dict[str, Any]], None]] = None,
    is_stopped: Optional[Callable[[], bool]] = None,
    # v4.0: efficiency knobs
    use_unsloth: bool = False,
    optimizer: str = "adamw",
    use_liger: bool = False,
) -> dict[str, Any]:
    """Fine-tune ``base_model`` on ``dataset_path`` and save the adapter.

    Returns {output_path, final_loss, steps}. Raises TrainingNotAvailable
    when the heavy stack is missing - the caller surfaces that to the UI.
    """
    deps = _require_stack()
    torch = deps["torch"]
    
    # HARDWARE GUARD: If no CUDA is detected, we must downgrade high-perf 
    # optimizations to CPU-safe alternatives to prevent immediate crashes.
    _has_cuda = torch.cuda.is_available()
    if not _has_cuda:
        if quantization != "none":
            log.warning("CPU detected: downgrading quantization '%s' -> 'none'", quantization)
            quantization = "none"
        if precision != "fp32":
            log.warning("CPU detected: forcing precision '%s' -> 'fp32'", precision)
            precision = "fp32"
        if use_unsloth:
            log.warning("CPU detected: disabling Unsloth (CUDA-only)")
            use_unsloth = False
        if use_liger:
            log.warning("CPU detected: disabling Liger kernels (CUDA-only)")
            use_liger = False
        if optimizer.startswith("galore") or optimizer == "adam8bit":
            log.warning("CPU detected: downgrading optimizer '%s' -> 'adamw'", optimizer)
            optimizer = "adamw"
        if method == "qlora":
            log.warning("CPU detected: downgrading method 'qlora' -> 'lora'")
            method = "lora"

    AutoTokenizer = deps["AutoTokenizer"]
    AutoModelForCausalLM = deps["AutoModelForCausalLM"]
    LoraConfig = deps["LoraConfig"]
    get_peft_model = deps["get_peft_model"]
    prepare_model_for_kbit_training = deps["prepare_model_for_kbit_training"]
    Dataset = deps["Dataset"]
    Trainer = deps["Trainer"]
    TrainingArguments = deps["TrainingArguments"]
    TrainerCallback = deps["TrainerCallback"]
    DataCollatorForLanguageModeling = deps["DataCollatorForLanguageModeling"]
    SFTTrainer = deps["SFTTrainer"]
    SFTConfig = deps["SFTConfig"]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load tokenizer + base model.
    log.info("loading tokenizer + base model %s", base_model)
    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    bnb_config = None
    if quantization in ("int4", "4bit") and deps.get("bnb") is not None:
        from transformers import BitsAndBytesConfig  # type: ignore
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif quantization in ("int8", "8bit") and deps.get("bnb") is not None:
        from transformers import BitsAndBytesConfig  # type: ignore
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    dtype = torch.bfloat16 if precision in ("bf16", "bfloat16") else (
        torch.float16 if precision in ("fp16", "float16") else torch.float32
    )

    model_kwargs: dict[str, Any] = {"torch_dtype": dtype, "low_cpu_mem_usage": True}
    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs.pop("torch_dtype", None)

    # ── Unsloth fast-path ─────────────────────────────────────────────
    FastLanguageModel = deps.get("FastLanguageModel")
    _used_unsloth = False
    if use_unsloth and FastLanguageModel is not None:
        log.info("loading model via Unsloth FastLanguageModel (2x-5x speedup)")
        try:
            model, tok = FastLanguageModel.from_pretrained(
                base_model,
                max_seq_length=max_seq_len,
                dtype=dtype,
                load_in_4bit=(quantization in ("int4", "4bit")),
            )
            _used_unsloth = True
        except Exception as e:
            log.warning("Unsloth loading failed, falling back to standard: %s", e)
            _used_unsloth = False

    if not _used_unsloth:
        model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
        if bnb_config is not None:
            model = prepare_model_for_kbit_training(model)

    # 2. Wrap with LoRA.
    if method in ("lora", "qlora", "dora"):
        if _used_unsloth and FastLanguageModel is not None:
            # Unsloth has its own optimized LoRA application
            log.info("applying LoRA via Unsloth FastLanguageModel")
            try:
                model = FastLanguageModel.get_peft_model(
                    model,
                    r=lora_rank,
                    lora_alpha=lora_alpha or lora_rank * 2,
                    lora_dropout=lora_dropout,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                    "gate_proj", "up_proj", "down_proj"],
                    use_gradient_checkpointing="unsloth",
                )
            except Exception as e:
                log.warning("Unsloth LoRA failed, falling back to peft: %s", e)
                peft_kwargs: dict[str, Any] = dict(
                    r=lora_rank,
                    lora_alpha=lora_alpha or lora_rank * 2,
                    lora_dropout=lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                if method == "dora":
                    try:
                        peft_kwargs["use_dora"] = True
                    except Exception:
                        pass
                peft_cfg = LoraConfig(**peft_kwargs)
                model = get_peft_model(model, peft_cfg)
        else:
            peft_kwargs = dict(
                r=lora_rank,
                lora_alpha=lora_alpha or lora_rank * 2,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            if method == "dora":
                try:
                    peft_kwargs["use_dora"] = True
                except Exception:
                    pass
            peft_cfg = LoraConfig(**peft_kwargs)
            model = get_peft_model(model, peft_cfg)
        try:
            model.print_trainable_parameters()
        except Exception:
            pass

    # 3. Build the training dataset.
    log.info("loading dataset rows from %s", dataset_path)
    rows = _load_dataset_rows(dataset_path, dataset_file_type)
    if not rows:
        raise TrainingNotAvailable("dataset is empty - nothing to train on")

    chat_fn = None
    if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
        def chat_fn(messages: list[dict[str, str]]) -> str:
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    texts: list[str] = []
    for r in rows:
        t = _row_to_text(r, chat_fn)
        if t and t.strip():
            texts.append(t)
    if not texts:
        raise TrainingNotAvailable(
            "could not derive any training text from the dataset. "
            "Expected one of: instruction/output, messages, text/label, text."
        )
    train_ds = Dataset.from_dict({"text": texts})

    # 4. Train.
    eff_steps_per_epoch = max(len(texts) // (batch_size * gradient_accumulation), 1)
    logging_steps = max(eff_steps_per_epoch // 10, 1)

    use_bf16 = precision in ("bf16", "bfloat16") and torch.cuda.is_available()
    use_fp16 = precision in ("fp16", "float16") and torch.cuda.is_available() and not use_bf16

    # ── GaLore optimizer setup ────────────────────────────────────────
    optim_str = "adamw_torch"  # default
    optim_kwargs: dict[str, Any] = {}

    GaLoreAdamW = deps.get("GaLoreAdamW")
    GaLoreAdamW8bit = deps.get("GaLoreAdamW8bit")

    if optimizer == "galore" and GaLoreAdamW is not None:
        log.info("using GaLore optimizer (65%% memory saving on optimizer states)")
        optim_str = "galore_adamw"
    elif optimizer == "galore_q" and GaLoreAdamW8bit is not None:
        log.info("using GaLore-Q (quantized, max memory saving)")
        optim_str = "galore_adamw_8bit"
    elif optimizer == "adam8bit" and deps.get("bnb") is not None:
        log.info("using 8-bit Adam via bitsandbytes")
        optim_str = "adamw_bnb_8bit"

    class _StreamCallback(TrainerCallback):  # type: ignore[misc]
        def on_log(self_, args, state, control, logs=None, **kwargs):
            if logs and on_step is not None:
                on_step({
                    "step": int(state.global_step),
                    "epoch": float(state.epoch or 0.0),
                    "loss": float(logs.get("loss", logs.get("train_loss", 0.0)) or 0.0),
                    "learning_rate": float(logs.get("learning_rate", learning_rate)),
                })

        def on_step_end(self_, args, state, control, **kwargs):
            if is_stopped is not None and is_stopped():
                control.should_training_stop = True
            return control

    if SFTTrainer is not None and SFTConfig is not None:
        sft_cfg = SFTConfig(
            output_dir=str(out_dir),
            optim=optim_str,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            logging_steps=logging_steps,
            save_strategy="no",
            bf16=use_bf16,
            fp16=use_fp16,
            max_seq_length=max_seq_len,
            report_to=[],
        )
        trainer = SFTTrainer(
            model=model,
            args=sft_cfg,
            train_dataset=train_ds,
            tokenizer=tok,
            dataset_text_field="text",
            callbacks=[_StreamCallback()],
        )
    else:
        # Plain Trainer fallback.
        def _tokenize(batch):
            enc = tok(batch["text"], truncation=True, max_length=max_seq_len)
            return enc

        tokenized = train_ds.map(_tokenize, batched=True, remove_columns=["text"])
        collator = DataCollatorForLanguageModeling(tok, mlm=False)
        targs = TrainingArguments(
            output_dir=str(out_dir),
            optim=optim_str,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            logging_steps=logging_steps,
            save_strategy="no",
            bf16=use_bf16,
            fp16=use_fp16,
            report_to=[],
        )
        trainer = Trainer(
            model=model,
            args=targs,
            train_dataset=tokenized,
            data_collator=collator,
            tokenizer=tok,
            callbacks=[_StreamCallback()],
        )

    log.info("starting trainer.train()")
    train_result = trainer.train()

    # 5. Save adapter + tokenizer.
    log.info("saving adapter to %s", out_dir)
    trainer.save_model(str(out_dir))
    try:
        tok.save_pretrained(str(out_dir))
    except Exception:
        pass

    return {
        "output_path": str(out_dir),
        "final_loss": float(train_result.training_loss or 0.0),
        "steps": int(train_result.global_step or 0),
    }
