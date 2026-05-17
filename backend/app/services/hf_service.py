"""Hugging Face Hub helpers: pull a base model into the local cache, push a
trained artifact. Uses the user's HF token from settings."""
from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.api.routes.settings import get_hf_token
from app.storage import store
from app.utils.config import settings


log = logging.getLogger("finetune-studio.hf")


def _safe(value: Any, fallback: str = "unknown") -> str:
    if value is None or value == "":
        return fallback
    return str(value)


def _build_model_card(model_id: str, repo_id: str) -> str:
    """Compose a Hugging Face model card from job/pipeline/dataset records.

    Best-effort: any missing piece falls back to a placeholder so we always
    upload *some* README rather than failing the push.
    """
    model_raw = store.read("models", model_id) or {}
    model_name = model_raw.get("name") or repo_id.split("/")[-1]
    job_id = model_raw.get("job_id")
    job_raw = store.read("jobs", job_id) if job_id else None
    pipeline_id = (job_raw or {}).get("pipeline_id")
    pipeline_raw = store.read("pipelines", pipeline_id) if pipeline_id else None
    config = (pipeline_raw or {}).get("config", {}) or {}
    dataset_id = config.get("dataset_id")
    dataset_raw = store.read("datasets", dataset_id) if dataset_id else None

    base_model = _safe(config.get("base_model"))
    method = (config.get("training_method") or "lora").lower()
    quantization = (config.get("quantization") or "none").lower()
    precision = _safe(config.get("precision"))
    epochs = _safe(config.get("epochs"))
    batch = _safe(config.get("batch_size"))
    grad_accum = _safe(config.get("gradient_accumulation"), "1")
    lr = _safe(config.get("learning_rate"))
    max_seq_len = _safe(config.get("max_seq_len"))
    lora_rank = _safe(config.get("lora_rank"))

    try:
        effective_batch = int(batch) * int(grad_accum)
    except (TypeError, ValueError):
        effective_batch = "?"

    dataset_name = _safe((dataset_raw or {}).get("name"), "custom")
    dataset_rows = _safe((dataset_raw or {}).get("row_count"))
    analysis = (dataset_raw or {}).get("analysis") or {}
    dataset_format = _safe(analysis.get("format") or analysis.get("kind"))

    metrics = (job_raw or {}).get("metrics") or []
    final_loss = (job_raw or {}).get("current_loss")
    if final_loss in (None, "") and metrics:
        final_loss = metrics[-1].get("loss")
    final_loss = _safe(final_loss, "n/a")
    started = _safe((job_raw or {}).get("started_at"), "n/a")
    completed = _safe((job_raw or {}).get("completed_at"), "n/a")

    tags = ["finetune", "peft", method]
    if quantization != "none":
        tags.append(quantization)
    if quantization in ("int4", "nf4", "qlora"):
        tags.append("qlora")
    tags = list(dict.fromkeys(tags))

    yaml_lines = ["---"]
    if base_model and base_model != "unknown":
        yaml_lines.append(f"base_model: {base_model}")
    yaml_lines.append("library_name: peft")
    yaml_lines.append("license: apache-2.0")
    yaml_lines.append("tags:")
    for t in tags:
        yaml_lines.append(f"- {t}")
    if dataset_name and dataset_name not in ("custom", "unknown"):
        yaml_lines.append("datasets:")
        yaml_lines.append(f"- {dataset_name}")
    yaml_lines.append("---")

    method_label = method.upper()
    if quantization != "none":
        method_label = f"{method_label} ({quantization})"

    lora_line = ""
    if method in ("lora", "qlora", "dora"):
        lora_line = f"- **LoRA rank**: {lora_rank}\n"

    inference_snippet = (
        "```python\n"
        "from peft import PeftModel\n"
        "from transformers import AutoModelForCausalLM, AutoTokenizer\n\n"
        f'BASE = "{base_model}"\n'
        f'ADAPTER = "{repo_id}"\n\n'
        "tok = AutoTokenizer.from_pretrained(BASE)\n"
        "base = AutoModelForCausalLM.from_pretrained(BASE)\n"
        "model = PeftModel.from_pretrained(base, ADAPTER)\n\n"
        'prompt = "Hello!"\n'
        'inputs = tok(prompt, return_tensors="pt").input_ids\n'
        "out = model.generate(inputs, max_new_tokens=128)\n"
        "print(tok.decode(out[0], skip_special_tokens=True))\n"
        "```"
    )

    body = (
        f"# {model_name}\n\n"
        f"Fine-tuned from [`{base_model}`](https://huggingface.co/{base_model}) "
        f"using **FineTune Studio** — an agentic fine-tuning IDE.\n\n"
        f"## Training summary\n\n"
        f"- **Method**: {method_label}\n"
        f"- **Precision**: {precision}\n"
        f"- **Dataset**: {dataset_name} ({dataset_rows} rows, format: {dataset_format})\n"
        f"- **Epochs**: {epochs}\n"
        f"- **Effective batch size**: {batch} × {grad_accum} = {effective_batch}\n"
        f"- **Learning rate**: {lr}\n"
        f"- **Max sequence length**: {max_seq_len}\n"
        f"{lora_line}"
        f"- **Final training loss**: {final_loss}\n"
        f"- **Started**: {started}\n"
        f"- **Completed**: {completed}\n\n"
        f"## Inference\n\n"
        f"{inference_snippet}\n\n"
        f"## License\n\n"
        f"Released under apache-2.0. Adapter inherits the base model's license, "
        f"which may carry additional terms.\n\n"
        f"---\n"
        f"*Card generated by FineTune Studio.*\n"
    )

    return "\n".join(yaml_lines) + "\n\n" + body


def _write_model_card(local_path: Path, model_id: str, repo_id: str) -> None:
    try:
        content = _build_model_card(model_id, repo_id)
        target = Path(local_path) / "README.md"
        target.write_text(content, encoding="utf-8")
        log.info(f"Wrote model card to {target}")
    except Exception as e:
        log.warning(f"Failed to write model card for {model_id}: {e}")

_pull_status: dict[str, dict[str, Any]] = {}
_push_status: dict[str, dict[str, Any]] = {}


def pull_status(repo_id: str) -> dict[str, Any]:
    return _pull_status.get(repo_id, {"status": "idle", "repo_id": repo_id})


def push_status(model_id: str) -> dict[str, Any]:
    return _push_status.get(model_id, {"status": "idle", "model_id": model_id})


def start_pull(repo_id: str) -> dict[str, Any]:
    """Kick off a base-model download into HF's local cache."""
    token = get_hf_token() or None
    _pull_status[repo_id] = {"status": "pulling", "repo_id": repo_id, "started_at": datetime.now(timezone.utc).isoformat()}

    def _worker() -> None:
        try:
            from huggingface_hub import snapshot_download  # local import; heavy
            target = settings.models_dir / "base" / repo_id.replace("/", "__")
            target.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(target),
                token=token,
                local_dir_use_symlinks=False,
            )
            _pull_status[repo_id] = {
                "status": "done",
                "repo_id": repo_id,
                "local_dir": str(target),
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
            _record_pulled(repo_id, str(target))
        except Exception as e:
            _pull_status[repo_id] = {"status": "failed", "repo_id": repo_id, "error": str(e)}

    threading.Thread(target=_worker, daemon=True).start()
    return _pull_status[repo_id]


def start_push(model_id: str, repo_id: str) -> dict[str, Any]:
    """Push a trained model directory to HF Hub."""
    token = get_hf_token()
    if not token:
        raise ValueError("HF token not set")
    raw = store.read("models", model_id)
    if not raw:
        raise ValueError("Model not found")
    local_path = raw.get("local_path")
    if not local_path or not Path(local_path).exists():
        raise ValueError("Model path missing on disk")

    _push_status[model_id] = {"status": "pushing", "repo_id": repo_id, "model_id": model_id}

    def _worker() -> None:
        try:
            log.info(f"Starting HF push for model {model_id} to {repo_id}")
            from huggingface_hub import HfApi, create_repo  # local import; heavy
            
            log.info(f"Verifying/creating repo {repo_id}")
            create_repo(repo_id, token=token, exist_ok=True, private=False)
            
            log.info(f"Uploading folder {local_path} to {repo_id}")
            # Ensure local_path is a string and exists
            p = Path(local_path)
            if not p.exists():
                raise ValueError(f"Local path {local_path} does not exist")

            _write_model_card(p, model_id, repo_id)

            api = HfApi()
            api.upload_folder(
                folder_path=str(local_path),
                repo_id=repo_id,
                token=token,
                commit_message=f"Upload trained model {model_id}"
            )
            
            log.info(f"Upload complete for {model_id}. Updating model record.")
            # Refetch raw in case it changed
            raw_updated = store.read("models", model_id) or raw
            raw_updated["hf_repo_id"] = repo_id
            raw_updated["is_pushed_to_hub"] = True
            raw_updated["push_status"] = "done"
            raw_updated["pushed_at"] = datetime.now(timezone.utc).isoformat()
            
            store.write("models", model_id, raw_updated)
            _push_status[model_id] = {"status": "done", "repo_id": repo_id, "model_id": model_id}
            log.info(f"Model {model_id} record updated with push status.")
            
        except Exception as e:
            log.exception(f"Failed to push model {model_id} to HF")
            _push_status[model_id] = {
                "status": "failed",
                "repo_id": repo_id,
                "model_id": model_id,
                "error": str(e),
                "failed_at": datetime.now(timezone.utc).isoformat()
            }

    threading.Thread(target=_worker, daemon=True).start()
    return _push_status[model_id]


def _record_pulled(repo_id: str, local_dir: str) -> None:
    """Track a base model so the UI can list it as a pull-source."""
    rec_id = f"base_{repo_id.replace('/', '__')}"
    raw = {
        "id": rec_id,
        "kind": "base",
        "repo_id": repo_id,
        "local_path": local_dir,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    store.write("models", rec_id, raw)
