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
