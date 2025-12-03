# services/preprocessing_engine.py
import os
import json
import time
import joblib
import psutil
import gzip
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable
import numpy as np
import pandas as pd
import dask.dataframe as dd  # For streaming large files
import structlog
from config.settings import settings
from services.ops.tabular import OP_REGISTRY as tabular_registry
from services.ops.nlp import OP_REGISTRY as nlp_registry
from services.ops.shared import OP_REGISTRY as shared_registry
logger = structlog.get_logger()

# Default phase order for legacy tabular mode
DEFAULT_TABULAR_ORDER = [
    "cleaning", "impute", "type_fix", "encoding", "scaling", "feature_engineering",
    "feature_selection", "row_ops"
]

# Op dependencies for DAG validation (e.g., impute before encoding)
TABULAR_DEPS = {
    "impute": ["cleaning"],
    "type_fix": ["cleaning"],
    "encoding": ["impute", "type_fix"],
    "scaling": ["encoding"],
    "feature_engineering": ["scaling"],
    "feature_selection": ["feature_engineering"],
    "row_ops": ["feature_selection"]
}
NLP_DEPS = {
    "cleaning": [],
    "tokenization": ["cleaning"],
    "prompt_construction": ["tokenization"],
    "chunking": ["prompt_construction"],
    "dedup": ["chunking"],
    "pii_removal": ["cleaning"],
    "jsonl_prep": ["chunking", "pii_removal", "dedup"],
    "splitting": ["jsonl_prep"],
    "metadata": ["splitting"]
}

class PreprocessingEngine:
    def __init__(self, upload_dir: str):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes = settings.max_file_size_mb * 1024 * 1024
        self.chunk_size = settings.max_chunk_size  # From settings

    def _check_system_limits(self, file_path: str):
        size = os.path.getsize(file_path)
        if size > self.max_bytes:
            raise MemoryError(f"File {size / (1024*1024):.2f}MB > {settings.max_file_size_mb}MB")
        mem = psutil.virtual_memory().available / (1024**3)  # GB
        if mem < 2:  # Arbitrary threshold
            logger.warning("Low memory; using Dask streaming")
        cpu = psutil.cpu_percent(interval=1)
        if cpu > 90:
            raise RuntimeError("High CPU usage; throttle")

    def _safe_read_tabular(self, file_path: str, nrows: Optional[int] = None, streaming: bool = False) -> pd.DataFrame:
        if streaming or os.path.getsize(file_path) > 100 * 1024**2:
            # Use Dask for large files (chunked reading)
            ddf = dd.read_csv(file_path, blocksize=self.chunk_size * 1024**2, on_bad_lines='skip')
            return ddf.head(nrows).compute() if nrows else ddf.compute()
        read_kwargs = {"encoding": "utf-8", "engine": "python", "on_bad_lines": "skip"}
        if nrows:
            read_kwargs["nrows"] = nrows
        return pd.read_csv(file_path, **read_kwargs)

    def _validate_dag(self, plan: Dict[str, Any], mode: str) -> List[str]:
        deps = TABULAR_DEPS if mode == "tabular" else NLP_DEPS
        pipeline = [step.get("op") for step in plan.get("pipeline", [])]
        errors = []
        for op in pipeline:
            req_deps = deps.get(op, [])
            prev_ops = pipeline[:pipeline.index(op)]
            missing = [d for d in req_deps if d not in prev_ops]
            if missing:
                errors.append(f"Op '{op}' requires {missing}; reorder pipeline")
        return errors

    def _rollback_step(self, backup_path: Path, target_path: Path):
        if backup_path.exists():
            backup_path.replace(target_path)
            logger.info("Rolled back to previous step")

    def _store_transformers(self, sidecar_path: Path, transformers: Dict[str, Any]):
        trans_path = sidecar_path.with_name(sidecar_path.stem + ".transformers.pkl")
        joblib.dump(transformers, trans_path)
        return str(trans_path)

    def run(self, file_path: str, plan: Dict[str, Any], provenance: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        mode = (plan.get("mode") or "tabular").lower()
        if mode not in ("tabular", "nlp_finetune"):
            logger.warning("Unknown mode in plan, defaulting to tabular", mode=mode)
            mode = "tabular"
        self._check_system_limits(file_path)
        errors = self._validate_dag(plan, mode)
        if errors:
            raise ValueError("DAG validation failed: " + "; ".join(errors))
        streaming = os.path.getsize(file_path) > 100 * 1024**2
        # Backup original for rollback
        original = Path(file_path)
        backup = self.upload_dir / f"backup_{original.name}"
        original.replace(backup)
        try:
            if mode == "tabular":
                result = self._run_tabular(backup, plan, provenance or {}, streaming)
            else:
                result = self._run_nlp(backup, plan, provenance or {}, streaming)
            return result
        except Exception as e:
            logger.error("Preprocessing failed", error=str(e))
            self._rollback_step(backup, original)
            raise

    def _resolve_op(self, op_name: str, mode: str) -> Optional[Callable]:
        if mode == "tabular":
            return tabular_registry.get(op_name)
        elif mode == "nlp_finetune":
            return nlp_registry.get(op_name)
        return shared_registry.get(op_name)

    def _run_tabular(self, file_path: str, plan: Dict[str, Any], provenance: Dict[str, Any], streaming: bool) -> Dict[str, Any]:
        t0 = time.time()
        warnings: List[str] = []
        meta_store: Dict[str, Any] = {}
        transformers: Dict[str, Any] = {}
        df = self._safe_read_tabular(file_path, streaming=streaming)
        original_name = Path(file_path).name
        pipeline = plan.get("pipeline", [])
        backups = []
        for i, step in enumerate(pipeline):
            op_name = step.get("op")
            if not op_name:
                continue
            func = self._resolve_op(op_name, "tabular")
            if func is None:
                warnings.append(f"Unknown op '{op_name}'")
                continue
            # Backup for this step
            backup = self.upload_dir / f"step_{i}_backup.csv"
            df.to_csv(backup, index=False)
            backups.append(backup)
            try:
                # Apply op
                df_new, meta_update, step_warns = func(df, step_config=step, meta_store=meta_store)
                # If sklearn-compatible, wrap for reversibility
                if "sklearn" in op_name:
                    from services.ops.tabular.build_sklearn_step import build_sklearn_step
                    pipe = build_sklearn_step(op_name, step)
                    transformers[op_name] = pipe
                df = df_new
                meta_store.update(meta_update)
                warnings.extend(step_warns)
            except Exception as e:
                logger.error("Step failed", op=op_name, error=str(e))
                self._rollback_step(backups[i], Path(file_path))
                raise
        # Output
        preprocessed_name = f"preprocessed_{original_name}"
        preprocessed_path = self.upload_dir / preprocessed_name
        self._atomic_write_csv(df, preprocessed_path)
        sidecar = {
            "mode": "tabular",
            "plan": plan,
            "created_at": time.time(),
            "duration_seconds": time.time() - t0,
            "warnings": warnings,
            "transformer_meta": meta_store,
            "transformers_path": self._store_transformers(preprocessed_path.with_suffix(".json"), transformers),
            "rows": int(len(df)),
            "columns": list(df.columns),
            "provenance": provenance,
            "cpu_usage": psutil.cpu_percent(),
            "mem_usage_gb": psutil.virtual_memory().used / (1024**3)
        }
        self._write_sidecar(preprocessed_path, sidecar)
        preview = df.head(10).to_dict(orient="records")
        return {
            "preprocessed_path": str(preprocessed_path),
            "sidecar_path": str(preprocessed_path.with_suffix(".preprocess.json")),
            "preview": preview,
            "meta": sidecar,
        }

    def _run_nlp(self, file_path: str, plan: Dict[str, Any], provenance: Dict[str, Any], streaming: bool) -> Dict[str, Any]:
        t0 = time.time()
        warnings: List[str] = []
        meta_store: Dict[str, Any] = {}
        transformers: Dict[str, Any] = {}
        df = self._safe_read_tabular(file_path, streaming=streaming)
        original_name = Path(file_path).name
        pipeline = plan.get("pipeline", [])
        backups = []
        for i, step in enumerate(pipeline):
            op_name = step.get("op")
            if not op_name:
                continue
            func = self._resolve_op(op_name, "nlp_finetune")
            if func is None:
                warnings.append(f"Unknown op '{op_name}'")
                continue
            backup = self.upload_dir / f"step_{i}_backup.csv"
            df.to_csv(backup, index=False)
            backups.append(backup)
            try:
                df_new, meta_update, step_warns = func(df, step_config=step, meta_store=meta_store, nlp_config=plan.get("nlp", {}))
                df = df_new
                meta_store.update(meta_update)
                warnings.extend(step_warns)
            except Exception as e:
                logger.error("NLP step failed", op=op_name, error=str(e))
                self._rollback_step(backups[i], Path(file_path))
                raise
        # Output as JSONL for NLP
        output_format = plan.get("output_format", "jsonl")
        records = df.to_dict(orient="records")
        if output_format == "jsonl":
            preprocessed_name = f"preprocessed_{original_name}.jsonl"
            preprocessed_path = self.upload_dir / preprocessed_name
            self._atomic_write_jsonl(records, preprocessed_path)
        else:
            preprocessed_name = f"preprocessed_{original_name}"
            preprocessed_path = self.upload_dir / preprocessed_name
            self._atomic_write_csv(pd.DataFrame(records), preprocessed_path)
        sidecar = {
            "mode": "nlp_finetune",
            "plan": plan,
            "created_at": time.time(),
            "duration_seconds": time.time() - t0,
            "warnings": warnings,
            "transformer_meta": meta_store,
            "transformers_path": self._store_transformers(preprocessed_path.with_suffix(".json"), transformers),
            "rows": int(len(df)),
            "columns": list(df.columns),
            "provenance": provenance,
            "cpu_usage": psutil.cpu_percent(),
            "mem_usage_gb": psutil.virtual_memory().used / (1024**3)
        }
        self._write_sidecar(preprocessed_path, sidecar)
        preview = records[:10]
        return {
            "preprocessed_path": str(preprocessed_path),
            "sidecar_path": str(preprocessed_path.with_suffix(".preprocess.json")),
            "preview": preview,
            "meta": sidecar,
        }

    def _atomic_write_csv(self, df: pd.DataFrame, dest: Path):
        partial = dest.with_suffix(dest.suffix + ".partial")
        df.to_csv(partial, index=False)
        partial.replace(dest)

    def _atomic_write_jsonl(self, records: List[Dict], dest: Path):
        partial = dest.with_suffix(dest.suffix + ".partial")
        with open(partial, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        partial.replace(dest)
        # Gzip backup for artifact safety
        with gzip.open(dest.with_suffix(".gz"), "wt", encoding="utf-8") as gz:
            for rec in records:
                gz.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _write_sidecar(self, dest: Path, sidecar: Dict[str, Any]):
        sidecar_path = dest.with_suffix(".preprocess.json")
        with open(sidecar_path, "w", encoding="utf-8") as f:
            json.dump(sidecar, f, indent=2, default=str)