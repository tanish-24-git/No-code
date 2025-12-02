# services/preprocessing_engine.py
import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import structlog

from config.settings import settings
from services.ops import tabular_ops
from services.ops import nlp_ops

logger = structlog.get_logger()

# Default phase order for legacy tabular mode
DEFAULT_TABULAR_ORDER = ["cleaning", "impute", "encoding", "scaling", "feature_engineering"]


class PreprocessingEngine:
    """
    Central engine that runs preprocessing for both:
      - Tabular ML ("tabular" mode)
      - NLP / Fine-tuning ("nlp_finetune" mode)

    It supports:
      1) Legacy tabular plan (global_defaults + columns + order)
      2) New pipeline-style plan: a list of operations {"op": "...", ...}
         that are looked up from the appropriate OP_REGISTRY.
    """

    def __init__(self, upload_dir: str):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes = settings.max_file_size_mb * 1024 * 1024

    # -------------------------------
    # Low-level helpers
    # -------------------------------

    def _check_system_limits(self, file_path: str):
        """
        Basic guard rails: file size check etc.
        We keep it conservative; if a file is extremely huge, we fail fast
        with a clear error message instead of blowing up memory.
        """
        try:
            size = os.path.getsize(file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")

        if size > self.max_bytes:
            # You can later implement streaming processing here.
            raise MemoryError(
                f"File size {size / (1024 * 1024):.2f} MB exceeds max "
                f"limit of {settings.max_file_size_mb} MB for in-memory preprocessing."
            )

    def _atomic_write_csv(self, df: pd.DataFrame, dest: Path):
        """
        Write CSV atomically: write to temporary `.partial` then rename.
        This avoids half-written files if process crashes mid-write.
        """
        partial = dest.with_suffix(dest.suffix + ".partial")
        df.to_csv(partial, index=False)
        partial.replace(dest)

    def _atomic_write_jsonl(self, records: List[Dict[str, Any]], dest: Path):
        """
        Write JSONL atomically: write to `.partial`, then rename.
        """
        partial = dest.with_suffix(dest.suffix + ".partial")
        with open(partial, "w", encoding="utf-8") as fh:
            for row in records:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        partial.replace(dest)

    def _write_sidecar(self, dest: Path, sidecar: Dict[str, Any]):
        """
        Sidecar metadata: <dest>.preprocess.json
        Stores plan, warnings, transformer meta, token stats, etc.
        """
        sidecar_path = dest.with_suffix(dest.suffix + ".preprocess.json")
        with open(sidecar_path, "w", encoding="utf-8") as fh:
            json.dump(sidecar, fh, indent=2)

    def _safe_read_tabular(self, file_path: str, nrows: Optional[int] = None) -> pd.DataFrame:
        """
        Robust CSV/Parquet/JSON reader for tabular data.
        Currently we support CSV strongly. You can extend for others.
        """
        # For now we assume csv; you can inspect suffix & route accordingly.
        read_kwargs = {
            "encoding": "utf-8",
            "engine": "python",
            "on_bad_lines": "skip"
        }
        if nrows is not None:
            read_kwargs["nrows"] = nrows
        return pd.read_csv(file_path, **read_kwargs)

    # -------------------------------
    # Public entrypoint
    # -------------------------------

    def run(
        self,
        file_path: str,
        plan: Dict[str, Any],
        provenance: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Top-level entrypoint called by services/preprocessing_service.py.

        Chooses mode based on plan["mode"]:

          - "tabular"      -> classical ML
          - "nlp_finetune" -> NLP fine-tuning / JSONL

        Returns a dict:
        {
          "preprocessed_path": "<path>",
          "sidecar_path": "<path>.preprocess.json",
          "preview": [...],
          "meta": {...}
        }
        """
        mode = (plan.get("mode") or "tabular").lower()
        if mode not in ("tabular", "nlp_finetune"):
            logger.warning("Unknown mode in plan, defaulting to tabular", mode=mode)
            mode = "tabular"

        self._check_system_limits(file_path)

        if mode == "tabular":
            return self._run_tabular(file_path, plan, provenance or {})
        else:
            return self._run_nlp(file_path, plan, provenance or {})

    # -------------------------------
    # TABULAR PIPELINE
    # -------------------------------

    def _run_tabular(
        self,
        file_path: str,
        plan: Dict[str, Any],
        provenance: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Tabular path.

        Two modes:
          1) If 'pipeline' is present in plan -> use generic op pipeline
          2) Else -> use legacy 'global_defaults + columns + order' flow
        """
        if "pipeline" in plan and isinstance(plan["pipeline"], list):
            return self._run_tabular_pipeline(file_path, plan, provenance)
        else:
            return self._run_tabular_legacy(file_path, plan, provenance)

    def _run_tabular_pipeline(
        self,
        file_path: str,
        plan: Dict[str, Any],
        provenance: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generic pipeline-based tabular execution.

        plan["pipeline"] = [
          { "op": "mean_impute", "columns": ["age"] },
          { "op": "standard_scale", "columns": ["age", "income"] },
          ...
        ]

        Each `op` name is resolved from tabular_ops.OP_REGISTRY.
        """
        t0 = time.time()
        warnings: List[str] = []
        meta_store: Dict[str, Any] = {}

        df = self._safe_read_tabular(file_path)
        original_name = Path(file_path).name

        for step in plan.get("pipeline", []):
            if not isinstance(step, dict):
                continue
            op_name = step.get("op")
            if not op_name:
                warnings.append("Pipeline step missing 'op' field; skipped")
                continue

            func = tabular_ops.OP_REGISTRY.get(op_name)
            if func is None:
                warnings.append(f"Unknown tabular op '{op_name}' skipped")
                continue

            try:
                df, meta_store, step_warns = tabular_ops.apply_op(
                    df=df,
                    op_name=op_name,
                    func=func,
                    step_config=step,
                    meta_store=meta_store,
                )
                if step_warns:
                    warnings.extend(step_warns)
            except Exception as e:
                # Fail-safe: capture error but stop pipeline with a clear message.
                logger.error("Tabular pipeline step failed", op=op_name, error=str(e))
                raise RuntimeError(f"Tabular op '{op_name}' failed: {e}") from e

        # Final output (CSV)
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
            "rows": int(len(df)),
            "columns": list(df.columns),
            "provenance": provenance or {},
        }
        self._write_sidecar(preprocessed_path, sidecar)

        preview = df.head(10).to_dict(orient="records")
        return {
            "preprocessed_path": str(preprocessed_path),
            "sidecar_path": str(preprocessed_path.with_suffix(preprocessed_path.suffix + ".preprocess.json")),
            "preview": preview,
            "meta": sidecar,
        }

    def _run_tabular_legacy(
        self,
        file_path: str,
        plan: Dict[str, Any],
        provenance: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Legacy path compatible with your current frontend:
          - global_defaults: { missing_strategy, encoding, scaling, scaler_method, ... }
          - columns: { col_name: {...} }
          - order: ["cleaning", "impute", "encoding", "scaling", ...]
        """
        import traceback

        t0 = time.time()
        warnings: List[str] = []
        meta_store: Dict[str, Any] = {}

        try:
            df = self._safe_read_tabular(file_path)
            original_name = Path(file_path).name

            global_defaults = plan.get("global_defaults", {}) or {}
            col_plan = plan.get("columns", {}) or {}
            order = plan.get("order", DEFAULT_TABULAR_ORDER)

            # ---- CLEANING ----
            if "cleaning" in order:
                # global cleaning
                if global_defaults.get("strip_whitespace", True):
                    df = tabular_ops.strip_whitespace(df)
                if global_defaults.get("lower_case", False):
                    df = tabular_ops.lower_case(df)

                # per-column cleaning & dropping
                for col, cfg in col_plan.items():
                    if not isinstance(cfg, dict):
                        continue
                    if cfg.get("action") == "drop":
                        if col in df.columns:
                            df = df.drop(columns=[col])
                            warnings.append(f"dropped column {col} by plan")
                            meta_store.setdefault("dropped_columns", []).append(col)
                            continue
                    if cfg.get("strip_whitespace", False):
                        df = tabular_ops.strip_whitespace(df, [col])
                    if cfg.get("lower_case", False):
                        df = tabular_ops.lower_case(df, [col])

            # ---- IMPUTATION ----
            if "impute" in order:
                missing_default = global_defaults.get("missing_strategy", "mean")

                # per-column overrides
                for col, cfg in col_plan.items():
                    if not isinstance(cfg, dict):
                        continue
                    strat = cfg.get("missing_strategy")
                    if strat:
                        df, warns = tabular_ops.apply_missing_strategy(df, strat, [col])
                        warnings.extend(warns)

                # global strategy on remaining columns
                df, warns = tabular_ops.apply_missing_strategy(df, missing_default, None)
                warnings.extend(warns)

            # ---- ENCODING ----
            if "encoding" in order:
                label_cols: List[str] = []
                onehot_cols: List[str] = []

                global_encoding = global_defaults.get("encoding", "onehot")

                for c in df.select_dtypes(include=["object", "category"]).columns:
                    cfg = col_plan.get(c, {}) or {}
                    enc = cfg.get("encoding") or global_encoding
                    if enc == "label":
                        label_cols.append(c)
                    elif enc == "onehot":
                        onehot_cols.append(c)
                    elif enc in (None, "passthrough", "keep"):
                        continue
                    else:
                        warnings.append(f"unsupported encoding {enc} for column {c} (skipped)")

                if label_cols:
                    df, meta_store = tabular_ops.label_encode_columns(df, label_cols, meta_store)
                if onehot_cols:
                    df, meta_store = tabular_ops.onehot_encode_columns(df, onehot_cols, meta_store)

            # ---- SCALING ----
            if "scaling" in order:
                scaling_default = global_defaults.get("scaling", True)
                if scaling_default:
                    scaler_method = global_defaults.get("scaler_method", "standard")
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    cols_to_scale: List[str] = []
                    for col in numeric_cols:
                        cfg = col_plan.get(col, {}) or {}
                        if cfg.get("scaling") is False:
                            continue
                        cols_to_scale.append(col)
                    if cols_to_scale:
                        df, meta_store = tabular_ops.apply_scaler(
                            df,
                            columns=cols_to_scale,
                            method=scaler_method,
                            save_meta=meta_store,
                        )

            # ---- FEATURE ENGINEERING (placeholder for phase 2) ----
            # You can later read feature engineering configs from plan["feature_engineering"]

            # Final CSV output
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
                "rows": int(len(df)),
                "columns": list(df.columns),
                "provenance": provenance or {},
            }
            self._write_sidecar(preprocessed_path, sidecar)
            preview = df.head(10).to_dict(orient="records")

            return {
                "preprocessed_path": str(preprocessed_path),
                "sidecar_path": str(preprocessed_path.with_suffix(preprocessed_path.suffix + ".preprocess.json")),
                "preview": preview,
                "meta": sidecar,
            }

        except Exception as e:
            tb = traceback.format_exc()
            logger.error("Legacy tabular preprocessing failed", error=str(e), traceback=tb)
            raise

    # -------------------------------
    # NLP PIPELINE
    # -------------------------------

    def _run_nlp(
        self,
        file_path: str,
        plan: Dict[str, Any],
        provenance: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        NLP / Fine-tuning preprocessing:

        Expects at minimum:
          plan["nlp"] = {
            "text_column": "...",         # input text column
            "target_column": "...",       # optional label/completion column
            ...
          }

        And a pipeline:
          plan["pipeline"] = [
            { "op": "text_lowercase", "field": "text" },
            { "op": "remove_urls", "field": "text" },
            { "op": "estimate_token_count", "field": "text" },
            { "op": "truncate_by_tokens", "field": "text", "max_tokens": 256 },
            { "op": "build_prompt_completion", ... },
            { "op": "to_jsonl", ... }
          ]

        We read tabular data (CSV) then operate mostly over one or more text columns.
        """
        t0 = time.time()
        warnings: List[str] = []
        meta_store: Dict[str, Any] = {}

        if "pipeline" not in plan or not isinstance(plan["pipeline"], list):
            raise ValueError("NLP mode requires a 'pipeline' list in the plan")

        nlp_cfg = plan.get("nlp", {}) or {}

        # We still read dataset via pandas for simplicity; later you can add streaming JSON/JSONL.
        df = self._safe_read_tabular(file_path)
        original_name = Path(file_path).name

        for step in plan["pipeline"]:
            if not isinstance(step, dict):
                continue
            op_name = step.get("op")
            if not op_name:
                warnings.append("NLP pipeline step missing 'op'; skipped")
                continue

            func = nlp_ops.OP_REGISTRY.get(op_name)
            if func is None:
                warnings.append(f"Unknown NLP op '{op_name}' skipped")
                continue

            try:
                df, meta_store, step_warns = nlp_ops.apply_op(
                    df=df,
                    op_name=op_name,
                    func=func,
                    step_config=step,
                    nlp_config=nlp_cfg,
                    meta_store=meta_store,
                )
                if step_warns:
                    warnings.extend(step_warns)
            except Exception as e:
                logger.error("NLP pipeline step failed", op=op_name, error=str(e))
                raise RuntimeError(f"NLP op '{op_name}' failed: {e}") from e

        # At this point, df likely contains columns like ["prompt", "completion"] or some JSON-ready fields.
        # We decide output format:
        output_format = plan.get("output_format", "jsonl").lower()
        if output_format not in ("jsonl", "csv"):
            output_format = "jsonl"

        if output_format == "jsonl":
            # Convert rows to JSON records and write JSONL
            records = df.to_dict(orient="records")
            preprocessed_name = f"preprocessed_{original_name}.jsonl"
            preprocessed_path = self.upload_dir / preprocessed_name
            self._atomic_write_jsonl(records, preprocessed_path)
        else:
            preprocessed_name = f"preprocessed_{original_name}"
            preprocessed_path = self.upload_dir / preprocessed_name
            self._atomic_write_csv(df, preprocessed_path)

        sidecar = {
            "mode": "nlp_finetune",
            "plan": plan,
            "created_at": time.time(),
            "duration_seconds": time.time() - t0,
            "warnings": warnings,
            "transformer_meta": meta_store,
            "rows": int(len(df)),
            "columns": list(df.columns),
            "provenance": provenance or {},
        }
        self._write_sidecar(preprocessed_path, sidecar)

        preview = df.head(10).to_dict(orient="records")
        return {
            "preprocessed_path": str(preprocessed_path),
            "sidecar_path": str(preprocessed_path.with_suffix(preprocessed_path.suffix + ".preprocess.json")),
            "preview": preview,
            "meta": sidecar,
        }
