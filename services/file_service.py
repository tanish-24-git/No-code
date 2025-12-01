# services/file_service.py
import os
import aiofiles
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from fastapi import UploadFile, HTTPException
import structlog
import anyio
from uuid import uuid4


from config.settings import settings
from utils.validators import file_validator
from services.insight_service import InsightService
from services.llm_service import LLMService

logger = structlog.get_logger()

class FileService:
    def __init__(self):
        self.upload_dir = Path(settings.upload_directory)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.insight_service = InsightService()
        # LLMService is optional — it can return {} if not configured
        try:
            self.llm_service = LLMService()
        except Exception as e:
            logger.warning("Failed to initialize LLMService; continuing without LLM", error=str(e))
            self.llm_service = None

    async def save_uploaded_file(self, file: UploadFile) -> str:
        """Stream Save uploaded file to disk (UUID-based name). Returns stored file path string."""
        # run light validation (filename, extension)
        await file_validator.validate_file(file)  # this will do lightweight checks

        ext = Path(file.filename).suffix or ".csv"
        # make safe generated name
        stored_name = f"{uuid4().hex}{ext}"
        stored_path = self.upload_dir / stored_name

        # stream write
        await file.seek(0)
        async with aiofiles.open(stored_path, 'wb') as f:
            while True:
                chunk = await file.read(1024 * 64)
                if not chunk:
                    break
                await f.write(chunk)

        logger.info("File saved", stored=stored_name, original=file.filename)
        # reset pointer for downstream consumers (best-effort)
        try:
            await file.seek(0)
        except Exception:
            try:
                file.file.seek(0)
            except Exception:
                pass

        return str(stored_path)

    async def analyze_dataset(self, file_path: str) -> Dict[str, Any]:
        """
        Async dataset analysis:
          - load a bounded sample synchronously in a thread using pandas
          - produce heuristic insights via InsightService
          - optionally call LLM (async) to augment suggestions (bounded sample only)
          Returns a dict matching models.responses.DatasetInsights expectations.
        """
        try:
            # read only a sample and small full summary offloaded to thread to avoid blocking
            def _read_sample_and_summary(fp: str):
                # be forgiving: try csv first; if fails, raise to outer handler
                df = pd.read_csv(fp, nrows=500)
                full_df = pd.read_csv(fp, nrows=1000)  # limited for summary calc to avoid OOM
                summary = {
                    "columns": list(full_df.columns),
                    "rows": int(sum(1 for _ in open(fp)) - 1),  # best-effort row count without loading entire file
                    "sample_rows": len(df),
                    "data_types": full_df.dtypes.astype(str).to_dict(),
                    "missing_values": full_df.isnull().sum().to_dict(),
                    "unique_values": {col: int(full_df[col].nunique()) for col in full_df.columns},
                    "file_size_mb": os.path.getsize(fp) / (1024 * 1024)
                }
                return summary, df

            summary, sample_df = await anyio.to_thread.run_sync(_read_sample_and_summary, file_path)

            # Heuristic insights (blocking small work on thread)
            insights_data = await anyio.to_thread.run_sync(self.insight_service.generate_insights, summary, sample_df)

            # Prepare sample rows CSV for LLM (bounded)
            sample_csv = sample_df.head(50).to_csv(index=False)

            # Ask LLM to augment suggestions (if configured) — make this optional and resilient
            llm_suggestions: Dict[str, Any] = {}
            if self.llm_service:
                try:
                    llm_suggestions = await self.llm_service.analyze_dataset(summary, sample_csv) or {}
                except Exception as e:
                    logger.warning("LLM analyze failed — continuing with heuristic insights", error=str(e))
                    llm_suggestions = {}
            else:
                logger.debug("LLMService not configured; skipping LLM augmentation")

            # Choose final suggested values preferring LLM if present, otherwise heuristic
            # Accept multiple possible key names returned by different LLM formats
            suggested_task_type = (
                llm_suggestions.get("suggested_task")
                or llm_suggestions.get("suggested_task_type")
                or insights_data.get("suggested_task_type")
                or "clustering"
            )

            suggested_target_column = (
                llm_suggestions.get("suggested_target")
                or llm_suggestions.get("suggested_target_column")
                or insights_data.get("suggested_target_column")
                or None
            )

            suggested_missing_strategy = (
                llm_suggestions.get("missing_value_strategy")
                or llm_suggestions.get("missing_strategy")
                or insights_data.get("suggested_missing_strategy")
                or "mean"
            )

            # Build merged result using the exact keys expected by UploadResponse/DatasetInsights
            merged: Dict[str, Any] = {
                "summary": summary,
                "insights": insights_data.get("insights", []),
                # these keys are required by Pydantic model
                "suggested_task_type": suggested_task_type,
                "suggested_target_column": suggested_target_column,
                "suggested_missing_strategy": suggested_missing_strategy,
                # keep originals for debugging / UI
                "heuristic_suggested_task_type": insights_data.get("suggested_task_type"),
                "heuristic_suggested_target_column": insights_data.get("suggested_target_column"),
                "heuristic_suggested_missing_strategy": insights_data.get("suggested_missing_strategy"),
                "llm_suggestions": llm_suggestions
            }

            # If LLM gave feature_engineering list, append to insights
            if isinstance(llm_suggestions, dict):
                fe = llm_suggestions.get("feature_engineering")
                if isinstance(fe, list) and fe:
                    merged["insights"].append("LLM suggested feature engineering: " + ", ".join(fe))

            return merged

        except Exception as e:
            logger.error(f"Error analyzing dataset: {str(e)}")
            # Keep error message readable for API clients
            raise HTTPException(status_code=500, detail=f"Failed to analyze dataset: {str(e)}")

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to prevent path traversal"""
        import re, os
        safe_name = re.sub(r'[^a-zA-Z0-9._-]', '_', os.path.basename(filename))
        return safe_name[:100]
