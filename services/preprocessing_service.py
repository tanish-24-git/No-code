from pathlib import Path
from typing import Dict, Any
import structlog
from config.settings import settings
from services.preprocessing_engine import PreprocessingEngine

logger = structlog.get_logger()

# Single global engine instance
_engine = PreprocessingEngine(settings.upload_directory)

def preprocess_dataset(file_path: str, plan: Dict[str, Any]) -> str:
    """
    Thin synchronous wrapper expected by main.py and background tasks.
    """
    if not isinstance(plan, dict):
        raise ValueError("preprocessing plan must be a dict")
    logger.info("Starting preprocessing", file=file_path, has_pipeline="pipeline" in plan, mode=plan.get("mode", "tabular"))
    result = _engine.run(file_path=file_path, plan=plan, provenance=plan.get("provenance"))
    preprocessed_path = result["preprocessed_path"]
    logger.info("Preprocessing completed", preprocessed_file=preprocessed_path)
    return preprocessed_path
