# config/settings.py
import os
import json
from pydantic_settings import BaseSettings
from typing import List, Optional, Any
def _parse_env_list(value: Optional[str], default: List[str]) -> List[str]:
    """Try JSON decode first, fallback to comma-separated parsing, else default."""
    if value is None or value == "":
        return default
    # Try JSON list
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except Exception:
        pass
    # Fallback: comma separated
    try:
        parts = [p.strip() for p in value.split(",") if p.strip()]
        if parts:
            return parts
    except Exception:
        pass
    return default
class Settings(BaseSettings):
    # Server
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = True
    # Uploads
    # prefer MB setting but allow raw bytes via MAX_UPLOAD_BYTES env
    max_file_size_mb: int = 100
    max_upload_bytes: Optional[int] = None
    allowed_file_extensions: List[str] = [".csv", ".xlsx", ".json", ".parquet"]
    upload_directory: str = "uploads"
    # default CORS origins
    _DEFAULT_CORS = ["http://localhost:3000", "http://127.0.0.1:3000", "http://127.0.0.1:8000", "http://localhost:8000"]
    # read raw env first (so we can parse flexibly)
    _raw_cors_env: Optional[str] = os.getenv("CORS_ORIGINS", None)
    cors_origins: List[str] = _parse_env_list(_raw_cors_env, _DEFAULT_CORS)
    max_chunk_size: int = 10000
    default_test_size: float = 0.2
    random_state: int = 42
    # Redis
    redis_url: str = os.getenv("REDIS_URL", "redis://redis:6379/0")
    # LLM config
    llm_provider: str = os.getenv("LLM_PROVIDER", "generic")
    llm_api_url: Optional[str] = os.getenv("LLM_API_URL")
    llm_api_key: Optional[str] = os.getenv("LLM_API_KEY")
    google_project: Optional[str] = os.getenv("GOOGLE_PROJECT")
    google_location: str = os.getenv("GOOGLE_LOCATION", "us-central1")
    vertex_model_id: Optional[str] = os.getenv("VERTEX_MODEL_ID")
    class Config:
        env_file = ".env"
        # keep env parsing strict (unknown env keys will not cause exceptions, because we only use fields we declare)
        # if you want to allow extra fields via pydantic, set extra = "ignore"
# instantiate settings
settings = Settings()
# If the user set MAX_UPLOAD_BYTES (raw bytes) and didn't set MAX_FILE_SIZE_MB,
# derive MB from bytes to keep rest of the app consistent.
try:
    if settings.max_upload_bytes and not os.getenv("MAX_FILE_SIZE_MB"):
        settings.max_file_size_mb = int(settings.max_upload_bytes) // (1024 * 1024)
except Exception:
    # if anything fails default to the existing value
    pass