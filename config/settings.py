# config/settings.py
import os
from pydantic_settings import BaseSettings
from typing import List, Optional

class Settings(BaseSettings):
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = True

    max_file_size_mb: int = 100
    allowed_file_extensions: List[str] = [".csv", ".xlsx", ".json", ".parquet"]
    upload_directory: str = "uploads"

    cors_origins: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000", "http://127.0.0.1:8000", "http://localhost:8000"]

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

settings = Settings()
