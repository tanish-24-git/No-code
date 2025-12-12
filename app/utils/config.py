"""
Configuration management using Pydantic Settings.
All configuration is loaded from environment variables.
"""
from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    debug: bool = Field(default=False, description="Debug mode")
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        description="CORS allowed origins"
    )
    
    # Redis Configuration
    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis connection URL")
    redis_max_connections: int = Field(default=50, description="Redis connection pool size")
    
    # PostgreSQL Configuration
    postgres_host: str = Field(default="localhost", description="PostgreSQL host")
    postgres_port: int = Field(default=5432, description="PostgreSQL port")
    postgres_user: str = Field(default="llm_platform", description="PostgreSQL user")
    postgres_password: str = Field(default="changeme123", description="PostgreSQL password")
    postgres_db: str = Field(default="llm_platform", description="PostgreSQL database name")
    database_url: str = Field(
        default="postgresql://llm_platform:changeme123@localhost:5432/llm_platform",
        description="Full database URL"
    )
    
    # MinIO Object Storage
    minio_endpoint: str = Field(default="localhost:9000", description="MinIO endpoint")
    minio_access_key: str = Field(default="minioadmin", description="MinIO access key")
    minio_secret_key: str = Field(default="minioadmin", description="MinIO secret key")
    minio_secure: bool = Field(default=False, description="Use HTTPS for MinIO")
    minio_bucket_datasets: str = Field(default="datasets", description="Datasets bucket")
    minio_bucket_models: str = Field(default="models", description="Models bucket")
    minio_bucket_checkpoints: str = Field(default="checkpoints", description="Checkpoints bucket")
    minio_bucket_artifacts: str = Field(default="artifacts", description="Artifacts bucket")
    
    # GPU Configuration
    gpu_enabled: bool = Field(default=True, description="Enable GPU support")
    cuda_visible_devices: str = Field(default="0", description="CUDA visible devices")
    
    # Training Defaults
    default_lora_r: int = Field(default=16, description="Default LoRA rank")
    default_lora_alpha: int = Field(default=32, description="Default LoRA alpha")
    default_lora_dropout: float = Field(default=0.05, description="Default LoRA dropout")
    default_batch_size: int = Field(default=4, description="Default batch size")
    default_learning_rate: float = Field(default=2e-4, description="Default learning rate")
    default_max_epochs: int = Field(default=3, description="Default max epochs")
    
    # Logging
    log_level: str = Field(default="INFO", description="Log level")
    log_retention_days: int = Field(default=7, description="Log retention in days")
    
    # Worker Configuration
    worker_concurrency: int = Field(default=2, description="Worker concurrency")
    worker_max_retries: int = Field(default=3, description="Max retries for failed tasks")


# Global settings instance
settings = Settings()
