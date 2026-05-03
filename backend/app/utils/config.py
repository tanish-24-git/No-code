"""Runtime configuration loaded from environment and the .env file.

The agent's LLM is configured here with four variables (OpenClaw-style):

    LLM_PROVIDER   one of: anthropic, openai
    LLM_API_KEY    the provider's API key
    LLM_MODEL      model id (e.g. claude-sonnet-4-5, gpt-4o, llama3.1:8b)
    LLM_BASE_URL   optional base URL override (used only by the openai provider)

The openai provider speaks the OpenAI chat-completions API, which means it
also covers Ollama, LM Studio, vLLM, OpenRouter, Together, Groq, and any
other server that exposes a compatible /v1 endpoint.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


# Provider is validated against the registry at runtime, not in the type
# system, so plug-in providers can be added without changing this file.
Provider = str


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = True

    data_dir: Path = Path("./data")
    upload_dir: Path = Path("./uploads")
    models_dir: Path = Path("./models")

    # Symmetric key for encrypting stored API keys at rest.
    # Auto-generated under data/.encryption_key on first run if blank.
    encryption_key: str = ""

    # Frontend origins allowed to call the API.
    cors_origins_raw: str = "http://localhost:3000,http://127.0.0.1:3000"

    # LLM provider for the agent (env-first; UI settings override).
    llm_provider: Optional[Provider] = None
    llm_api_key: str = ""
    llm_model: str = ""
    llm_base_url: str = ""

    # Hugging Face token (optional; used to pull and push models).
    hf_token: str = ""

    log_level: str = "INFO"

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.cors_origins_raw.split(",") if o.strip()]

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.upload_dir, self.models_dir):
            p.mkdir(parents=True, exist_ok=True)
        for sub in ("datasets", "pipelines", "jobs", "inferences", "models"):
            (self.data_dir / sub).mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
