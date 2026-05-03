"""Provider registry. Adding a new LLM provider is a one-entry change here.

Each provider has:
    engine        which streaming function in providers.py to use
                  ("anthropic" or "openai" — those are the only two SDKs we ship)
    label         human-friendly name shown in the UI
    base_url      default API base URL; the user can override it in settings
    needs_key     whether an API key is mandatory (False for local servers)
    sample_models list of model ids shown as suggestions in the UI

The OpenAI engine works for any server that exposes a chat-completions API
with the standard request shape, which is most of them. We document the
default base_url per provider so users do not have to look it up.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    engine: str  # "anthropic" | "openai"
    label: str
    base_url: str
    needs_key: bool
    sample_models: tuple[str, ...]
    notes: str = ""


PROVIDERS: dict[str, ProviderSpec] = {
    # ── Native engines ─────────────────────────────────────────────────────
    "anthropic": ProviderSpec(
        name="anthropic",
        engine="anthropic",
        label="Anthropic (Claude)",
        base_url="https://api.anthropic.com",
        needs_key=True,
        sample_models=(
            "claude-sonnet-4-5",
            "claude-opus-4-5",
            "claude-haiku-4-5",
            "claude-3-5-sonnet-latest",
        ),
    ),
    "openai": ProviderSpec(
        name="openai",
        engine="openai",
        label="OpenAI",
        base_url="https://api.openai.com/v1",
        needs_key=True,
        sample_models=("gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini"),
    ),
    # ── Cloud providers via OpenAI-compatible endpoints ────────────────────
    "gemini": ProviderSpec(
        name="gemini",
        engine="openai",
        label="Google Gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        needs_key=True,
        sample_models=(
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
        ),
        notes="Uses Gemini's OpenAI-compatible endpoint. Keys from aistudio.google.com.",
    ),
    "groq": ProviderSpec(
        name="groq",
        engine="openai",
        label="Groq",
        base_url="https://api.groq.com/openai/v1",
        needs_key=True,
        sample_models=(
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ),
    ),
    "grok": ProviderSpec(
        name="grok",
        engine="openai",
        label="xAI Grok",
        base_url="https://api.x.ai/v1",
        needs_key=True,
        sample_models=("grok-2-latest", "grok-2-mini"),
    ),
    "deepseek": ProviderSpec(
        name="deepseek",
        engine="openai",
        label="DeepSeek",
        base_url="https://api.deepseek.com/v1",
        needs_key=True,
        sample_models=("deepseek-chat", "deepseek-reasoner"),
    ),
    "mistral": ProviderSpec(
        name="mistral",
        engine="openai",
        label="Mistral AI",
        base_url="https://api.mistral.ai/v1",
        needs_key=True,
        sample_models=(
            "mistral-large-latest",
            "mistral-small-latest",
            "open-mistral-nemo",
        ),
    ),
    "together": ProviderSpec(
        name="together",
        engine="openai",
        label="Together AI",
        base_url="https://api.together.xyz/v1",
        needs_key=True,
        sample_models=(
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "meta-llama/Llama-3.1-8B-Instruct-Turbo",
            "Qwen/Qwen2.5-72B-Instruct-Turbo",
        ),
    ),
    "fireworks": ProviderSpec(
        name="fireworks",
        engine="openai",
        label="Fireworks AI",
        base_url="https://api.fireworks.ai/inference/v1",
        needs_key=True,
        sample_models=(
            "accounts/fireworks/models/llama-v3p1-70b-instruct",
            "accounts/fireworks/models/llama-v3p1-8b-instruct",
            "accounts/fireworks/models/qwen2p5-72b-instruct",
        ),
    ),
    "openrouter": ProviderSpec(
        name="openrouter",
        engine="openai",
        label="OpenRouter",
        base_url="https://openrouter.ai/api/v1",
        needs_key=True,
        sample_models=(
            "anthropic/claude-3.5-sonnet",
            "openai/gpt-4o",
            "meta-llama/llama-3.1-70b-instruct",
            "google/gemini-2.0-flash-exp",
        ),
        notes="Single key, hundreds of models. See openrouter.ai/models.",
    ),
    "perplexity": ProviderSpec(
        name="perplexity",
        engine="openai",
        label="Perplexity",
        base_url="https://api.perplexity.ai",
        needs_key=True,
        sample_models=("sonar", "sonar-pro", "sonar-reasoning"),
    ),
    "cohere": ProviderSpec(
        name="cohere",
        engine="openai",
        label="Cohere",
        base_url="https://api.cohere.ai/compatibility/v1",
        needs_key=True,
        sample_models=("command-r-plus", "command-r", "command-r7b"),
    ),
    "huggingface": ProviderSpec(
        name="huggingface",
        engine="openai",
        label="Hugging Face Inference",
        base_url="https://router.huggingface.co/v1",
        needs_key=True,
        sample_models=(
            "meta-llama/Llama-3.1-8B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct",
        ),
        notes="HF Inference Router. Use your HF token as the API key.",
    ),
    # ── Local servers (no API key needed) ──────────────────────────────────
    "ollama": ProviderSpec(
        name="ollama",
        engine="openai",
        label="Ollama (local)",
        base_url="http://localhost:11434/v1",
        needs_key=False,
        sample_models=(
            "llama3.1:8b",
            "qwen2.5:7b",
            "mistral:7b",
            "gemma2:9b",
        ),
        notes="Local. Run `ollama serve` and `ollama pull <model>` first.",
    ),
    "lmstudio": ProviderSpec(
        name="lmstudio",
        engine="openai",
        label="LM Studio (local)",
        base_url="http://localhost:1234/v1",
        needs_key=False,
        sample_models=(),
        notes="Local. Whatever model you have loaded in LM Studio's server tab.",
    ),
    "vllm": ProviderSpec(
        name="vllm",
        engine="openai",
        label="vLLM (local)",
        base_url="http://localhost:8000/v1",
        needs_key=False,
        sample_models=(),
        notes="Local. Whatever model your vLLM server is serving.",
    ),
    # ── Generic escape hatch ───────────────────────────────────────────────
    "custom": ProviderSpec(
        name="custom",
        engine="openai",
        label="Custom (OpenAI-compatible)",
        base_url="",
        needs_key=False,
        sample_models=(),
        notes="Bring your own base URL. Anything that speaks OpenAI's chat-completions API.",
    ),
}


PROVIDER_NAMES: tuple[str, ...] = tuple(PROVIDERS.keys())


def get_spec(name: str | None) -> ProviderSpec | None:
    if not name:
        return None
    return PROVIDERS.get(name)


def resolve_base_url(name: str | None, override: str | None) -> str:
    """If the user provided a base_url override, use it. Otherwise fall back
    to the provider's default. Empty string means 'use SDK default'."""
    if override:
        return override
    spec = get_spec(name)
    return spec.base_url if spec else ""


def resolve_engine(name: str | None) -> str:
    spec = get_spec(name)
    return spec.engine if spec else "openai"
