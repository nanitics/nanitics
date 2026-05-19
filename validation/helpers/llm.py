"""LLM and embedding client factories for validation scripts.

Reads provider-appropriate env vars and returns configured SDK clients.
Failure mode: raises ``ValueError`` naming the missing env var and the
install extra.
"""

from __future__ import annotations

import os
from typing import Literal

from nanitics.infrastructure import (
    EmbeddingClient,
    LLMClient,
)

Provider = Literal["anthropic", "openai", "mistral", "litellm"]
EmbeddingProvider = Literal["voyage"]

DEFAULT_MODELS: dict[Provider, str] = {
    "anthropic": "claude-haiku-4-5-20251001",
    "openai": "gpt-4o-mini",
    "mistral": "mistral-small-latest",
    "litellm": "anthropic/claude-haiku-4-5-20251001",
}

DEFAULT_EMBEDDING_MODELS: dict[EmbeddingProvider, str] = {
    "voyage": "voyage-3-lite",
}

DEFAULT_JUDGE_PROVIDER: Provider = "anthropic"
DEFAULT_JUDGE_MODEL: str = "claude-haiku-4-5"

_PROVIDER_ENV: dict[Provider, tuple[str, str]] = {
    "anthropic": ("ANTHROPIC_API_KEY", "anthropic"),
    "openai": ("OPENAI_API_KEY", "openai"),
    "mistral": ("MISTRAL_API_KEY", "mistral"),
    "litellm": ("ANTHROPIC_API_KEY", "litellm"),
}

_EMBEDDING_PROVIDER_ENV: dict[EmbeddingProvider, tuple[str, str]] = {
    "voyage": ("VOYAGE_API_KEY", "voyage"),
}


def _require_env(env_var: str, *, provider: str, extra: str) -> str:
    """Read an env var or raise a uniform ValueError naming the install extra."""
    value = os.environ.get(env_var)
    if not value:
        raise ValueError(f"Provider '{provider}' requires {env_var}. Install with: uv sync --extra {extra}")
    return value


def make_llm_client(
    provider: Provider = "anthropic",
    *,
    model: str | None = None,
    enable_caching: bool = False,
) -> LLMClient:
    """Construct a real LLM client from environment variables.

    Reads the provider-appropriate API key env var and constructs the
    corresponding SDK client. Validation scripts never hand-wire API keys
    or model names.

    Args:
        provider: Which SDK client to build. Defaults to Anthropic.
        model: Override the default model for this provider. If ``None``,
            uses the entry in :data:`DEFAULT_MODELS`.
        enable_caching: Enable Anthropic prompt caching. Ignored for
            non-Anthropic providers. Off by default to match the
            ``AnthropicLLMClient`` default.

    Returns:
        A configured :class:`LLMClient` ready to call.

    Raises:
        ValueError: If the required env var is missing. The message names
            the env var and the install extra.
    """
    env_var, extra = _PROVIDER_ENV[provider]
    api_key = _require_env(env_var, provider=provider, extra=extra)
    resolved_model = model or DEFAULT_MODELS[provider]

    if provider == "anthropic":
        from nanitics.infrastructure import AnthropicLLMClient

        return AnthropicLLMClient(
            model=resolved_model,
            api_key=api_key,
            enable_caching=enable_caching,
        )
    if provider == "openai":
        from nanitics.infrastructure import OpenAILLMClient

        return OpenAILLMClient(model=resolved_model, api_key=api_key)
    if provider == "mistral":
        from nanitics.specialized import MistralLLMClient

        return MistralLLMClient(model=resolved_model, api_key=api_key)
    # litellm
    from nanitics.infrastructure import LiteLLMClient

    return LiteLLMClient(model=resolved_model, api_key=api_key)


def make_embedding_client(
    provider: EmbeddingProvider = "voyage",
    *,
    model: str | None = None,
) -> EmbeddingClient:
    """Construct a real embedding client from environment variables.

    Args:
        provider: Which embedding SDK client to build. Defaults to Voyage.
        model: Override the default embedding model for this provider.

    Returns:
        A configured :class:`EmbeddingClient` ready to call.

    Raises:
        ValueError: If the required env var is missing. The message names
            the env var and the install extra.
    """
    env_var, extra = _EMBEDDING_PROVIDER_ENV[provider]
    api_key = _require_env(env_var, provider=provider, extra=extra)
    resolved_model = model or DEFAULT_EMBEDDING_MODELS[provider]

    # voyage is the only provider today
    from nanitics.infrastructure import VoyageEmbeddingClient

    return VoyageEmbeddingClient(api_key=api_key, model=resolved_model)
