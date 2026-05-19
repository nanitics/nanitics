"""LLM provider factory for the full-stack compose.

Reads ``NANITICS_LLM_PROVIDER`` (``anthropic`` or ``openai``) and
``NANITICS_LLM_MODEL`` to construct the selected client. The provider's
API key (``ANTHROPIC_API_KEY`` or ``OPENAI_API_KEY``) must match the
selected provider. Missing values raise :class:`RuntimeError` at call time.

Unlike ``docker/observatory-dev/app.py``, this factory never falls back to
``MockLLMClient``. The full-stack compose is the real-LLM surface — a
missing key is a loud configuration failure, not a silent mock
substitution. See ``.env.example`` for the env-var inventory.
"""

from __future__ import annotations

import os

from nanitics.infrastructure import LLMClient

_PROVIDER_ENV = "NANITICS_LLM_PROVIDER"
_MODEL_ENV = "NANITICS_LLM_MODEL"
_ANTHROPIC_KEY_ENV = "ANTHROPIC_API_KEY"
_OPENAI_KEY_ENV = "OPENAI_API_KEY"

_ACCEPTED_PROVIDERS = ("anthropic", "openai")


def build_llm_client() -> LLMClient:
    """Construct the ``LLMClient`` selected by ``NANITICS_LLM_PROVIDER``.

    Returns:
        An :class:`AnthropicLLMClient` or :class:`OpenAILLMClient`,
        depending on the provider selected.

    Raises:
        RuntimeError: When ``NANITICS_LLM_PROVIDER`` is unset or set to
            an unknown value, ``NANITICS_LLM_MODEL`` is unset, or the
            provider's API key env var is unset.
    """
    provider = os.environ.get(_PROVIDER_ENV)
    if provider not in _ACCEPTED_PROVIDERS:
        accepted = ", ".join(repr(name) for name in _ACCEPTED_PROVIDERS)
        raise RuntimeError(f"Unknown {_PROVIDER_ENV}={provider!r}; expected one of {accepted}.")

    model = os.environ.get(_MODEL_ENV)
    if not model:
        raise RuntimeError(f"{_MODEL_ENV} is required; the compose does not ship a default model.")

    if provider == "anthropic":
        api_key = os.environ.get(_ANTHROPIC_KEY_ENV)
        if not api_key:
            raise RuntimeError(f"{_PROVIDER_ENV}=anthropic requires {_ANTHROPIC_KEY_ENV}.")
        from nanitics.infrastructure import AnthropicLLMClient

        return AnthropicLLMClient(model=model, api_key=api_key)

    # provider == "openai" — the guard above narrowed the set.
    api_key = os.environ.get(_OPENAI_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"{_PROVIDER_ENV}=openai requires {_OPENAI_KEY_ENV}.")
    from nanitics.infrastructure import OpenAILLMClient

    return OpenAILLMClient(model=model, api_key=api_key)
