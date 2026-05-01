"""OpenAI model-capability profiles.

This module encodes per-model request-shape variance *as data*, not as
conditional code in the OpenAI client. The chat-completions API diverged
between classic chat models (``gpt-4o*``, ``gpt-4-turbo``) — which accept
``max_tokens`` — and the reasoning families (``o1*``, ``o3*``, ``o4*``,
``gpt-5*``) — which require ``max_completion_tokens`` and reject the
older key with HTTP 400 ``unsupported_parameter``.

Adding a new model family is a one-line entry in ``_PROFILES``. Resolving
an unknown model id raises ``LLMProviderError`` loudly — we never silently
default to a shape that may be wrong.

This module is internal (underscore prefix) and not re-exported from any
``__init__.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from nanitics.infrastructure.errors import LLMProviderError

TokenParam = Literal["max_tokens", "max_completion_tokens"]
Family = Literal["chat", "reasoning"]


@dataclass(frozen=True)
class ModelProfile:
    """Per-model OpenAI request-shape capabilities.

    Attributes:
        token_param: The kwarg name used to cap completion length for
            this model. Chat-family models use ``"max_tokens"``; reasoning
            models use ``"max_completion_tokens"``.
        family: High-level classification. Currently informational; reserved
            for future capability axes that depend on family membership.
    """

    token_param: TokenParam
    family: Family


_CHAT = ModelProfile(token_param="max_tokens", family="chat")
_REASONING = ModelProfile(token_param="max_completion_tokens", family="reasoning")

# Concrete model ids → profile. New families are one-line entries.
_PROFILES: dict[str, ModelProfile] = {
    # Chat family
    "gpt-4o-mini": _CHAT,
    "gpt-4o": _CHAT,
    "gpt-4-turbo": _CHAT,
    # Reasoning family
    "gpt-5": _REASONING,
    "gpt-5-mini": _REASONING,
    "o1": _REASONING,
    "o1-mini": _REASONING,
    "o1-preview": _REASONING,
    "o3": _REASONING,
    "o3-mini": _REASONING,
    "o4-mini": _REASONING,
}


def profile_for(model: str) -> ModelProfile:
    """Resolve the ``ModelProfile`` for an OpenAI model id.

    Matching strategy: exact match first; otherwise, the longest registry
    key that is a prefix of ``model`` (so dated revisions like
    ``"gpt-5-2025-11-07"`` resolve to ``"gpt-5"``'s profile).

    Raises ``LLMProviderError`` if no match is found. No silent default —
    the OpenAI request surface has mutated twice in 18 months; a silent
    default masks the next mutation.
    """
    if model in _PROFILES:
        return _PROFILES[model]

    best_match: str | None = None
    for key in _PROFILES:
        if model.startswith(key) and (best_match is None or len(key) > len(best_match)):
            best_match = key

    if best_match is not None:
        return _PROFILES[best_match]

    raise LLMProviderError(
        f"Unknown OpenAI model id {model!r}: no profile registered. "
        f"Add an entry to nanitics/infrastructure/llm/_openai_profiles.py.",
        provider="openai",
    )
