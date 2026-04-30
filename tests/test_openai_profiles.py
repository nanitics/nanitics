from __future__ import annotations

import dataclasses

import pytest

from nanitics.infrastructure.errors import LLMProviderError
from nanitics.infrastructure.llm._openai_profiles import (
    _PROFILES,
    ModelProfile,
    profile_for,
)


class TestProfileFor:
    def test_exact_match_returns_registered_profile(self) -> None:
        assert profile_for("gpt-4o-mini") is _PROFILES["gpt-4o-mini"]

    def test_reasoning_family_exact_match(self) -> None:
        profile = profile_for("o3-mini")
        assert profile.token_param == "max_completion_tokens"
        assert profile.family == "reasoning"

    def test_dated_revision_resolves_to_longest_prefix(self) -> None:
        profile = profile_for("gpt-5-2025-11-07")
        assert profile is _PROFILES["gpt-5"]

    def test_prefix_match_prefers_longer_key(self) -> None:
        # "gpt-5-mini-2026-01-01" must resolve to "gpt-5-mini", not "gpt-5".
        profile = profile_for("gpt-5-mini-2026-01-01")
        assert profile is _PROFILES["gpt-5-mini"]

    def test_o1_prefix_resolves_to_reasoning(self) -> None:
        profile = profile_for("o1-2024-12-17")
        assert profile is _PROFILES["o1"]

    def test_unknown_model_raises_loudly(self) -> None:
        with pytest.raises(LLMProviderError) as exc_info:
            profile_for("claude-3-opus")
        assert exc_info.value.provider == "openai"
        assert "claude-3-opus" in str(exc_info.value)

    def test_empty_model_id_raises(self) -> None:
        with pytest.raises(LLMProviderError):
            profile_for("")


class TestModelProfile:
    def test_is_frozen_dataclass(self) -> None:
        profile = ModelProfile(token_param="max_tokens", family="chat")
        with pytest.raises(dataclasses.FrozenInstanceError):
            profile.token_param = "max_completion_tokens"  # type: ignore[misc]
