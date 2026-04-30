"""Unit tests for `validation.helpers.llm`.

Exercises env-var resolution, error messages on missing config, model-default
application, and per-provider branching. No real API calls.
"""

from __future__ import annotations

import pytest

from validation.helpers.llm import (
    DEFAULT_EMBEDDING_MODELS,
    DEFAULT_MODELS,
    make_embedding_client,
    make_llm_client,
)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "MISTRAL_API_KEY", "VOYAGE_API_KEY"):
        monkeypatch.delenv(var, raising=False)


def test_make_llm_client_anthropic_default_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    client = make_llm_client("anthropic")
    assert client.model == DEFAULT_MODELS["anthropic"]


def test_make_llm_client_anthropic_override_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    client = make_llm_client("anthropic", model="claude-sonnet-4-6")
    assert client.model == "claude-sonnet-4-6"


def test_make_llm_client_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = make_llm_client("openai")
    assert client.model == DEFAULT_MODELS["openai"]


def test_make_llm_client_mistral(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    client = make_llm_client("mistral")
    assert client.model == DEFAULT_MODELS["mistral"]


def test_make_llm_client_litellm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    client = make_llm_client("litellm")
    assert client.model == DEFAULT_MODELS["litellm"]


def test_make_llm_client_missing_anthropic() -> None:
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY") as excinfo:
        make_llm_client("anthropic")
    msg = str(excinfo.value)
    assert "uv sync --extra anthropic" in msg


def test_make_llm_client_missing_openai() -> None:
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        make_llm_client("openai")


def test_make_llm_client_missing_mistral() -> None:
    with pytest.raises(ValueError, match="MISTRAL_API_KEY"):
        make_llm_client("mistral")


def test_make_llm_client_missing_litellm() -> None:
    # LiteLLM routes to Anthropic by default, so the shared env var is the gate.
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY") as excinfo:
        make_llm_client("litellm")
    assert "uv sync --extra litellm" in str(excinfo.value)


def test_make_embedding_client_voyage_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    client = make_embedding_client("voyage")
    assert client._model == DEFAULT_EMBEDDING_MODELS["voyage"]  # type: ignore[attr-defined]


def test_make_embedding_client_voyage_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    client = make_embedding_client("voyage", model="voyage-3")
    assert client._model == "voyage-3"  # type: ignore[attr-defined]


def test_make_embedding_client_missing_voyage() -> None:
    with pytest.raises(ValueError, match="VOYAGE_API_KEY") as excinfo:
        make_embedding_client("voyage")
    assert "uv sync --extra voyage" in str(excinfo.value)
