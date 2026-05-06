"""Pytest skip decorators for per-resource gating.

Each decorator resolves its skip condition lazily at test-run time so that
collection is cheap and import order does not matter.
"""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Callable
from typing import Any, TypeVar, cast

import pytest

F = TypeVar("F", bound=Callable[..., Any])


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def requires_postgres(fn: F) -> F:
    """Mark the decorated test as requiring Postgres + ``asyncpg``.

    Applies the ``postgres`` marker. The validation conftest scans for
    this marker during collection and (a) lazily provisions a pgvector
    container on demand when ``POSTGRES_URL`` is unset and Docker is
    reachable, and (b) skips the test with reason
    ``Skipping: POSTGRES_URL not set or asyncpg not installed`` when
    provisioning fails or ``asyncpg`` is unavailable. Provisioning is
    not attempted when no collected item carries the marker, so runs
    that don't need Postgres pay no Docker cost.
    """
    return cast("F", pytest.mark.postgres(fn))


def _docker_unreachable() -> bool:
    if not _has_module("docker"):
        return True
    try:
        import docker

        client = docker.from_env(timeout=2)  # type: ignore[attr-defined]
        client.ping()
    except Exception:
        return True
    return False


def requires_docker(fn: F) -> F:
    """Skip if the Docker daemon is unreachable.

    Uses :func:`docker.from_env` with a short timeout; swallows connection
    errors and translates them into a ``pytest.skip`` with reason
    ``Skipping: Docker daemon unreachable``.
    """
    reason = "Skipping: Docker daemon unreachable"
    return cast("F", pytest.mark.skipif(_docker_unreachable(), reason=reason)(fn))


def requires_voyage(fn: F) -> F:
    """Skip if ``VOYAGE_API_KEY`` is missing.

    Skip message: ``Skipping: VOYAGE_API_KEY not set``.
    """
    reason = "Skipping: VOYAGE_API_KEY not set"
    missing = "VOYAGE_API_KEY" not in os.environ
    return cast("F", pytest.mark.skipif(missing, reason=reason)(fn))


def requires_tavily(fn: F) -> F:
    """Skip if ``TAVILY_API_KEY`` is missing.

    Skip message: ``Skipping: TAVILY_API_KEY not set``.
    """
    reason = "Skipping: TAVILY_API_KEY not set"
    missing = "TAVILY_API_KEY" not in os.environ
    return cast("F", pytest.mark.skipif(missing, reason=reason)(fn))


def requires_mcp(fn: F) -> F:
    """Skip if the ``mcp`` extra is not installed.

    Skip message: ``Skipping: mcp extra not installed``.
    """
    reason = "Skipping: mcp extra not installed"
    missing = not _has_module("mcp")
    return cast("F", pytest.mark.skipif(missing, reason=reason)(fn))


def requires_litellm(fn: F) -> F:
    """Skip if the ``litellm`` extra is not installed.

    Skip message: ``Skipping: litellm extra not installed``.
    """
    reason = "Skipping: litellm extra not installed"
    missing = not _has_module("litellm")
    return cast("F", pytest.mark.skipif(missing, reason=reason)(fn))


def requires_mistral(fn: F) -> F:
    """Skip if ``MISTRAL_API_KEY`` is missing.

    Skip message: ``Skipping: MISTRAL_API_KEY not set``.
    """
    reason = "Skipping: MISTRAL_API_KEY not set"
    missing = "MISTRAL_API_KEY" not in os.environ
    return cast("F", pytest.mark.skipif(missing, reason=reason)(fn))


def requires_openai(fn: F) -> F:
    """Skip if ``OPENAI_API_KEY`` is missing.

    Skip message: ``Skipping: OPENAI_API_KEY not set``.
    """
    reason = "Skipping: OPENAI_API_KEY not set"
    missing = "OPENAI_API_KEY" not in os.environ
    return cast("F", pytest.mark.skipif(missing, reason=reason)(fn))
