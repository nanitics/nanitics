"""Pytest configuration for the validation suite.

- Loads ``.env`` from the repo root so local runs have access to credentials.
- Hard-skips the suite when ``ANTHROPIC_API_KEY`` is unset (the table-stakes
  credential for every real-service run).
- Provides session-scoped fixtures for real clients and the
  ``traced_emitter`` fixture that auto-saves a trace on teardown.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest

from nanitics import InMemoryEmitter

try:
    from dotenv import load_dotenv

    _env_file = Path(__file__).resolve().parents[1] / ".env"
    if _env_file.exists():
        load_dotenv(_env_file)
except ImportError:  # pragma: no cover — dotenv is a dev dep, not a hard requirement.
    pass


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Hard-skip every collected item when ``ANTHROPIC_API_KEY`` is unset."""
    if "ANTHROPIC_API_KEY" in os.environ:
        return
    skip = pytest.mark.skip(reason="Validation suite requires ANTHROPIC_API_KEY")
    for item in items:
        item.add_marker(skip)
    terminal = config.pluginmanager.getplugin("terminalreporter")
    if terminal is not None:
        terminal.write_line(
            "Validation suite: ANTHROPIC_API_KEY not set — all scripts skipped.",
            yellow=True,
        )


_pgvector_container: Any = None


def pytest_configure(config: pytest.Config) -> None:
    """Auto-provision a pgvector container when ``POSTGRES_URL`` is unset.

    Runs before collection so decorators that gate on ``POSTGRES_URL``
    (``requires_postgres``) observe the provisioned URL and do not skip.
    A no-op when ``testcontainers`` is not installed or the Docker
    daemon is unreachable — postgres-dependent tests then skip as usual.
    """
    global _pgvector_container
    if "POSTGRES_URL" in os.environ:
        return
    from validation.helpers.postgres_container import maybe_start_pgvector

    started = maybe_start_pgvector()
    if started is None:
        return
    url, container = started
    os.environ["POSTGRES_URL"] = url
    _pgvector_container = container
    terminal = config.pluginmanager.getplugin("terminalreporter")
    if terminal is not None:
        terminal.write_line(
            f"Validation suite: auto-provisioned pgvector container at {url}",
            cyan=True,
        )


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Stop the auto-provisioned pgvector container, if any."""
    global _pgvector_container
    if _pgvector_container is None:
        return
    from validation.helpers.postgres_container import stop_pgvector

    stop_pgvector(_pgvector_container)
    _pgvector_container = None
    os.environ.pop("POSTGRES_URL", None)


@pytest.fixture(scope="session")
async def anthropic_client() -> AsyncIterator[Any]:
    """Session-scoped real Anthropic client.

    The suite-level gate ensures ``ANTHROPIC_API_KEY`` is present by the time
    this fixture is requested.
    """
    from validation.helpers.llm import make_llm_client

    client = make_llm_client("anthropic")
    try:
        yield client
    finally:
        close = getattr(client, "close", None)
        if close is not None:
            result = close()
            if hasattr(result, "__await__"):
                await result


@pytest.fixture(scope="session")
async def openai_client() -> AsyncIterator[Any]:
    """Session-scoped real OpenAI client. Skips if ``OPENAI_API_KEY`` is missing."""
    if "OPENAI_API_KEY" not in os.environ:
        pytest.skip("Skipping: OPENAI_API_KEY not set")
    from validation.helpers.llm import make_llm_client

    client = make_llm_client("openai")
    try:
        yield client
    finally:
        close = getattr(client, "close", None)
        if close is not None:
            result = close()
            if hasattr(result, "__await__"):
                await result


@pytest.fixture(scope="session")
async def voyage_client() -> AsyncIterator[Any]:
    """Session-scoped real Voyage embedding client. Skips if ``VOYAGE_API_KEY`` is missing."""
    if "VOYAGE_API_KEY" not in os.environ:
        pytest.skip("Skipping: VOYAGE_API_KEY not set")
    from validation.helpers.llm import make_embedding_client

    client = make_embedding_client("voyage")
    try:
        yield client
    finally:
        close = getattr(client, "close", None)
        if close is not None:
            result = close()
            if hasattr(result, "__await__"):
                await result


def _trace_filename(node_name: str) -> str:
    """Derive a stable trace filename from a pytest node name.

    Strips a leading ``test_`` so ``test_smoke_react_agent`` becomes
    ``smoke_react_agent.json`` — matching the flat, overwrite-in-place
    convention ``/analyze-run`` depends on.
    """
    stem = node_name.removeprefix("test_") or node_name
    return f"{stem}.json"


@pytest.fixture
def traced_emitter(request: pytest.FixtureRequest) -> Iterator[InMemoryEmitter]:
    """Yield an ``InMemoryEmitter`` and auto-save its trace on teardown.

    The emitter's ``trace_id`` is the pytest node id so the exported trace
    is self-identifying. The trace is written even when the test fails —
    the most valuable time to inspect it — because the save runs in the
    fixture's finaliser, not from within the test body.

    Filename: ``validation/traces/<test-name-without-test-prefix>.json``.
    Override the base directory with ``VALIDATION_TRACE_DIR``.
    """
    from validation.helpers.trace import save_trace, validation_trace_dir

    emitter = InMemoryEmitter(trace_id=request.node.nodeid)
    try:
        yield emitter
    finally:
        save_trace(
            emitter,
            validation_trace_dir() / _trace_filename(request.node.name),
            script=request.node.nodeid,
        )
