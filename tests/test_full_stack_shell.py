"""Deterministic unit tests for the ``docker/full-stack/`` shell.

No Docker, no real Postgres, no real LLM. Exercises the ``create_app``
factory with injected dependencies and the ``build_llm_client``
provider-factory branches via monkeypatched environment variables.

The ``docker/full-stack/`` directory is not a Python package (the
Dockerfile copies each module to a flat ``/srv/`` working directory and
uvicorn imports them as top-level names). These tests load each module
by file path using :mod:`importlib.util` so coverage and imports stay
honest about how the image actually runs.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from nanitics.infrastructure import (
    AnthropicLLMClient,
    OpenAILLMClient,
)
from nanitics.tracing import InMemoryPersistentTraceStore

# ---------------------------------------------------------------------------
# Module loading — docker/full-stack/ is not a package, so load by path.
# ---------------------------------------------------------------------------

_FULL_STACK_DIR = Path(__file__).resolve().parent.parent / "docker" / "full-stack"


def _load_module(name: str, path: Path) -> ModuleType:
    """Load a single file as a fresh top-level module.

    Registered in ``sys.modules`` under ``name`` so sibling modules
    (``app.py`` imports ``llm_provider`` and ``runners``) resolve
    against the test-loaded copies rather than re-triggering discovery.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def shell_modules(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[ModuleType, ModuleType, ModuleType]]:
    """Load ``llm_provider``, ``runners``, and ``app`` by path.

    ``app.py``'s module-level ``app = create_app()`` builds a factory
    against a default ``POSTGRES_DSN``; the lifespan never runs during
    import, so module load is safe. Each test gets a fresh module tuple
    so ``REGISTRATIONS`` mutations do not leak across tests.
    """
    # Ensure stale entries from a previous test don't satisfy the sibling
    # imports inside ``app.py``.
    for name in ("llm_provider", "runners", "app"):
        sys.modules.pop(name, None)

    # ``runners.py`` imports the ``sql_analyst`` package as a sibling
    # (``from sql_analyst.runner import ...``). In the runtime image the
    # package is a top-level directory on ``sys.path``; mirror that here
    # so the by-path module loader can satisfy that import.
    if str(_FULL_STACK_DIR) not in sys.path:
        sys.path.insert(0, str(_FULL_STACK_DIR))

    # Ensure ``app.py``'s import-time ``app = create_app()`` does not fail
    # if a test process has these env vars unset — we do not exercise the
    # module-level app's lifespan, but ``create_app`` itself reads nothing
    # at construction. Still, set a harmless placeholder DSN so any
    # future accidental resolution is deterministic.
    monkeypatch.setenv("POSTGRES_DSN", "postgresql://test:test@localhost:5432/test")

    llm_provider = _load_module("llm_provider", _FULL_STACK_DIR / "llm_provider.py")
    runners = _load_module("runners", _FULL_STACK_DIR / "runners.py")
    app_module = _load_module("app", _FULL_STACK_DIR / "app.py")

    # Shell-level tests exercise the plumbing, not the runners
    # themselves. Reset the registration list on both the ``runners``
    # module and the ``app`` module's bound copy so no runner-specific
    # dependency (env vars, sandbox DSN derivation, etc.) needs to be
    # satisfied by shell-level tests. Tests that need a registration
    # inject one via :func:`monkeypatch.setattr` below.
    runners.REGISTRATIONS = []
    app_module.REGISTRATIONS = []

    try:
        yield llm_provider, runners, app_module
    finally:
        for name in ("app", "runners", "llm_provider"):
            sys.modules.pop(name, None)


# ---------------------------------------------------------------------------
# Scenario 1–6 — build_llm_client branches
# ---------------------------------------------------------------------------


def _clear_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "NANITICS_LLM_PROVIDER",
        "NANITICS_LLM_MODEL",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)


def test_build_llm_client_anthropic_happy_path(
    shell_modules: tuple[ModuleType, ModuleType, ModuleType],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scenario 1 — anthropic + model + key → AnthropicLLMClient."""
    llm_provider, _, _ = shell_modules
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("NANITICS_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("NANITICS_LLM_MODEL", "claude-haiku-4-5-20251001")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    client = llm_provider.build_llm_client()

    assert isinstance(client, AnthropicLLMClient)


def test_build_llm_client_openai_happy_path(
    shell_modules: tuple[ModuleType, ModuleType, ModuleType],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scenario 2 — openai + model + key → OpenAILLMClient."""
    llm_provider, _, _ = shell_modules
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("NANITICS_LLM_PROVIDER", "openai")
    monkeypatch.setenv("NANITICS_LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")

    client = llm_provider.build_llm_client()

    assert isinstance(client, OpenAILLMClient)


def test_build_llm_client_anthropic_missing_key(
    shell_modules: tuple[ModuleType, ModuleType, ModuleType],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scenario 3 — anthropic selected but no key → RuntimeError."""
    llm_provider, _, _ = shell_modules
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("NANITICS_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("NANITICS_LLM_MODEL", "claude-haiku-4-5-20251001")

    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        llm_provider.build_llm_client()


def test_build_llm_client_openai_missing_key(
    shell_modules: tuple[ModuleType, ModuleType, ModuleType],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scenario 4 — openai selected but no key → RuntimeError."""
    llm_provider, _, _ = shell_modules
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("NANITICS_LLM_PROVIDER", "openai")
    monkeypatch.setenv("NANITICS_LLM_MODEL", "gpt-4o-mini")

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        llm_provider.build_llm_client()


def test_build_llm_client_unknown_provider(
    shell_modules: tuple[ModuleType, ModuleType, ModuleType],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scenario 5 — unknown provider → RuntimeError names accepted values."""
    llm_provider, _, _ = shell_modules
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("NANITICS_LLM_PROVIDER", "gemini")

    with pytest.raises(RuntimeError, match=r"anthropic.*openai|openai.*anthropic"):
        llm_provider.build_llm_client()


def test_build_llm_client_missing_model(
    shell_modules: tuple[ModuleType, ModuleType, ModuleType],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scenario 6 — model unset → RuntimeError names NANITICS_LLM_MODEL."""
    llm_provider, _, _ = shell_modules
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("NANITICS_LLM_PROVIDER", "anthropic")

    with pytest.raises(RuntimeError, match="NANITICS_LLM_MODEL"):
        llm_provider.build_llm_client()


# ---------------------------------------------------------------------------
# Helpers for shell tests — inject an in-memory trace store so the lifespan
# never opens a real asyncpg pool.
# ---------------------------------------------------------------------------


def _stub_build_client() -> Any:
    """Test-scoped ``build_client`` that never runs an LLM call.

    Returned by the ``build_client`` kwarg on ``create_app`` so
    ``ShellContext.build_client`` has a callable but tests never
    invoke it.
    """
    return MagicMock(name="stub-llm-client")


def _make_app(
    app_module: ModuleType,
    *,
    trace_store: Any | None = None,
    readiness_probe: Any | None = None,
    pool: Any | None = None,
) -> FastAPI:
    if trace_store is None:
        trace_store = InMemoryPersistentTraceStore()
    return app_module.create_app(
        build_client=_stub_build_client,
        trace_store=trace_store,
        readiness_probe=readiness_probe,
        pool=pool,
    )


# ---------------------------------------------------------------------------
# Scenario 7 — /healthz always 200.
# ---------------------------------------------------------------------------


def test_healthz_returns_ok(
    shell_modules: tuple[ModuleType, ModuleType, ModuleType],
) -> None:
    _, _, app_module = shell_modules
    app = _make_app(app_module, readiness_probe=AsyncMock(return_value=None))

    with TestClient(app) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# Scenario 8 — /readyz 200 when probe succeeds.
# ---------------------------------------------------------------------------


def test_readyz_reports_ready_when_probe_succeeds(
    shell_modules: tuple[ModuleType, ModuleType, ModuleType],
) -> None:
    _, _, app_module = shell_modules
    app = _make_app(app_module, readiness_probe=AsyncMock(return_value=None))

    with TestClient(app) as client:
        response = client.get("/readyz")

    assert response.status_code == 200
    assert response.json() == {"ready": True, "store": "ok"}


# ---------------------------------------------------------------------------
# Scenario 9 — /readyz 503 when probe raises.
# ---------------------------------------------------------------------------


def test_readyz_reports_not_ready_when_probe_fails(
    shell_modules: tuple[ModuleType, ModuleType, ModuleType],
) -> None:
    _, _, app_module = shell_modules
    probe = AsyncMock(side_effect=RuntimeError("connection refused"))
    app = _make_app(app_module, readiness_probe=probe)

    with TestClient(app) as client:
        response = client.get("/readyz")

    assert response.status_code == 503
    body = response.json()
    assert body["ready"] is False
    assert body["store"] == "error"
    assert body["detail"] == "trace store probe failed"


# ---------------------------------------------------------------------------
# Scenario 10 — /runners empty when REGISTRATIONS is empty.
# ---------------------------------------------------------------------------


def test_runners_index_empty_when_registrations_empty(
    shell_modules: tuple[ModuleType, ModuleType, ModuleType],
) -> None:
    _, runners, app_module = shell_modules
    # The ``shell_modules`` fixture resets REGISTRATIONS to [] so shell
    # tests exercise the plumbing independently of any runner package.
    assert runners.REGISTRATIONS == []

    app = _make_app(app_module, readiness_probe=AsyncMock(return_value=None))
    with TestClient(app) as client:
        response = client.get("/runners")

    assert response.status_code == 200
    assert response.json() == []


# ---------------------------------------------------------------------------
# Scenario 11 — /runners returns injected registration shape.
# ---------------------------------------------------------------------------


def test_runners_index_with_injected_registration(
    shell_modules: tuple[ModuleType, ModuleType, ModuleType],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, runners, app_module = shell_modules

    def fake_register(app: FastAPI, context: Any) -> None:  # pragma: no cover (not invoked by /runners)
        # Body intentionally empty — scenario 12 exercises the
        # registration callable path separately.
        return

    fake = runners.RunnerRegistration(
        slug="test-runner",
        title="Test Runner",
        description="A test-only registration.",
        register=fake_register,
    )
    monkeypatch.setattr(runners, "REGISTRATIONS", [fake])
    # ``app.py`` imports ``REGISTRATIONS`` from ``runners`` by name into its
    # own module namespace, so the patch must also rewrite the ``app``
    # module's bound reference.
    monkeypatch.setattr(app_module, "REGISTRATIONS", [fake])

    app = _make_app(app_module, readiness_probe=AsyncMock(return_value=None))
    with TestClient(app) as client:
        response = client.get("/runners")

    assert response.status_code == 200
    assert response.json() == [
        {
            "slug": "test-runner",
            "title": "Test Runner",
            "description": "A test-only registration.",
        }
    ]


# ---------------------------------------------------------------------------
# Scenario 12 — lifespan iterates registrations exactly once.
# ---------------------------------------------------------------------------


def test_lifespan_invokes_each_registration_exactly_once(
    shell_modules: tuple[ModuleType, ModuleType, ModuleType],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, runners, app_module = shell_modules

    calls: list[tuple[FastAPI, Any]] = []

    def register_one(app: FastAPI, context: Any) -> None:
        calls.append((app, context))

    def register_two(app: FastAPI, context: Any) -> None:
        calls.append((app, context))

    reg_one = runners.RunnerRegistration(slug="one", title="One", description="first", register=register_one)
    reg_two = runners.RunnerRegistration(slug="two", title="Two", description="second", register=register_two)
    monkeypatch.setattr(runners, "REGISTRATIONS", [reg_one, reg_two])
    monkeypatch.setattr(app_module, "REGISTRATIONS", [reg_one, reg_two])

    app = _make_app(app_module, readiness_probe=AsyncMock(return_value=None))
    with TestClient(app) as client:
        client.get("/healthz")  # any request triggers lifespan startup

    assert len(calls) == 2
    # Each registration sees the same FastAPI app and the same context.
    app_seen_one, context_one = calls[0]
    app_seen_two, context_two = calls[1]
    assert app_seen_one is app_seen_two is app
    assert context_one is context_two
    assert context_one.build_client is _stub_build_client


# ---------------------------------------------------------------------------
# Scenario 13 — lifespan teardown closes the asyncpg pool it owns.
# ---------------------------------------------------------------------------


def test_lifespan_closes_owned_pool_on_teardown(
    shell_modules: tuple[ModuleType, ModuleType, ModuleType],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercises the ``owns_pool=True`` branch in the lifespan.

    Monkeypatches ``asyncpg.create_pool`` and ``PostgresTraceStore`` so
    no real network I/O happens, then asserts the pool's ``close()``
    coroutine is awaited during lifespan teardown.
    """
    _, _, app_module = shell_modules

    fake_pool = MagicMock(name="asyncpg-pool")
    fake_pool.close = AsyncMock()

    async def fake_create_pool(*_args: Any, **_kwargs: Any) -> Any:
        return fake_pool

    fake_store = MagicMock(name="postgres-trace-store")
    fake_store.ensure_schema = AsyncMock()
    # The observatory router calls these when listing runs / events —
    # route handlers are not exercised in this test but FastAPI pulls
    # the router in during startup, so the attributes must exist.
    fake_store.list_runs = AsyncMock(return_value=[])
    fake_store.count_runs = AsyncMock(return_value=0)

    monkeypatch.setattr(app_module.asyncpg, "create_pool", fake_create_pool)
    monkeypatch.setattr(app_module, "PostgresTraceStore", lambda pool: fake_store)

    # Call create_app with no injected trace_store — this drives the
    # "owns pool" branch where the lifespan constructs its own pool.
    app = app_module.create_app(
        build_client=_stub_build_client,
        postgres_dsn="postgresql://driver:unused@localhost/unused",
        readiness_probe=AsyncMock(return_value=None),
    )

    with TestClient(app) as client:
        client.get("/healthz")

    fake_store.ensure_schema.assert_awaited_once()
    fake_pool.close.assert_awaited_once()


# ---------------------------------------------------------------------------
# Additional coverage — DSN resolution paths + default probe.
# ---------------------------------------------------------------------------


def test_resolve_postgres_dsn_prefers_env_when_set(
    shell_modules: tuple[ModuleType, ModuleType, ModuleType],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, app_module = shell_modules
    monkeypatch.setenv("POSTGRES_DSN", "postgresql://explicit/override")

    assert app_module._resolve_postgres_dsn() == "postgresql://explicit/override"


def test_resolve_postgres_dsn_derives_from_user_password_db(
    shell_modules: tuple[ModuleType, ModuleType, ModuleType],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, app_module = shell_modules
    monkeypatch.delenv("POSTGRES_DSN", raising=False)
    monkeypatch.setenv("POSTGRES_USER", "svc")
    monkeypatch.setenv("POSTGRES_PASSWORD", "pw")
    monkeypatch.setenv("POSTGRES_DB", "nanitics_test")

    dsn = app_module._resolve_postgres_dsn()

    assert dsn == "postgresql://svc:pw@postgres:5432/nanitics_test"


def test_resolve_postgres_dsn_falls_back_to_compose_defaults(
    shell_modules: tuple[ModuleType, ModuleType, ModuleType],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, app_module = shell_modules
    monkeypatch.delenv("POSTGRES_DSN", raising=False)
    monkeypatch.delenv("POSTGRES_USER", raising=False)
    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
    monkeypatch.delenv("POSTGRES_DB", raising=False)

    dsn = app_module._resolve_postgres_dsn()

    assert dsn == "postgresql://nanitics:nanitics-local@postgres:5432/nanitics"


async def test_default_probe_executes_select_1(
    shell_modules: tuple[ModuleType, ModuleType, ModuleType],
) -> None:
    """The default readiness probe acquires a connection and runs SELECT 1."""
    _, _, app_module = shell_modules

    fake_conn = MagicMock(name="conn")
    fake_conn.execute = AsyncMock()

    class _AcquireCtx:
        async def __aenter__(self) -> Any:
            return fake_conn

        async def __aexit__(self, *_exc: Any) -> None:
            return None

    fake_pool = MagicMock(name="pool")
    fake_pool.acquire = MagicMock(return_value=_AcquireCtx())

    await app_module._default_probe(fake_pool)

    fake_conn.execute.assert_awaited_once()
    args, kwargs = fake_conn.execute.call_args
    assert args[0] == "SELECT 1"
    # Timeout is forwarded to the connection driver.
    assert kwargs.get("timeout") == app_module._READINESS_PROBE_TIMEOUT_SECONDS


def test_readyz_uses_default_probe_when_none_injected(
    shell_modules: tuple[ModuleType, ModuleType, ModuleType],
) -> None:
    """``/readyz`` exercises ``_default_probe`` when no probe is injected.

    This covers the ``readiness_probe is None`` branch together with the
    pool-absent guard that guarantees a loud failure if the shell is
    asked to report readiness before its lifespan has initialized.
    """
    _, _, app_module = shell_modules

    # Inject a trace store but no pool: lifespan skips opening one and
    # ``state["pool"]`` stays ``None``. The default probe branch then
    # raises, which ``/readyz`` maps to a 503.
    app = app_module.create_app(
        build_client=_stub_build_client,
        trace_store=InMemoryPersistentTraceStore(),
    )

    with TestClient(app) as client:
        response = client.get("/readyz")

    assert response.status_code == 503
    body = response.json()
    assert body["ready"] is False
    assert body["store"] == "error"
    assert body["detail"] == "trace store probe failed"


# ---------------------------------------------------------------------------
# Scenario 14 — lifespan drains async startup handlers added by runners.
# ---------------------------------------------------------------------------


def test_lifespan_drains_async_startup_handler_added_by_runner(
    shell_modules: tuple[ModuleType, ModuleType, ModuleType],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A runner that schedules an async ``@app.on_event('startup')`` hook
    from inside ``register()`` must have that hook invoked by the lifespan
    before the first request — even though ``on_startup`` handlers normally
    fire *before* the lifespan body runs.
    """
    _, runners, app_module = shell_modules

    invocations: list[str] = []

    def register_with_async_hook(app: FastAPI, context: Any) -> None:
        @app.on_event("startup")
        async def _async_hook() -> None:
            invocations.append("async")

    reg = runners.RunnerRegistration(
        slug="async-hook-runner",
        title="Async Hook",
        description="Tests async startup hook draining.",
        register=register_with_async_hook,
    )
    monkeypatch.setattr(runners, "REGISTRATIONS", [reg])
    monkeypatch.setattr(app_module, "REGISTRATIONS", [reg])

    app = _make_app(app_module, readiness_probe=AsyncMock(return_value=None))
    with TestClient(app) as client:
        client.get("/healthz")

    assert invocations == ["async"]


def test_lifespan_drains_sync_startup_handler_added_by_runner(
    shell_modules: tuple[ModuleType, ModuleType, ModuleType],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same as the async variant but with a synchronous startup hook."""
    _, runners, app_module = shell_modules

    invocations: list[str] = []

    def register_with_sync_hook(app: FastAPI, context: Any) -> None:
        @app.on_event("startup")
        def _sync_hook() -> None:
            invocations.append("sync")

    reg = runners.RunnerRegistration(
        slug="sync-hook-runner",
        title="Sync Hook",
        description="Tests sync startup hook draining.",
        register=register_with_sync_hook,
    )
    monkeypatch.setattr(runners, "REGISTRATIONS", [reg])
    monkeypatch.setattr(app_module, "REGISTRATIONS", [reg])

    app = _make_app(app_module, readiness_probe=AsyncMock(return_value=None))
    with TestClient(app) as client:
        client.get("/healthz")

    assert invocations == ["sync"]


def test_readyz_runs_default_probe_against_injected_pool(
    shell_modules: tuple[ModuleType, ModuleType, ModuleType],
) -> None:
    """Injecting a pool lets ``_default_probe`` run against it successfully."""
    _, _, app_module = shell_modules

    fake_conn = MagicMock(name="conn")
    fake_conn.execute = AsyncMock()

    class _AcquireCtx:
        async def __aenter__(self) -> Any:
            return fake_conn

        async def __aexit__(self, *_exc: Any) -> None:
            return None

    fake_pool = MagicMock(name="pool")
    fake_pool.acquire = MagicMock(return_value=_AcquireCtx())

    app = app_module.create_app(
        build_client=_stub_build_client,
        trace_store=InMemoryPersistentTraceStore(),
        pool=fake_pool,
    )

    with TestClient(app) as client:
        response = client.get("/readyz")

    assert response.status_code == 200
    assert response.json() == {"ready": True, "store": "ok"}
    fake_conn.execute.assert_awaited_once()
