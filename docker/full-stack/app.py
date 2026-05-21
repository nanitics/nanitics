"""Full-stack Nanitics FastAPI shell.

Hosts three surfaces in one container:

- the application itself (``/healthz``, ``/readyz``, ``/runners``),
- the embedded Observatory (``/api/observatory/*``), and
- any showcase runners registered in ``runners.py`` at startup.

Each showcase runner appends one :class:`RunnerRegistration` to
:data:`runners.REGISTRATIONS`; the shell starts with an empty
registration list and handles that case gracefully.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

import asyncpg
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from llm_provider import build_llm_client
from runners import REGISTRATIONS, ShellContext

from nanitics.infrastructure import LLMClient
from nanitics.observatory import mount_observatory
from nanitics.tracing import (
    PersistentTraceStore,
    PostgresTraceStore,
    TracedExecutor,
)

_READINESS_PROBE_TIMEOUT_SECONDS = 2.0


def _resolve_postgres_dsn() -> str:
    """Read ``POSTGRES_DSN`` or derive it from the compose's creds."""
    dsn = os.environ.get("POSTGRES_DSN")
    if dsn:
        return dsn
    user = os.environ.get("POSTGRES_USER", "nanitics")
    password = os.environ.get("POSTGRES_PASSWORD", "nanitics-local")
    db = os.environ.get("POSTGRES_DB", "nanitics")
    return f"postgresql://{user}:{password}@postgres:5432/{db}"


async def _default_probe(pool: asyncpg.Pool) -> None:
    """Probe the asyncpg pool with a short-timeout ``SELECT 1``.

    Raised exceptions propagate to the caller — ``/readyz`` surfaces
    them as the 503 ``detail`` payload.
    """
    async with pool.acquire() as conn:
        await conn.execute("SELECT 1", timeout=_READINESS_PROBE_TIMEOUT_SECONDS)


def create_app(
    *,
    build_client: Callable[[], LLMClient] = build_llm_client,
    postgres_dsn: str | None = None,
    trace_store: PersistentTraceStore | None = None,
    pool: asyncpg.Pool | None = None,
    readiness_probe: Callable[[], Awaitable[None]] | None = None,
) -> FastAPI:
    """Construct the full-stack Nanitics app.

    Every dependency is injectable so tests can exercise the shell
    without Docker or a real Postgres.

    Args:
        build_client: Factory returning a fresh :class:`LLMClient`.
        postgres_dsn: Explicit DSN override. When ``None``, the lifespan
            resolves the DSN at startup from env vars.
        trace_store: Pre-constructed trace store. When ``None``, the
            lifespan builds a :class:`PostgresTraceStore` against a
            freshly-opened asyncpg pool.
        pool: Pre-constructed asyncpg pool. When passed alongside a
            ``trace_store``, the shell does not open or close a pool.
        readiness_probe: Override the ``/readyz`` probe callable. When
            ``None``, the probe runs ``SELECT 1`` on the shell's pool.

    Returns:
        A configured :class:`FastAPI` instance.
    """

    state: dict[str, Any] = {
        "trace_store": trace_store,
        "pool": pool,
        "owns_pool": False,
    }

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        store = state["trace_store"]
        active_pool = state["pool"]
        if store is None:
            dsn = postgres_dsn if postgres_dsn is not None else _resolve_postgres_dsn()
            active_pool = await asyncpg.create_pool(dsn, min_size=2, max_size=10)
            state["pool"] = active_pool
            state["owns_pool"] = True
            store = PostgresTraceStore(pool=active_pool)
            await store.ensure_schema()
            state["trace_store"] = store

        executor = TracedExecutor(store)
        context = ShellContext(
            executor=executor,
            trace_store=store,
            pool=active_pool,
            build_client=build_client,
        )
        app.state.shell_context = context

        mount_observatory(app, store, prefix="/api/observatory")

        # Snapshot before registration so we only run newly-added handlers.
        _pre_register_startup = {id(h) for h in app.router.on_startup}
        for registration in REGISTRATIONS:
            registration.register(app, context)
        # Runners add their schema-bootstrap hooks via @app.on_event("startup"),
        # which fires before the lifespan. Because register() is called from
        # inside the lifespan (after Starlette has already dispatched on_startup),
        # those hooks are registered too late to run automatically — invoke them
        # explicitly here, before the first request.
        for _handler in app.router.on_startup:
            if id(_handler) not in _pre_register_startup:
                if asyncio.iscoroutinefunction(_handler):
                    await _handler()
                else:
                    _handler()

        try:
            yield
        finally:
            if state["owns_pool"]:
                owned_pool = state["pool"]
                await owned_pool.close()

    app = FastAPI(title="Nanitics full-stack shell", lifespan=lifespan)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        """Process-liveness probe. Always 200 when the process is up."""
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz() -> JSONResponse:
        """Readiness probe. 503 when the trace-store probe fails."""
        try:
            if readiness_probe is not None:
                await readiness_probe()
            else:
                active_pool = state["pool"]
                if active_pool is None:
                    raise RuntimeError("shell not initialized")
                await _default_probe(active_pool)
        except Exception:
            return JSONResponse(
                status_code=503,
                content={"ready": False, "store": "error", "detail": "trace store probe failed"},
            )
        return JSONResponse(
            status_code=200,
            content={"ready": True, "store": "ok"},
        )

    @app.get("/runners")
    async def runners_index() -> list[dict[str, str]]:
        """Return every registered runner's slug, title, and description."""
        return [
            {
                "slug": registration.slug,
                "title": registration.title,
                "description": registration.description,
            }
            for registration in REGISTRATIONS
        ]

    return app


app = create_app()
