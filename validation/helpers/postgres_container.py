"""Session-scoped pgvector container for the validation suite.

Starts a ``pgvector/pgvector:pg16`` container on demand, builds an
asyncpg-compatible ``postgresql://`` URL, and tears the container down
on session finish. Imported only by :mod:`validation.conftest` — the
hooks keep this module off the import path for plain ``pytest tests``
runs.

The functions are no-ops when ``testcontainers`` or Docker are not
available; callers gate by checking :func:`maybe_start_pgvector`'s
return value.
"""

from __future__ import annotations

import contextlib
from typing import Any


def _docker_reachable() -> bool:
    try:
        import docker
    except ImportError:
        return False
    try:
        docker.from_env(timeout=2).ping()
    except Exception:
        return False
    return True


def maybe_start_pgvector() -> tuple[str, Any] | None:
    """Start a pgvector container if possible; otherwise return None.

    Returns:
        ``(postgres_url, container)`` on success. The URL is an
        asyncpg-compatible ``postgresql://user:pass@host:port/db`` string.
        Returns ``None`` when ``testcontainers`` is not installed or the
        Docker daemon is unreachable — in that case postgres-dependent
        tests skip via :func:`requires_postgres`.
    """
    try:
        from testcontainers.postgres import PostgresContainer
    except ImportError:
        return None
    if not _docker_reachable():
        return None

    container = PostgresContainer(image="pgvector/pgvector:pg16")
    container.start()
    host = container.get_container_host_ip()
    port = container.get_exposed_port(5432)
    url = f"postgresql://{container.username}:{container.password}@{host}:{port}/{container.dbname}"
    return url, container


def stop_pgvector(container: Any) -> None:
    """Stop a container returned by :func:`maybe_start_pgvector`.

    Swallows exceptions during shutdown — the session is ending either
    way, and raising would mask the real test exit status.
    """
    with contextlib.suppress(Exception):
        container.stop()
