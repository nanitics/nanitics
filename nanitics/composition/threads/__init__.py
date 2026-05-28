"""Thread identity — message-list continuity across ``Agent.run`` calls.

See :mod:`nanitics.composition.threads.store` for the protocol, the
reference implementation, and the in-process lock structure.
"""

from nanitics.composition.threads.store import (
    InMemoryThreadStore,
    ThreadLocks,
    ThreadStore,
)

try:
    from nanitics.composition.threads.postgres_thread_store import (
        PostgresThreadStore,
        get_thread_schema_sql,
    )
except ImportError:
    PostgresThreadStore = None  # type: ignore[assignment,misc]
    get_thread_schema_sql = None  # type: ignore[assignment]

__all__ = [
    "InMemoryThreadStore",
    "PostgresThreadStore",
    "ThreadLocks",
    "ThreadStore",
    "get_thread_schema_sql",
]
