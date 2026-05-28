"""Thread identity primitive — message-list continuity across ``Agent.run`` calls.

Carries a per-thread :class:`~nanitics.infrastructure.llm.protocol.Message`
prefix across invocations keyed by an opaque string the consumer chooses.
A subsequent ``Agent.run`` with the same ``thread_key`` sees prior
assistant turns, tool calls, and tool results as its own conversation
history — distinct from information continuity (side-store injection)
provided by the memory primitives.

The replayed messages bypass the ``<nanitics:context provider=...>``
wrapper that ``Agent._inject_context`` applies to ``ContextProvider``
contributions. They are real :class:`Message` objects spliced directly
into the per-run message list so the model treats them as its own prior
turns rather than as injected context. See
``temp/sdk-thread-identity/design-rationale.md`` §4 for the rationale.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from nanitics.infrastructure.errors import ThreadInUseError
from nanitics.infrastructure.llm.protocol import Message


@runtime_checkable
class ThreadStore(Protocol):
    """Persists per-thread message-list prefixes for behavioral continuity.

    A thread is a sequence of :class:`Message` objects keyed by an opaque
    string the consumer chooses. The store carries the messages across
    ``Agent.run`` calls so a subsequent run with the same ``thread_key``
    sees prior assistant turns, tool calls, and tool results as its own
    conversation history.

    Concurrency: implementations need not be thread-safe across
    processes. Serialization of concurrent same-key runs in one process
    is the caller's responsibility, handled by :class:`ThreadLocks` in
    ``Agent.run``. Cross-process serialization is out of scope: the
    Postgres-backed :class:`~nanitics.composition.threads.postgres_thread_store.PostgresThreadStore`
    persists durably across restarts but does not coordinate concurrent
    same-key appends from multiple processes — consumers running
    multiple processes against the same logical thread must coordinate
    externally. A Postgres advisory-lock primitive remains a follow-up.

    The store has no built-in trimming or compaction. Threads grow
    unbounded; consumers concerned with token budget should configure
    :class:`~nanitics.context.ContextManagement` (which trims for
    window-fit correctness at the LLM-call boundary) or wrap the store
    with a compaction-aware decorator.

    Checkpoint interaction: a run that suspends inside ``Agent.run``
    snapshots its thread prefix into ``RunCheckpoint.state`` at suspend
    time. On resume the run uses that frozen prefix and does not
    re-consult the live store; concurrent external appends to the same
    key between suspend and resume are silently overridden by the
    resumed run's continuation when it appends its new messages on
    completion.
    """

    async def load(self, thread_key: str) -> list[Message]:
        """Return the message prefix for the thread.

        Returns an empty list if the key is unknown. Never raises for
        missing keys; raise only on persistence failures (which
        propagate).
        """
        ...

    async def append(self, thread_key: str, messages: list[Message]) -> None:
        """Append messages to the thread, preserving order.

        Messages produced by one ``Agent.run`` are appended together as a
        single batch on successful completion. Failed runs do not call
        ``append`` — the thread is not advanced on failure.
        """
        ...

    async def clear(self, thread_key: str) -> None:
        """Remove all messages for the thread.

        No-op on unknown keys.
        """
        ...


class InMemoryThreadStore:
    """Reference :class:`ThreadStore` backed by an in-process dict.

    Suitable for tests, single-process workloads, and demos. Not durable
    across restarts; not safe across processes. For durable persistence
    use :class:`~nanitics.composition.threads.postgres_thread_store.PostgresThreadStore`.

    The store has no built-in trimming or compaction — see the
    :class:`ThreadStore` docstring for the rationale and pointers to
    :class:`~nanitics.context.ContextManagement`.
    """

    def __init__(self) -> None:
        self._threads: dict[str, list[Message]] = {}

    async def load(self, thread_key: str) -> list[Message]:
        return list(self._threads.get(thread_key, []))

    async def append(self, thread_key: str, messages: list[Message]) -> None:
        self._threads.setdefault(thread_key, []).extend(messages)

    async def clear(self, thread_key: str) -> None:
        self._threads.pop(thread_key, None)


class ThreadLocks:
    """In-process serialization of concurrent runs against the same ``thread_key``.

    Owns an :class:`asyncio.Lock` per key, lazily allocated. The agent
    acquires the lock for the duration of a single ``run`` and releases
    it in a ``finally`` block. Concurrent acquisition of the same key
    raises :class:`~nanitics.infrastructure.errors.ThreadInUseError`
    immediately rather than queueing — see
    ``temp/sdk-thread-identity/design-rationale.md`` §2.

    Cross-process locking is out of scope. Consumers running multiple
    processes against the same logical thread must coordinate
    externally, even when using
    :class:`~nanitics.composition.threads.postgres_thread_store.PostgresThreadStore`
    — its persistence layer is durable but does not serialize concurrent
    appends across processes. A Postgres advisory-lock primitive remains
    a follow-up.
    """

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}

    def _lock_for(self, thread_key: str) -> asyncio.Lock:
        if thread_key not in self._locks:
            self._locks[thread_key] = asyncio.Lock()
        return self._locks[thread_key]

    @contextlib.asynccontextmanager
    async def hold(self, thread_key: str) -> AsyncIterator[None]:
        """Acquire the lock for the duration of a with-block.

        Raises :class:`~nanitics.infrastructure.errors.ThreadInUseError`
        immediately (no wait) if the lock is held. The ``locked()``
        check and the ``async with`` acquire run without yielding to the
        event loop in between, so the apparent race window is empty
        under single-threaded asyncio semantics.
        """
        lock = self._lock_for(thread_key)
        if lock.locked():
            raise ThreadInUseError(thread_key=thread_key)
        async with lock:
            yield


__all__ = [
    "InMemoryThreadStore",
    "ThreadLocks",
    "ThreadStore",
]
