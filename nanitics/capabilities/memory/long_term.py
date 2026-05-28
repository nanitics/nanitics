from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LongTermStore(Protocol):
    """Protocol for a key-value store that persists across agent runs.

    Stores string values under descriptive keys. Supports optional
    namespaces for isolating data between different agents or contexts.

    **For:** named facts the agent explicitly stores and recalls — user
    preferences, learned constants, project metadata, anything addressable
    by a descriptive key.

    **Not for:** similarity-based retrieval over a corpus (use
    ``SemanticStore``), in-run scratchpad work (use ``WorkingMemory``),
    learning from past task outcomes (use ``EpisodeStore``), or
    multi-agent coordination (use ``SharedMemory``).
    """

    async def store(self, key: str, value: str, namespace: str | None = None) -> None:
        """Store a value under a key.

        Overwrites any existing value for the same key and namespace.

        Args:
            key: Descriptive key identifying the stored value.
            value: The string value to store.
            namespace: Optional namespace for isolation.
        """
        ...

    async def retrieve(self, key: str, namespace: str | None = None) -> str | None:
        """Retrieve a value by exact key.

        Args:
            key: The key to look up.
            namespace: Optional namespace to search within.

        Returns:
            The stored value, or None if the key does not exist.
        """
        ...

    async def delete(self, key: str, namespace: str | None = None) -> None:
        """Remove a stored key-value pair.

        Args:
            key: The key to delete.
            namespace: Optional namespace.
        """
        ...

    async def list_keys(self, namespace: str | None = None) -> list[str]:
        """List all keys in the store.

        Args:
            namespace: Optional namespace to list keys from.

        Returns:
            List of stored keys.
        """
        ...


class InMemoryLongTermStore:
    """In-memory implementation of the ``LongTermStore`` protocol.

    Stores data in a nested dict keyed by namespace and key. Useful for
    testing and development. Data is lost when the process ends — for
    production use, implement ``LongTermStore`` with database storage.
    """

    def __init__(self) -> None:
        self._data: dict[str | None, dict[str, str]] = {}

    async def store(self, key: str, value: str, namespace: str | None = None) -> None:
        if namespace not in self._data:
            self._data[namespace] = {}
        self._data[namespace][key] = value

    async def retrieve(self, key: str, namespace: str | None = None) -> str | None:
        return self._data.get(namespace, {}).get(key)

    async def delete(self, key: str, namespace: str | None = None) -> None:
        if namespace in self._data:
            self._data[namespace].pop(key, None)

    async def list_keys(self, namespace: str | None = None) -> list[str]:
        return list(self._data.get(namespace, {}).keys())
