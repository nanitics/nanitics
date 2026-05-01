from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Literal, Protocol, runtime_checkable
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from nanitics.capabilities.memory.context_provider import ContextContent
from nanitics.infrastructure.llm.protocol import Message
from nanitics.infrastructure.observability.emitter import EventEmitter
from nanitics.infrastructure.observability.events import SharedMemoryReadEvent


class SharedEntry(BaseModel):
    """An entry on the shared memory board.

    Entries are attributed to an author and support a lifecycle of
    active → superseded or active → retracted.

    Attributes:
        id: Auto-generated UUID.
        content: The contribution text.
        author: Name of the agent that wrote the entry.
        scope: Optional topic scope for organizing entries.
        metadata: Arbitrary metadata dictionary.
        timestamp: When the entry was created.
        status: Lifecycle state: ``"active"``, ``"superseded"``, or ``"retracted"``.
        superseded_by: ID of the entry that replaced this one.
        retracted_reason: Reason this entry was retracted.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str
    author: str
    scope: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    status: Literal["active", "superseded", "retracted"] = "active"
    superseded_by: str | None = None
    retracted_reason: str | None = None


@runtime_checkable
class SharedMemory(Protocol):
    """Protocol for a shared state board used in multi-agent coordination.

    Entries are attributed to authors, support scoping by topic, and
    follow a lifecycle (active → superseded or retracted). Only the
    original author can supersede or retract their own entries.
    """

    async def write(
        self, content: str, *, author: str, scope: str | None = None, metadata: dict[str, Any] | None = None
    ) -> str:
        """Write a new entry to the board.

        Args:
            content: The contribution text.
            author: Name of the agent writing the entry.
            scope: Optional topic scope.
            metadata: Optional metadata.

        Returns:
            The entry ID.
        """
        ...

    async def read(
        self,
        *,
        scope: str | None = None,
        author: str | None = None,
        after: datetime | None = None,
        limit: int | None = None,
        include_inactive: bool = False,
    ) -> list[SharedEntry]:
        """Read entries from the board.

        Args:
            scope: Filter by scope.
            author: Filter by author.
            after: Only entries after this timestamp.
            limit: Maximum number of entries.
            include_inactive: Include superseded and retracted entries.

        Returns:
            Entries sorted newest-first.
        """
        ...

    async def read_by_id(self, entry_id: str) -> SharedEntry | None:
        """Read a single entry by ID.

        Args:
            entry_id: The entry to look up.

        Returns:
            The entry, or None if not found.
        """
        ...

    async def supersede(
        self, entry_id: str, new_content: str, *, author: str, metadata: dict[str, Any] | None = None
    ) -> str:
        """Replace an entry with updated content.

        The original entry is marked as superseded. Only the original
        author can supersede their own entries.

        Args:
            entry_id: The entry to supersede.
            new_content: The replacement content.
            author: Must match the original entry's author.
            metadata: Optional metadata for the new entry.

        Returns:
            The new entry ID.

        Raises:
            ValueError: If the entry is not found or author doesn't match.
        """
        ...

    async def retract(self, entry_id: str, reason: str, *, author: str) -> None:
        """Mark an entry as retracted.

        The entry is preserved with the retraction reason but hidden
        from default reads. Only the original author can retract.

        Args:
            entry_id: The entry to retract.
            reason: Explanation of why the entry is invalid.
            author: Must match the original entry's author.

        Raises:
            ValueError: If the entry is not found or author doesn't match.
        """
        ...

    async def count(self, *, scope: str | None = None, include_inactive: bool = False) -> int:
        """Count entries on the board.

        Args:
            scope: Count only entries in this scope.
            include_inactive: Include superseded and retracted entries.

        Returns:
            Number of matching entries.
        """
        ...

    async def clear(self, *, scope: str | None = None) -> None:
        """Remove all entries, optionally within a scope.

        Args:
            scope: If provided, only clear entries in this scope.
                If None, clear all entries.
        """
        ...


class InMemorySharedMemory:
    """In-memory implementation of the ``SharedMemory`` protocol.

    Stores entries in a list with support for superseding and retracting.
    Useful for testing and development. Data is lost when the process
    ends — for production, implement ``SharedMemory`` with database storage.
    """

    def __init__(self) -> None:
        self._entries: list[SharedEntry] = []

    async def write(
        self,
        content: str,
        *,
        author: str,
        scope: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        entry = SharedEntry(
            content=content,
            author=author,
            scope=scope,
            metadata=metadata or {},
        )
        self._entries.append(entry)
        return entry.id

    async def read(
        self,
        *,
        scope: str | None = None,
        author: str | None = None,
        after: datetime | None = None,
        limit: int | None = None,
        include_inactive: bool = False,
    ) -> list[SharedEntry]:
        results = list(self._entries)
        if not include_inactive:
            results = [e for e in results if e.status == "active"]
        if scope is not None:
            results = [e for e in results if e.scope == scope]
        if author is not None:
            results = [e for e in results if e.author == author]
        if after is not None:
            results = [e for e in results if e.timestamp > after]
        results.reverse()  # newest first
        if limit is not None:
            results = results[:limit]
        return results

    async def read_by_id(self, entry_id: str) -> SharedEntry | None:
        for entry in self._entries:
            if entry.id == entry_id:
                return entry
        return None

    async def supersede(
        self,
        entry_id: str,
        new_content: str,
        *,
        author: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        original = await self.read_by_id(entry_id)
        if original is None:
            raise ValueError(f"Entry '{entry_id}' not found")
        if original.author != author:
            raise ValueError(f"Only the original author '{original.author}' can supersede this entry")

        new_entry = SharedEntry(
            content=new_content,
            author=author,
            scope=original.scope,
            metadata=metadata or {},
        )
        # Mark original as superseded
        idx = next(i for i, e in enumerate(self._entries) if e.id == entry_id)
        self._entries[idx] = original.model_copy(
            update={"status": "superseded", "superseded_by": new_entry.id},
        )
        self._entries.append(new_entry)
        return new_entry.id

    async def retract(self, entry_id: str, reason: str, *, author: str) -> None:
        original = await self.read_by_id(entry_id)
        if original is None:
            raise ValueError(f"Entry '{entry_id}' not found")
        if original.author != author:
            raise ValueError(f"Only the original author '{original.author}' can retract this entry")

        idx = next(i for i, e in enumerate(self._entries) if e.id == entry_id)
        self._entries[idx] = original.model_copy(
            update={"status": "retracted", "retracted_reason": reason},
        )

    async def count(self, *, scope: str | None = None, include_inactive: bool = False) -> int:
        entries = self._entries
        if not include_inactive:
            entries = [e for e in entries if e.status == "active"]
        if scope is not None:
            entries = [e for e in entries if e.scope == scope]
        return len(entries)

    async def clear(self, *, scope: str | None = None) -> None:
        if scope is None:
            self._entries.clear()
        else:
            self._entries = [e for e in self._entries if e.scope != scope]


class SharedMemoryProvider:
    """Context provider that injects shared memory board state into the LLM context.

    Reads active entries (optionally filtered by scope) and formats them
    as a ``[Shared Memory Board]`` context block with attribution.
    Emits a ``SharedMemoryReadEvent`` on each read.

    Args:
        memory: The shared memory store to read from.
        emitter: Optional event emitter for observability.
        scopes: If provided, only show entries from these scopes.
            If None, show all scopes.
        max_entries: Maximum number of entries to include (default: 50).
    """

    def __init__(
        self,
        memory: SharedMemory,
        emitter: EventEmitter | None = None,
        scopes: list[str] | None = None,
        max_entries: int = 50,
        *,
        emitter_provider: Callable[[], EventEmitter | None] | None = None,
    ) -> None:
        self._memory = memory
        self._static_emitter = emitter
        self._emitter_provider: Callable[[], EventEmitter | None] | None = emitter_provider
        self._scopes = scopes
        self._max_entries = max_entries

    @property
    def _emitter(self) -> EventEmitter | None:
        """Emitter used for trace events.

        Resolves through ``emitter_provider`` when set (so the provider
        follows its owning agent's per-task bound emitter); otherwise
        the static emitter passed at construction.
        """
        if self._emitter_provider is not None:
            return self._emitter_provider()
        return self._static_emitter

    async def provide(self, messages: list[Message]) -> ContextContent | None:
        if self._scopes is not None:
            all_entries: list[SharedEntry] = []
            for scope in self._scopes:
                entries = await self._memory.read(scope=scope, limit=self._max_entries)
                all_entries.extend(entries)
            # Re-sort newest first and cap
            all_entries.sort(key=lambda e: e.timestamp, reverse=True)
            all_entries = all_entries[: self._max_entries]
        else:
            all_entries = await self._memory.read(limit=self._max_entries)

        emitter = self._emitter
        if emitter is not None:
            emitter.emit(
                SharedMemoryReadEvent(
                    trace_id=emitter.trace_id,
                    span_id=emitter.span_id,
                    parent_span_id=emitter.parent_span_id,
                    scope=None if self._scopes is None else ",".join(self._scopes),
                    author_filter=None,
                    entries_returned=len(all_entries),
                )
            )

        if not all_entries:
            return None

        lines = ["[Shared Memory Board]", ""]
        # Group by scope
        scoped: dict[str | None, list[SharedEntry]] = {}
        for entry in all_entries:
            scoped.setdefault(entry.scope, []).append(entry)

        for scope_key, entries in scoped.items():
            if scope_key is not None:
                lines.append(f"## {scope_key}")
                lines.append("")
            for entry in entries:
                ts = entry.timestamp.strftime("%Y-%m-%d %H:%M:%S")
                scope_label = f" (scope: {entry.scope})" if entry.scope and scope_key is None else ""
                lines.append(f"[{entry.author}, {ts}]{scope_label}")
                lines.append(entry.content)
                lines.append("")

        formatted = "\n".join(lines).rstrip()
        return ContextContent(content=formatted, priority=5, protected=False, provider_name="shared_memory")


_SHARED_MEMORY_INSTRUCTIONS = (
    "A shared memory board is visible to all participating agents. "
    "Entries are attributed — you can see who wrote what and when. "
    "Use scopes to organize contributions by topic. "
    "Contribute when you have relevant information; don't write redundantly. "
    "Read recent entries to understand what others have contributed. "
    "When your analysis evolves, supersede your previous entry rather than writing a contradictory one. "
    "When you determine a previous contribution is wrong, retract it with a reason. "
    "You can only supersede or retract your own entries, not other agents' entries."
)


class SharedMemoryContributor:
    """System prompt contributor that teaches the agent how to use shared memory.

    Adds instructions explaining attribution, scoping, superseding, and
    retracting entries on the shared memory board.
    """

    def system_prompt_section(self) -> tuple[str, str]:
        return ("shared_memory", _SHARED_MEMORY_INSTRUCTIONS)
