"""Tests for shared memory: data model, store operations, tools, provider, contributor, events."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from nanitics.capabilities.memory.shared import (
    InMemorySharedMemory,
    SharedEntry,
    SharedMemory,
    SharedMemoryContributor,
    SharedMemoryProvider,
)
from nanitics.capabilities.memory.shared_tools import create_shared_memory_tools
from nanitics.core.tools.function_tool import FunctionTool
from nanitics.core.tools.registry import ToolRegistry
from nanitics.infrastructure.llm.protocol import ToolCall
from nanitics.infrastructure.observability.emitter import InMemoryEmitter
from nanitics.infrastructure.observability.events import (
    SharedMemoryReadEvent,
    SharedMemoryRetractEvent,
    SharedMemorySupersededEvent,
    SharedMemoryWriteEvent,
)
from tests.testing_helpers import make_emitter


def make_store() -> InMemorySharedMemory:
    return InMemorySharedMemory()


# ──────────────────────────────────────────────────────────
# SharedEntry Model
# ──────────────────────────────────────────────────────────


class TestSharedEntryModel:
    def test_construction_with_defaults(self) -> None:
        entry = SharedEntry(content="hello", author="agent-a")
        assert entry.content == "hello"
        assert entry.author == "agent-a"
        assert entry.id  # auto-generated
        assert entry.scope is None
        assert entry.metadata == {}
        assert entry.timestamp is not None
        assert entry.status == "active"
        assert entry.superseded_by is None
        assert entry.retracted_reason is None

    def test_construction_with_all_fields(self) -> None:
        ts = datetime.now(UTC)
        entry = SharedEntry(
            id="custom-id",
            content="test content",
            author="agent-b",
            scope="findings",
            metadata={"key": "value"},
            timestamp=ts,
            status="superseded",
            superseded_by="new-id",
            retracted_reason=None,
        )
        assert entry.id == "custom-id"
        assert entry.scope == "findings"
        assert entry.metadata == {"key": "value"}
        assert entry.status == "superseded"
        assert entry.superseded_by == "new-id"

    def test_frozen(self) -> None:
        entry = SharedEntry(content="hello", author="agent-a")
        with pytest.raises(ValidationError):
            entry.content = "changed"

    def test_default_status(self) -> None:
        entry = SharedEntry(content="test", author="a")
        assert entry.status == "active"


# ──────────────────────────────────────────────────────────
# InMemorySharedMemory
# ──────────────────────────────────────────────────────────


class TestInMemorySharedMemory:
    async def test_write_and_read(self) -> None:
        store = make_store()
        entry_id = await store.write("first", author="agent-a")
        assert entry_id
        entries = await store.read()
        assert len(entries) == 1
        assert entries[0].content == "first"
        assert entries[0].author == "agent-a"

    async def test_read_ordering_newest_first(self) -> None:
        store = make_store()
        await store.write("first", author="a")
        await store.write("second", author="b")
        await store.write("third", author="c")
        entries = await store.read()
        assert [e.content for e in entries] == ["third", "second", "first"]

    async def test_read_scope_filter(self) -> None:
        store = make_store()
        await store.write("finding-1", author="a", scope="findings")
        await store.write("decision-1", author="a", scope="decisions")
        await store.write("finding-2", author="b", scope="findings")
        entries = await store.read(scope="findings")
        assert len(entries) == 2
        assert all(e.scope == "findings" for e in entries)

    async def test_read_author_filter(self) -> None:
        store = make_store()
        await store.write("by-a", author="agent-a")
        await store.write("by-b", author="agent-b")
        await store.write("by-a-2", author="agent-a")
        entries = await store.read(author="agent-a")
        assert len(entries) == 2
        assert all(e.author == "agent-a" for e in entries)

    async def test_read_after_filter(self) -> None:
        store = make_store()
        await store.write("old", author="a")
        cutoff = datetime.now(UTC)
        await store.write("new", author="a")
        entries = await store.read(after=cutoff)
        assert len(entries) == 1
        assert entries[0].content == "new"

    async def test_read_limit(self) -> None:
        store = make_store()
        for i in range(10):
            await store.write(f"entry-{i}", author="a")
        entries = await store.read(limit=3)
        assert len(entries) == 3
        # Newest first
        assert entries[0].content == "entry-9"

    async def test_read_include_inactive_default_excludes(self) -> None:
        store = make_store()
        entry_id = await store.write("original", author="a")
        await store.supersede(entry_id, "replacement", author="a")
        entries = await store.read()
        assert len(entries) == 1
        assert entries[0].content == "replacement"

    async def test_read_include_inactive_true(self) -> None:
        store = make_store()
        entry_id = await store.write("original", author="a")
        await store.supersede(entry_id, "replacement", author="a")
        entries = await store.read(include_inactive=True)
        assert len(entries) == 2

    async def test_read_by_id_found(self) -> None:
        store = make_store()
        entry_id = await store.write("test", author="a")
        entry = await store.read_by_id(entry_id)
        assert entry is not None
        assert entry.content == "test"

    async def test_read_by_id_not_found(self) -> None:
        store = make_store()
        entry = await store.read_by_id("nonexistent")
        assert entry is None

    async def test_read_by_id_returns_regardless_of_status(self) -> None:
        store = make_store()
        entry_id = await store.write("original", author="a")
        await store.retract(entry_id, "wrong", author="a")
        entry = await store.read_by_id(entry_id)
        assert entry is not None
        assert entry.status == "retracted"

    async def test_supersede(self) -> None:
        store = make_store()
        original_id = await store.write("original", author="a", scope="findings")
        new_id = await store.supersede(original_id, "updated", author="a")
        original = await store.read_by_id(original_id)
        new_entry = await store.read_by_id(new_id)
        assert original is not None
        assert original.status == "superseded"
        assert original.superseded_by == new_id
        assert new_entry is not None
        assert new_entry.content == "updated"
        assert new_entry.status == "active"
        assert new_entry.scope == "findings"  # inherits scope

    async def test_supersede_rejects_non_author(self) -> None:
        store = make_store()
        entry_id = await store.write("original", author="agent-a")
        with pytest.raises(ValueError, match="Only the original author"):
            await store.supersede(entry_id, "updated", author="agent-b")

    async def test_retract(self) -> None:
        store = make_store()
        entry_id = await store.write("wrong info", author="a")
        await store.retract(entry_id, "this was incorrect", author="a")
        entry = await store.read_by_id(entry_id)
        assert entry is not None
        assert entry.status == "retracted"
        assert entry.retracted_reason == "this was incorrect"

    async def test_retract_rejects_non_author(self) -> None:
        store = make_store()
        entry_id = await store.write("info", author="agent-a")
        with pytest.raises(ValueError, match="Only the original author"):
            await store.retract(entry_id, "wrong", author="agent-b")

    async def test_supersede_nonexistent_entry(self) -> None:
        store = make_store()
        with pytest.raises(ValueError, match="not found"):
            await store.supersede("nonexistent", "new content", author="a")

    async def test_retract_nonexistent_entry(self) -> None:
        store = make_store()
        with pytest.raises(ValueError, match="not found"):
            await store.retract("nonexistent", "reason", author="a")

    async def test_count_active_only(self) -> None:
        store = make_store()
        await store.write("a", author="a")
        entry_id = await store.write("b", author="a")
        await store.retract(entry_id, "wrong", author="a")
        assert await store.count() == 1

    async def test_count_include_inactive(self) -> None:
        store = make_store()
        await store.write("a", author="a")
        entry_id = await store.write("b", author="a")
        await store.retract(entry_id, "wrong", author="a")
        assert await store.count(include_inactive=True) == 2

    async def test_count_scoped(self) -> None:
        store = make_store()
        await store.write("a", author="a", scope="findings")
        await store.write("b", author="a", scope="decisions")
        await store.write("c", author="a", scope="findings")
        assert await store.count(scope="findings") == 2
        assert await store.count(scope="decisions") == 1

    async def test_clear_all(self) -> None:
        store = make_store()
        await store.write("a", author="a")
        await store.write("b", author="b")
        await store.clear()
        assert await store.count() == 0

    async def test_clear_scoped(self) -> None:
        store = make_store()
        await store.write("a", author="a", scope="findings")
        await store.write("b", author="a", scope="decisions")
        await store.clear(scope="findings")
        assert await store.count() == 1
        entries = await store.read()
        assert entries[0].scope == "decisions"

    async def test_protocol_compliance(self) -> None:
        store = make_store()
        assert isinstance(store, SharedMemory)


# ──────────────────────────────────────────────────────────
# Tools
# ──────────────────────────────────────────────────────────


class TestSharedMemoryTools:
    def _get_tool(self, tools: list[FunctionTool], name: str) -> FunctionTool:
        return next(t for t in tools if t.schema.name == name)

    async def test_write_tool_auto_attributes(self) -> None:
        store = make_store()
        tools = create_shared_memory_tools(store, "researcher")
        write_tool = self._get_tool(tools, "write_to_shared")
        result = await write_tool.execute(content="found something", scope="findings")
        assert "Written to shared memory" in result.content
        entries = await store.read()
        assert len(entries) == 1
        assert entries[0].author == "researcher"
        assert entries[0].scope == "findings"

    async def test_read_tool_formats_output(self) -> None:
        store = make_store()
        await store.write("entry content", author="agent-a", scope="findings")
        tools = create_shared_memory_tools(store, "agent-b")
        read_tool = self._get_tool(tools, "read_shared")
        result = await read_tool.execute()
        assert "agent-a" in result.content
        assert "entry content" in result.content

    async def test_read_tool_empty(self) -> None:
        store = make_store()
        tools = create_shared_memory_tools(store, "agent-a")
        read_tool = self._get_tool(tools, "read_shared")
        result = await read_tool.execute()
        assert result.content == "No entries in shared memory."

    async def test_supersede_tool(self) -> None:
        store = make_store()
        entry_id = await store.write("original", author="researcher")
        tools = create_shared_memory_tools(store, "researcher")
        supersede_tool = self._get_tool(tools, "supersede_shared")
        result = await supersede_tool.execute(entry_id=entry_id, new_content="updated")
        assert "Superseded" in result.content
        entries = await store.read()
        assert len(entries) == 1
        assert entries[0].content == "updated"

    async def test_retract_tool(self) -> None:
        store = make_store()
        entry_id = await store.write("wrong", author="researcher")
        tools = create_shared_memory_tools(store, "researcher")
        retract_tool = self._get_tool(tools, "retract_shared")
        result = await retract_tool.execute(entry_id=entry_id, reason="incorrect data")
        assert "Retracted" in result.content
        entries = await store.read()
        assert len(entries) == 0


# ──────────────────────────────────────────────────────────
# Event Emission
# ──────────────────────────────────────────────────────────


class TestSharedMemoryEvents:
    def _get_tool(self, tools: list[FunctionTool], name: str) -> FunctionTool:
        return next(t for t in tools if t.schema.name == name)

    def _make_registry(self, tools: list[FunctionTool], emitter: InMemoryEmitter) -> ToolRegistry:
        registry = ToolRegistry(emitter=emitter)
        for t in tools:
            registry.register(t)
        return registry

    async def test_write_event(self) -> None:
        store = make_store()
        emitter = make_emitter()
        tools = create_shared_memory_tools(store, "agent-a")
        registry = self._make_registry(tools, emitter)
        await registry.dispatch(ToolCall(id="1", name="write_to_shared", arguments={"content": "test content"}))
        write_events = [e for e in emitter.events if isinstance(e, SharedMemoryWriteEvent)]
        assert len(write_events) == 1
        assert write_events[0].author == "agent-a"
        assert write_events[0].content == "test content"
        assert write_events[0].entry_count == 1

    async def test_read_event(self) -> None:
        store = make_store()
        emitter = make_emitter()
        await store.write("entry", author="a")
        tools = create_shared_memory_tools(store, "agent-b")
        registry = self._make_registry(tools, emitter)
        await registry.dispatch(ToolCall(id="1", name="read_shared", arguments={"scope": "findings"}))
        read_events = [e for e in emitter.events if isinstance(e, SharedMemoryReadEvent)]
        assert len(read_events) == 1
        assert read_events[0].scope == "findings"

    async def test_supersede_event(self) -> None:
        store = make_store()
        emitter = make_emitter()
        entry_id = await store.write("original", author="agent-a")
        tools = create_shared_memory_tools(store, "agent-a")
        registry = self._make_registry(tools, emitter)
        await registry.dispatch(
            ToolCall(id="1", name="supersede_shared", arguments={"entry_id": entry_id, "new_content": "updated"})
        )
        events = [e for e in emitter.events if isinstance(e, SharedMemorySupersededEvent)]
        assert len(events) == 1
        assert events[0].original_entry_id == entry_id
        assert events[0].author == "agent-a"
        assert events[0].content == "updated"

    async def test_retract_event(self) -> None:
        store = make_store()
        emitter = make_emitter()
        entry_id = await store.write("wrong", author="agent-a")
        tools = create_shared_memory_tools(store, "agent-a")
        registry = self._make_registry(tools, emitter)
        await registry.dispatch(
            ToolCall(id="1", name="retract_shared", arguments={"entry_id": entry_id, "reason": "incorrect"})
        )
        events = [e for e in emitter.events if isinstance(e, SharedMemoryRetractEvent)]
        assert len(events) == 1
        assert events[0].entry_id == entry_id
        assert events[0].reason == "incorrect"


# ──────────────────────────────────────────────────────────
# SharedMemoryProvider
# ──────────────────────────────────────────────────────────


class TestSharedMemoryProvider:
    async def test_returns_formatted_content(self) -> None:
        store = make_store()
        await store.write("finding 1", author="agent-a", scope="findings")
        await store.write("finding 2", author="agent-b", scope="findings")
        provider = SharedMemoryProvider(store)
        result = await provider.provide([])
        assert result is not None
        assert result.provider_name == "shared_memory"
        assert result.priority == 5
        assert result.protected is False
        assert "agent-a" in result.content
        assert "agent-b" in result.content
        assert "finding 1" in result.content
        assert "finding 2" in result.content

    async def test_respects_scope_filter(self) -> None:
        store = make_store()
        await store.write("finding", author="a", scope="findings")
        await store.write("decision", author="a", scope="decisions")
        provider = SharedMemoryProvider(store, scopes=["findings"])
        result = await provider.provide([])
        assert result is not None
        assert "finding" in result.content
        assert "decision" not in result.content

    async def test_respects_max_entries(self) -> None:
        store = make_store()
        for i in range(10):
            await store.write(f"entry-{i}", author="a")
        provider = SharedMemoryProvider(store, max_entries=3)
        result = await provider.provide([])
        assert result is not None
        # Should have at most 3 entries
        assert result.content.count("[a,") == 3

    async def test_returns_none_when_empty(self) -> None:
        store = make_store()
        provider = SharedMemoryProvider(store)
        result = await provider.provide([])
        assert result is None

    async def test_excludes_inactive_entries(self) -> None:
        store = make_store()
        entry_id = await store.write("wrong", author="a")
        await store.retract(entry_id, "incorrect", author="a")
        provider = SharedMemoryProvider(store)
        result = await provider.provide([])
        assert result is None

    async def test_emits_read_event(self) -> None:
        store = make_store()
        emitter = make_emitter()
        await store.write("test", author="a")
        provider = SharedMemoryProvider(store, emitter=emitter)
        await provider.provide([])
        read_events = [e for e in emitter.events if isinstance(e, SharedMemoryReadEvent)]
        assert len(read_events) == 1
        assert read_events[0].entries_returned == 1


# ──────────────────────────────────────────────────────────
# SharedMemoryContributor
# ──────────────────────────────────────────────────────────


class TestSharedMemoryContributor:
    def test_returns_system_prompt_section(self) -> None:
        contributor = SharedMemoryContributor()
        name, content = contributor.system_prompt_section()
        assert name == "shared_memory"
        assert "shared memory board" in content
        assert "supersede" in content
        assert "retract" in content
        assert "own entries" in content
