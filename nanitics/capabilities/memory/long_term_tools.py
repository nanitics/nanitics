from __future__ import annotations

from nanitics.capabilities.memory.long_term import LongTermStore
from nanitics.infrastructure.observability.events import (
    LongTermDeleteEvent,
    LongTermListEvent,
    LongTermRetrieveEvent,
    LongTermStoreEvent,
)
from nanitics.strategies.tools.context import ToolContext
from nanitics.strategies.tools.function_tool import FunctionTool, tool


def create_long_term_memory_tools(
    store: LongTermStore,
    namespace: str | None = None,
) -> list[FunctionTool]:
    """Build the agent-facing tool set for keyed long-term memory bound to *store*."""

    @tool(
        name="store_memory",
        description=(
            "Store information that persists across conversations. "
            "Use a descriptive key that clearly identifies the content "
            "(e.g., 'user_preferred_output_format', 'project_tech_stack'). "
            "Keys must be self-documenting — when listing stored memories, "
            "the key alone should reveal what's stored. "
            "Storing to an existing key overwrites the previous value."
        ),
    )
    async def store_memory(key: str, value: str, context: ToolContext) -> str:
        await store.store(key, value, namespace=namespace)
        emitter = context.emitter if context is not None else None
        if emitter is not None:
            emitter.emit(
                LongTermStoreEvent(
                    trace_id=emitter.trace_id,
                    span_id=emitter.span_id,
                    parent_span_id=emitter.parent_span_id,
                    key=key,
                    value=value,
                    namespace=namespace,
                )
            )
        return f"Stored value under key '{key}'."

    @tool(
        name="recall_memory",
        description=(
            "Retrieve a previously stored memory by its exact key. "
            "Returns the stored value, or indicates if nothing is stored under that key."
        ),
    )
    async def recall_memory(key: str, context: ToolContext) -> str:
        value = await store.retrieve(key, namespace=namespace)
        found = value is not None
        emitter = context.emitter if context is not None else None
        if emitter is not None:
            emitter.emit(
                LongTermRetrieveEvent(
                    trace_id=emitter.trace_id,
                    span_id=emitter.span_id,
                    parent_span_id=emitter.parent_span_id,
                    key=key,
                    namespace=namespace,
                    found=found,
                    value=value,
                )
            )
        if found:
            assert value is not None  # guaranteed by found check
            return value
        return f"No value found for key '{key}'."

    @tool(
        name="delete_memory",
        description=("Remove a stored memory by key. Use when information is no longer relevant or accurate."),
    )
    async def delete_memory(key: str, context: ToolContext) -> str:
        await store.delete(key, namespace=namespace)
        emitter = context.emitter if context is not None else None
        if emitter is not None:
            emitter.emit(
                LongTermDeleteEvent(
                    trace_id=emitter.trace_id,
                    span_id=emitter.span_id,
                    parent_span_id=emitter.parent_span_id,
                    key=key,
                    namespace=namespace,
                )
            )
        return f"Deleted key '{key}'."

    @tool(
        name="list_memory_keys",
        description=(
            "List all keys in long-term memory. "
            "Use this to discover what information was stored in previous conversations. "
            "Keys are descriptive, so the list alone should tell you which memories are worth recalling."
        ),
    )
    async def list_memory_keys(context: ToolContext) -> str:
        keys = await store.list_keys(namespace=namespace)
        emitter = context.emitter if context is not None else None
        if emitter is not None:
            emitter.emit(
                LongTermListEvent(
                    trace_id=emitter.trace_id,
                    span_id=emitter.span_id,
                    parent_span_id=emitter.parent_span_id,
                    namespace=namespace,
                    keys=keys,
                )
            )
        if not keys:
            return "No keys stored."
        return ", ".join(keys)

    return [store_memory, recall_memory, delete_memory, list_memory_keys]
