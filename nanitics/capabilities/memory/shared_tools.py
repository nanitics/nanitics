from __future__ import annotations

from nanitics.capabilities.memory.shared import SharedMemory
from nanitics.core.tools.context import ToolContext
from nanitics.core.tools.function_tool import FunctionTool, tool
from nanitics.infrastructure.observability.emitter import EventEmitter
from nanitics.infrastructure.observability.events import (
    SharedMemoryReadEvent,
    SharedMemoryRetractEvent,
    SharedMemorySupersededEvent,
    SharedMemoryWriteEvent,
)


def create_shared_memory_tools(
    store: SharedMemory,
    agent_name: str,
    emitter: EventEmitter | None = None,
) -> list[FunctionTool]:
    """Build the agent-facing tool set for shared-board memory bound to *store*, attributed to *agent_name*."""

    @tool(
        name="write_to_shared",
        description=(
            "Write a contribution to the shared memory board visible to all agents. "
            "Use scope to organize by topic (e.g., 'findings', 'decisions'). "
            "Your name is automatically attributed."
        ),
    )
    async def write_to_shared(
        content: str,
        context: ToolContext,
        scope: str | None = None,
    ) -> str:
        entry_id = await store.write(content, author=agent_name, scope=scope)
        active_count = await store.count()
        em = context.emitter if context is not None else emitter
        if em is not None:
            em.emit(
                SharedMemoryWriteEvent(
                    trace_id=em.trace_id,
                    span_id=em.span_id,
                    parent_span_id=em.parent_span_id,
                    entry_id=entry_id,
                    author=agent_name,
                    content=content,
                    scope=scope,
                    entry_count=active_count,
                )
            )
        return f"Written to shared memory (id: {entry_id})."

    @tool(
        name="read_shared",
        description=(
            "Read active entries from the shared memory board. "
            "Optionally filter by scope. Returns entries with attribution."
        ),
    )
    async def read_shared(
        context: ToolContext,
        scope: str | None = None,
        limit: int = 20,
    ) -> str:
        entries = await store.read(scope=scope, limit=limit)
        em = context.emitter if context is not None else emitter
        if em is not None:
            em.emit(
                SharedMemoryReadEvent(
                    trace_id=em.trace_id,
                    span_id=em.span_id,
                    parent_span_id=em.parent_span_id,
                    scope=scope,
                    author_filter=None,
                    entries_returned=len(entries),
                )
            )
        if not entries:
            return "No entries in shared memory."
        lines: list[str] = []
        for entry in entries:
            ts = entry.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            scope_label = f" (scope: {entry.scope})" if entry.scope else ""
            lines.append(f"[{entry.author}, {ts}]{scope_label}")
            lines.append(entry.content)
            lines.append("")
        return "\n".join(lines).rstrip()

    @tool(
        name="supersede_shared",
        description=(
            "Replace one of your own previous entries with updated content. "
            "The original is preserved in the log but marked as superseded. "
            "You can only supersede your own entries."
        ),
    )
    async def supersede_shared(
        entry_id: str,
        new_content: str,
        context: ToolContext,
    ) -> str:
        new_id = await store.supersede(entry_id, new_content, author=agent_name)
        original = await store.read_by_id(entry_id)
        scope = original.scope if original else None
        em = context.emitter if context is not None else emitter
        if em is not None:
            em.emit(
                SharedMemorySupersededEvent(
                    trace_id=em.trace_id,
                    span_id=em.span_id,
                    parent_span_id=em.parent_span_id,
                    original_entry_id=entry_id,
                    new_entry_id=new_id,
                    author=agent_name,
                    content=new_content,
                    scope=scope,
                )
            )
        return f"Superseded entry {entry_id} with new entry {new_id}."

    @tool(
        name="retract_shared",
        description=(
            "Mark one of your own previous entries as invalid. "
            "The entry remains in the log but is hidden from default reads. "
            "You can only retract your own entries."
        ),
    )
    async def retract_shared(
        entry_id: str,
        reason: str,
        context: ToolContext,
    ) -> str:
        original = await store.read_by_id(entry_id)
        scope = original.scope if original else None
        await store.retract(entry_id, reason, author=agent_name)
        em = context.emitter if context is not None else emitter
        if em is not None:
            em.emit(
                SharedMemoryRetractEvent(
                    trace_id=em.trace_id,
                    span_id=em.span_id,
                    parent_span_id=em.parent_span_id,
                    entry_id=entry_id,
                    author=agent_name,
                    reason=reason,
                    scope=scope,
                )
            )
        return f"Retracted entry {entry_id}."

    return [write_to_shared, read_shared, supersede_shared, retract_shared]
