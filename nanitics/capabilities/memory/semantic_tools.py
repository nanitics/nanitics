from __future__ import annotations

from nanitics.capabilities.memory.semantic import SemanticStore
from nanitics.infrastructure.observability.events import (
    SemanticDeleteEvent,
    SemanticSearchEvent,
    SemanticStoreEvent,
)
from nanitics.strategies.tools.context import ToolContext
from nanitics.strategies.tools.function_tool import FunctionTool, tool


def create_semantic_memory_tools(
    store: SemanticStore,
    namespace: str | None = None,
) -> list[FunctionTool]:
    """Build the agent-facing tool set for semantic memory bound to *store*."""

    @tool(
        name="store_knowledge",
        description=(
            "Store information for later retrieval by similarity search. "
            "Use this to save facts, findings, or any knowledge that may be "
            "useful to recall later. Content is searchable by meaning, not "
            "exact keywords."
        ),
    )
    async def store_knowledge(content: str, context: ToolContext, metadata: str | None = None) -> str:
        meta = {"description": metadata} if metadata else None
        if namespace:
            meta = {**(meta or {}), "_namespace": namespace}
        entry_id = await store.add(content, metadata=meta)
        emitter = context.emitter if context is not None else None
        if emitter is not None:
            emitter.emit(
                SemanticStoreEvent(
                    trace_id=emitter.trace_id,
                    span_id=emitter.span_id,
                    parent_span_id=emitter.parent_span_id,
                    content=content,
                    entry_id=entry_id,
                    namespace=namespace,
                )
            )
        return f"Stored knowledge (id: {entry_id})."

    @tool(
        name="search_knowledge",
        description=(
            "Search stored knowledge by semantic similarity. "
            "Provide a natural language query describing what you're looking for. "
            "Returns the most relevant matches ranked by similarity."
        ),
    )
    async def search_knowledge(query: str, context: ToolContext, limit: int = 5) -> str:
        if namespace:
            # Fetch all results so namespace filtering doesn't truncate below limit
            all_results = await store.search(query, limit=10000)
            results = [r for r in all_results if r.metadata and r.metadata.get("_namespace") == namespace][:limit]
        else:
            results = await store.search(query, limit=limit)
        emitter = context.emitter if context is not None else None
        if emitter is not None:
            emitter.emit(
                SemanticSearchEvent(
                    trace_id=emitter.trace_id,
                    span_id=emitter.span_id,
                    parent_span_id=emitter.parent_span_id,
                    query=query,
                    results_count=len(results),
                    top_score=results[0].score if results else None,
                    namespace=namespace,
                )
            )
        if not results:
            return "No matching knowledge found."
        lines = []
        for r in results:
            meta_str = ""
            if r.metadata:
                display_meta = {k: v for k, v in r.metadata.items() if k != "_namespace"}
                if display_meta:
                    meta_str = f" | metadata: {display_meta}"
            lines.append(f"[{r.score:.3f}] (id: {r.id}) {r.content}{meta_str}")
        return "\n".join(lines)

    @tool(
        name="delete_knowledge",
        description=(
            "Remove a stored knowledge entry by ID. Use the ID returned from store_knowledge or search_knowledge."
        ),
    )
    async def delete_knowledge(id: str, context: ToolContext) -> str:
        await store.delete(id)
        emitter = context.emitter if context is not None else None
        if emitter is not None:
            emitter.emit(
                SemanticDeleteEvent(
                    trace_id=emitter.trace_id,
                    span_id=emitter.span_id,
                    parent_span_id=emitter.parent_span_id,
                    entry_id=id,
                    namespace=namespace,
                )
            )
        return f"Deleted knowledge entry '{id}'."

    return [store_knowledge, search_knowledge, delete_knowledge]
