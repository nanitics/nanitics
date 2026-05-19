from __future__ import annotations

from nanitics.capabilities.memory.episodic import (
    Episode,
    EpisodeStore,
    OutcomeType,
    RecallFilters,
)
from nanitics.infrastructure.observability.events import (
    EpisodeForgetEvent,
    EpisodeRecallEvent,
    EpisodeRecordEvent,
)
from nanitics.strategies.tools.context import ToolContext
from nanitics.strategies.tools.function_tool import FunctionTool, tool

_OUTCOME_MAP = {
    "success": OutcomeType.SUCCESS,
    "failure": OutcomeType.FAILURE,
    "partial": OutcomeType.PARTIAL,
}

_VALID_OUTCOMES = ", ".join(f"'{k}'" for k in _OUTCOME_MAP)


def create_episodic_memory_tools(
    store: EpisodeStore,
    namespace: str | None = None,
) -> list[FunctionTool]:
    """Build the agent-facing tool set for episodic memory bound to *store*."""

    @tool(
        name="recall_episodes",
        description=(
            "Search past experiences by situation description. "
            "Returns episodes from previous runs that faced similar situations, "
            "including what was tried and what happened. Use this to learn from "
            "past successes and failures before choosing a strategy. "
            "Optional outcome_filter must be one of: 'success', 'failure', 'partial'."
        ),
    )
    async def recall_episodes(
        query: str,
        context: ToolContext,
        limit: int = 5,
        outcome_filter: str | None = None,
    ) -> str:
        if outcome_filter and outcome_filter not in _OUTCOME_MAP:
            return f"Invalid outcome_filter '{outcome_filter}'. Must be one of: {_VALID_OUTCOMES}."
        filters = RecallFilters(
            outcome=_OUTCOME_MAP[outcome_filter] if outcome_filter else None,
            metadata_filters={"_namespace": namespace} if namespace else None,
        )
        results = await store.recall(query, filters=filters, limit=limit)
        emitter = context.emitter if context is not None else None
        if emitter is not None:
            emitter.emit(
                EpisodeRecallEvent(
                    trace_id=emitter.trace_id,
                    span_id=emitter.span_id,
                    parent_span_id=emitter.parent_span_id,
                    query=query,
                    results_count=len(results),
                    top_score=results[0].similarity_score if results else None,
                    namespace=namespace,
                )
            )
        if not results:
            return "No matching past experiences found."
        lines = []
        for r in results:
            ep = r.episode
            parts = [
                f"[{r.similarity_score:.3f}] (id: {ep.id}) {ep.outcome.value}",
                f"  Situation: {ep.situation}",
                f"  Action: {ep.action}",
            ]
            if ep.outcome_detail:
                parts.append(f"  Outcome: {ep.outcome_detail}")
            if ep.reflection:
                parts.append(f"  Reflection: {ep.reflection}")
            lines.append("\n".join(parts))
        return "\n\n".join(lines)

    @tool(
        name="record_episode",
        description=(
            "Record a new experience for future reference. "
            "Capture what the situation was, what action was taken, and what "
            "the outcome was (must be 'success', 'failure', or 'partial'). "
            "Include reflections on why it worked or didn't. "
            "This helps future runs learn from this experience."
        ),
    )
    async def record_episode(
        situation: str,
        action: str,
        outcome: str,
        context: ToolContext,
        outcome_detail: str | None = None,
        reflection: str | None = None,
    ) -> str:
        if outcome not in _OUTCOME_MAP:
            return f"Invalid outcome '{outcome}'. Must be one of: {_VALID_OUTCOMES}."
        meta = {}
        if namespace:
            meta["_namespace"] = namespace
        episode = Episode(
            situation=situation,
            action=action,
            outcome=_OUTCOME_MAP[outcome],
            outcome_detail=outcome_detail,
            reflection=reflection,
            metadata=meta if meta else None,
        )
        episode_id = await store.record(episode)
        emitter = context.emitter if context is not None else None
        if emitter is not None:
            emitter.emit(
                EpisodeRecordEvent(
                    trace_id=emitter.trace_id,
                    span_id=emitter.span_id,
                    parent_span_id=emitter.parent_span_id,
                    episode_id=episode_id,
                    situation=situation,
                    outcome=outcome,
                    has_reflection=reflection is not None,
                    namespace=namespace,
                )
            )
        return f"Recorded experience (id: {episode_id})."

    @tool(
        name="forget_episode",
        description=("Remove a stored experience by ID. Use the ID returned from recall_episodes or record_episode."),
    )
    async def forget_episode(id: str, context: ToolContext) -> str:
        await store.forget(id)
        emitter = context.emitter if context is not None else None
        if emitter is not None:
            emitter.emit(
                EpisodeForgetEvent(
                    trace_id=emitter.trace_id,
                    span_id=emitter.span_id,
                    parent_span_id=emitter.parent_span_id,
                    episode_id=id,
                    namespace=namespace,
                )
            )
        return f"Forgot experience '{id}'."

    return [recall_episodes, record_episode, forget_episode]
