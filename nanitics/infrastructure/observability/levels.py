"""Event type → level classification for trace events.

Levels are inclusive: "debug" includes "info" events, "verbose" includes all.
"""

from __future__ import annotations

from typing import Literal, TypeAlias

TraceLevel: TypeAlias = Literal["info", "debug", "verbose"]
"""Trace event severity — levels are inclusive (threshold ``debug`` also admits ``info``)."""

LEVEL_ORDER: dict[TraceLevel, int] = {"info": 0, "debug": 1, "verbose": 2}

# Info: user-visible milestones — agent lifecycle, workflow lifecycle,
# multi-agent coordination, HITL, planning milestones, evaluation, durability.
INFO_EVENTS: frozenset[str] = frozenset(
    {
        "agent.start",
        "agent.complete",
        "agent.error",
        "workflow.start",
        "workflow.complete",
        "workflow.error",
        "workflow.step.complete",
        "workflow.structure",
        "multi_agent.delegation",
        "multi_agent.handoff",
        "multi_agent.supervision",
        "hitl.request",
        "hitl.response",
        "blackboard.start",
        "blackboard.round",
        "blackboard.complete",
        "planning.plan.created",
        "planning.goal.status_changed",
        "evaluation.result",
        "execution.suspended",
        "execution.resumed",
        "run.start",
        "run.complete",
        "run.failed",
        "run.suspended",
    }
)

# Debug: operational detail — LLM calls, tool calls, agent steps, context ops,
# error recovery, planning revisions, reflection, revision workflows, code execution.
DEBUG_EVENTS: frozenset[str] = frozenset(
    {
        "llm.request",
        "llm.response",
        "tool.invoke",
        "tool.result",
        "agent.step",
        "context.truncation",
        "context.summarization",
        "context.assembly",
        "error.retry",
        "error.correction",
        "error.degradation",
        "planning.step.updated",
        "planning.plan.revised",
        "revision.start",
        "revision.attempt",
        "revision.complete",
        "reflection.generated",
        "evaluation.revision",
        "code.execution",
        "code.execution.result",
    }
)

# Everything else is "verbose": memory operations, spans, tree search/MCTS,
# coordination primitives, model routing, checkpoint, llm.token, etc.


def classify_level(event_type: str) -> TraceLevel:
    """Classify an SDK event type into a trace level."""
    if event_type in INFO_EVENTS:
        return "info"
    if event_type in DEBUG_EVENTS:
        return "debug"
    return "verbose"


def is_level_included(event_level: TraceLevel, threshold: TraceLevel) -> bool:
    """Return True if *event_level* should be included at *threshold*.

    Levels are inclusive: threshold "debug" includes both "info" and "debug".
    """
    return LEVEL_ORDER[event_level] <= LEVEL_ORDER[threshold]
