"""Tests for trace level classification."""

import pytest

from nanitics.infrastructure.observability.levels import (
    DEBUG_EVENTS,
    INFO_EVENTS,
    LEVEL_ORDER,
    classify_level,
    is_level_included,
)

# All event types defined in the SDK TraceEvent union (extracted from events.py).
ALL_EVENT_TYPES = frozenset(
    {
        "agent.complete",
        "agent.error",
        "agent.start",
        "agent.step",
        "blackboard.complete",
        "blackboard.round",
        "blackboard.start",
        "checkpoint.saved",
        "code.execution",
        "code.execution.result",
        "context.assembly",
        "context.summarization",
        "context.truncation",
        "error.correction",
        "error.degradation",
        "error.retry",
        "evaluation.result",
        "evaluation.revision",
        "execution.resumed",
        "execution.suspended",
        "hitl.request",
        "hitl.response",
        "llm.request",
        "llm.response",
        "llm.token",
        "mcts.backpropagation",
        "mcts.iteration",
        "memory.episode.forget",
        "memory.episode.recall",
        "memory.episode.record",
        "memory.longterm.delete",
        "memory.longterm.list",
        "memory.longterm.retrieve",
        "memory.longterm.store",
        "memory.semantic.delete",
        "memory.semantic.search",
        "memory.semantic.store",
        "memory.shared.read",
        "memory.shared.retract",
        "memory.shared.supersede",
        "memory.shared.write",
        "memory.working.read",
        "memory.working.update",
        "model.routing",
        "multi_agent.bidding.allocated",
        "multi_agent.bidding.bid",
        "multi_agent.bidding.complete",
        "multi_agent.bidding.start",
        "multi_agent.broadcast.complete",
        "multi_agent.broadcast.response",
        "multi_agent.broadcast.start",
        "multi_agent.bus.complete",
        "multi_agent.bus.delivered",
        "multi_agent.bus.published",
        "multi_agent.bus.start",
        "multi_agent.consensus.agreement",
        "multi_agent.consensus.complete",
        "multi_agent.consensus.start",
        "multi_agent.consensus.vote",
        "multi_agent.debate.argument",
        "multi_agent.debate.complete",
        "multi_agent.debate.resolution",
        "multi_agent.debate.start",
        "multi_agent.delegation",
        "multi_agent.handoff",
        "multi_agent.peer.complete",
        "multi_agent.peer.consultation",
        "multi_agent.peer.start",
        "multi_agent.supervision",
        "planning.goal.status_changed",
        "planning.plan.created",
        "planning.plan.revised",
        "planning.step.updated",
        "reflection.generated",
        "revision.attempt",
        "revision.complete",
        "revision.start",
        "run.complete",
        "run.failed",
        "run.start",
        "run.suspended",
        "span.end",
        "span.start",
        "tool.invoke",
        "tool.result",
        "tree_search.complete",
        "tree_search.node.created",
        "tree_search.node.evaluated",
        "tree_search.node.pruned",
        "workflow.complete",
        "workflow.error",
        "workflow.start",
        "workflow.step.complete",
        "workflow.structure",
    }
)


class TestClassifyLevel:
    """Verify every SDK event type maps to the expected level."""

    @pytest.mark.parametrize(
        "event_type",
        sorted(INFO_EVENTS),
        ids=lambda e: e,
    )
    def test_info_events(self, event_type: str) -> None:
        assert classify_level(event_type) == "info"

    @pytest.mark.parametrize(
        "event_type",
        sorted(DEBUG_EVENTS),
        ids=lambda e: e,
    )
    def test_debug_events(self, event_type: str) -> None:
        assert classify_level(event_type) == "debug"

    # All event types not in INFO_EVENTS or DEBUG_EVENTS should be verbose.
    VERBOSE_EVENTS = ALL_EVENT_TYPES - INFO_EVENTS - DEBUG_EVENTS

    @pytest.mark.parametrize(
        "event_type",
        sorted(VERBOSE_EVENTS),
        ids=lambda e: e,
    )
    def test_verbose_events(self, event_type: str) -> None:
        assert classify_level(event_type) == "verbose"

    def test_unknown_event_type_is_verbose(self) -> None:
        assert classify_level("some.unknown.event") == "verbose"

    def test_info_and_debug_are_disjoint(self) -> None:
        assert frozenset() == INFO_EVENTS & DEBUG_EVENTS

    def test_all_classified_events_exist_in_sdk(self) -> None:
        """Verify INFO_EVENTS and DEBUG_EVENTS only contain known SDK event types."""
        classified = INFO_EVENTS | DEBUG_EVENTS
        unknown = classified - ALL_EVENT_TYPES
        assert unknown == frozenset(), f"Classified events not in SDK: {unknown}"


class TestIsLevelIncluded:
    """Verify inclusive level filtering."""

    def test_info_included_at_info(self) -> None:
        assert is_level_included("info", "info") is True

    def test_debug_excluded_at_info(self) -> None:
        assert is_level_included("debug", "info") is False

    def test_verbose_excluded_at_info(self) -> None:
        assert is_level_included("verbose", "info") is False

    def test_info_included_at_debug(self) -> None:
        assert is_level_included("info", "debug") is True

    def test_debug_included_at_debug(self) -> None:
        assert is_level_included("debug", "debug") is True

    def test_verbose_excluded_at_debug(self) -> None:
        assert is_level_included("verbose", "debug") is False

    def test_all_included_at_verbose(self) -> None:
        assert is_level_included("info", "verbose") is True
        assert is_level_included("debug", "verbose") is True
        assert is_level_included("verbose", "verbose") is True


class TestLevelOrder:
    """Verify level ordering constants."""

    def test_info_lowest(self) -> None:
        assert LEVEL_ORDER["info"] < LEVEL_ORDER["debug"] < LEVEL_ORDER["verbose"]

    def test_exactly_three_levels(self) -> None:
        assert len(LEVEL_ORDER) == 3
