"""Regression tests for concurrent-bind safety.

Verifies that sharing one agent across concurrent orchestrator tasks
(MapReduce fan-out, Parallel branches) does not corrupt the agent's
per-call emitter state or bleed spans across tasks.

On main (pre-refactor) ``test_mapreduce_shared_agent_no_lookup_error``
raises ``LookupError`` from ``InMemoryEmitter.span_id`` because
``Agent.bind()`` mutates ``self._emitter`` across tasks.
"""

from __future__ import annotations

import pytest

from nanitics import InMemoryEmitter, ReActAgent, ToolCall, tool
from nanitics.composition.orchestration.adapters import AgentStep
from nanitics.composition.orchestration.mapreduce import MapReduce
from nanitics.composition.orchestration.parallel import Parallel
from nanitics.infrastructure.observability.events import (
    AgentStartEvent,
    LLMRequestEvent,
    ToolInvokeEvent,
)
from tests.testing_helpers import make_response, make_usage


@tool(name="echo", description="Echo the input back.")
async def _echo_tool(text: str) -> str:
    return f"echo:{text}"


def _make_shared_agent(*, name: str = "shared-react", with_tool: bool = False) -> ReActAgent:
    """ReAct agent wired to a MockLLMClient that always returns a final text."""
    from nanitics.infrastructure.llm.mock import MockLLMClient
    from nanitics.infrastructure.llm.protocol import LLMResponse

    if with_tool:
        # One tool_use response followed by a final text response; repeat for every item.
        responses: list = []
        for i in range(32):
            responses.append(
                LLMResponse(
                    content=None,
                    tool_calls=[ToolCall(id=f"tc-{i}", name="echo", arguments={"text": f"item-{i}"})],
                    usage=make_usage(),
                    model="test-model",
                    stop_reason="tool_use",
                )
            )
            responses.append(make_response(content=f"answer-{i}"))
        client = MockLLMClient(responses)
        tools = [_echo_tool]
    else:
        responses = [make_response(content=f"answer-{i}") for i in range(32)]
        client = MockLLMClient(responses)
        tools = []

    emitter = InMemoryEmitter(trace_id="shared-agent-trace")
    return ReActAgent(
        name=name,
        llm_client=client,
        emitter=emitter,
        system_prompt="you are a test agent",
        tools=tools,
    )


class TestConcurrentBindSafety:
    async def test_mapreduce_shared_agent_no_lookup_error(self) -> None:
        """Three MapReduce items sharing one tool-using ReAct agent must all
        complete without raising LookupError on the tool registry's emitter.
        """
        agent = _make_shared_agent(with_tool=True)
        workflow_emitter = InMemoryEmitter(trace_id="mapreduce-trace")
        mr = MapReduce(
            name="mr",
            step=AgentStep(agent),
            emitter=workflow_emitter,
            splitter=lambda x: x,
            reducer=lambda results: [r.output for r in results],
            max_concurrency=None,
        )
        result = await mr.execute(["task-a", "task-b", "task-c"])
        assert result.metadata["total_items"] == 3
        assert len(result.output) == 3
        # Each item ran the echo tool exactly once.
        tool_events = [e for e in workflow_emitter.events if isinstance(e, ToolInvokeEvent)]
        assert len(tool_events) == 3

    async def test_mapreduce_shared_agent_span_lineage(self) -> None:
        """Each item's trace lineage is independent — no cross-task span bleed."""
        agent = _make_shared_agent()
        workflow_emitter = InMemoryEmitter(trace_id="mapreduce-lineage-trace")
        mr = MapReduce(
            name="mr",
            step=AgentStep(agent),
            emitter=workflow_emitter,
            splitter=lambda x: x,
            reducer=lambda results: [r.output for r in results],
            max_concurrency=None,
        )
        await mr.execute(["a", "b", "c"])

        agent_starts = [e for e in workflow_emitter.events if isinstance(e, AgentStartEvent)]
        assert len(agent_starts) == 3
        # No cross-item span_id bleed: each AgentStartEvent's span_id is unique.
        agent_span_ids = [e.span_id for e in agent_starts]
        assert len(set(agent_span_ids)) == 3
        # And each AgentStartEvent's parent_span_id is distinct (each
        # item's bound emitter has its own synthetic root span).
        parent_ids = [e.parent_span_id for e in agent_starts]
        assert len(set(parent_ids)) == 3

    async def test_parallel_shared_agent_no_cross_task_bleed(self) -> None:
        """Two Parallel branches sharing one ReAct agent emit disjoint LLM events."""
        agent = _make_shared_agent()
        workflow_emitter = InMemoryEmitter(trace_id="parallel-trace")
        par = Parallel(
            name="par",
            steps=[AgentStep(agent), AgentStep(agent)],
            emitter=workflow_emitter,
        )
        await par.execute("shared-input")

        llm_events = [e for e in workflow_emitter.events if isinstance(e, LLMRequestEvent)]
        # Two branches, each issues one LLM request.
        assert len(llm_events) == 2
        parents = {e.parent_span_id for e in llm_events}
        # Each branch's LLM event should have a distinct parent span
        # (its own step span), not the same one.
        assert len(parents) == 2


class TestBoundAgentAccessors:
    """Coverage for the ``BoundAgent`` public accessors."""

    async def test_bound_agent_exposes_unmutated_agent(self) -> None:
        agent = _make_shared_agent()
        parent = InMemoryEmitter(trace_id="t")
        bound = agent.bind(parent)
        assert bound.agent is agent
        assert bound.emitter is not parent
        assert bound.emitter.trace_id == parent.trace_id


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])
