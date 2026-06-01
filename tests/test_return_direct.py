"""Tests for tool-terminated runs (``return_direct``) in ``ReActAgent``.

A tool marked ``return_direct=True`` ends the run on its result, skipping the
closing LLM turn (and, with ``output_schema``, the structured-synthesis call).
These tests cover the flag plumbing across the three construction surfaces, the
loop termination semantics (single, multi-call, terminate-on-first), the
``output_schema`` / ``output_evaluator`` short-circuits, the suspend/resume
path, and the ``return_direct=False`` regression.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from nanitics.collaboration.approval_wrapped import ApprovalWrappedTool
from nanitics.collaboration.durable_provider import DurableHumanInputProvider
from nanitics.collaboration.hitl_store import InMemoryHitlRequestStore
from nanitics.collaboration.protocol import HumanDecision, HumanInputResponse
from nanitics.composition.durability.suspension import SuspendExecution
from nanitics.composition.multi_agent.agent_tool import AgentTool
from nanitics.infrastructure import MockLLMClient
from nanitics.infrastructure.llm.protocol import ToolSchema
from nanitics.strategies import ReActAgent, tool
from nanitics.strategies.agents.base import AgentResult
from nanitics.strategies.agents.evaluation import (
    EvaluationContext,
    EvaluationResult,
    EvaluationVerdict,
)
from nanitics.strategies.tools import FunctionTool, ToolResult
from nanitics.tracing import ToolCall
from tests.testing_helpers import make_emitter, make_response

_RUN_ID = "return-direct-test"


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #


@tool(name="propose", description="Emit a proposal and end the run", return_direct=True)
async def propose_tool() -> ToolResult:
    return ToolResult(content="PROPOSAL", metadata={"proposals": [1, 2, 3]})


@tool(name="lookup", description="A normal, non-terminal tool")
async def lookup_tool() -> str:
    return "looked up"


async def _noop() -> str:
    return "noop"


def _agent(client: MockLLMClient, *, tools: list, **kwargs) -> ReActAgent:
    return ReActAgent(
        name="react",
        llm_client=client,
        emitter=make_emitter(),
        system_prompt="be terse",
        tools=tools,
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# Step 1-3: flag plumbing across the three construction surfaces
# --------------------------------------------------------------------------- #


class TestFlagPlumbing:
    def test_tool_schema_default_false(self) -> None:
        schema = ToolSchema(name="t", description="d", parameters={})
        assert schema.return_direct is False

    def test_tool_schema_explicit_true(self) -> None:
        schema = ToolSchema(name="t", description="d", parameters={}, return_direct=True)
        assert schema.return_direct is True

    def test_function_tool_threads_flag(self) -> None:
        ft = FunctionTool(
            _noop,
            name="t",
            description="d",
            parameters_schema={"type": "object", "properties": {}},
            return_direct=True,
        )
        assert ft.schema.return_direct is True

    def test_function_tool_defaults_false(self) -> None:
        ft = FunctionTool(
            _noop,
            name="t",
            description="d",
            parameters_schema={"type": "object", "properties": {}},
        )
        assert ft.schema.return_direct is False

    def test_decorator_threads_flag(self) -> None:
        assert propose_tool.schema.return_direct is True
        assert lookup_tool.schema.return_direct is False

    def test_agent_tool_threads_flag(self) -> None:
        delegate = _agent(MockLLMClient([make_response(content="x")]), tools=[])
        emitter = make_emitter()
        at_direct = AgentTool(agent=delegate, emitter=emitter, description="d", return_direct=True)
        at_plain = AgentTool(agent=delegate, emitter=emitter, description="d")
        assert at_direct.schema.return_direct is True
        assert at_plain.schema.return_direct is False

    def test_flag_never_serialized_to_provider(self) -> None:
        """Regression: the flag is SDK-side only — adapters emit only
        name/description/parameters, so it never leaks to the wire."""
        from nanitics.infrastructure.llm.anthropic import _to_anthropic_tools

        wire = _to_anthropic_tools([propose_tool.schema])
        assert wire is not None
        assert "return_direct" not in wire[0]
        assert set(wire[0]) == {"name", "description", "input_schema"}


# --------------------------------------------------------------------------- #
# Step 4 / 7: loop termination semantics
# --------------------------------------------------------------------------- #


class TestReturnDirectTermination:
    async def test_single_return_direct_ends_run(self) -> None:
        tc = ToolCall(id="tc1", name="propose", arguments={})
        # Only ONE scripted response: if the loop asked for a closing turn the
        # mock would raise "no more scripted responses".
        client = MockLLMClient([make_response(content="proposing", tool_calls=[tc])])
        agent = _agent(client, tools=[propose_tool])

        result = await agent.run("go")

        assert result.termination_reason == "return_direct"
        assert result.output == "PROPOSAL"
        assert result.parsed is None
        assert len(client.calls) == 1  # no closing generation

    async def test_terminal_metadata_round_trips_to_message(self) -> None:
        tc = ToolCall(id="tc1", name="propose", arguments={})
        client = MockLLMClient([make_response(content="proposing", tool_calls=[tc])])
        agent = _agent(client, tools=[propose_tool])

        result = await agent.run("go")

        tool_results = [m for m in result.messages if m.role == "tool_result"]
        assert tool_results[-1].metadata == {"proposals": [1, 2, 3]}

    async def test_output_schema_synthesis_skipped(self) -> None:
        class Out(BaseModel):
            value: int

        tc = ToolCall(id="tc1", name="propose", arguments={})
        client = MockLLMClient([make_response(content="proposing", tool_calls=[tc])])
        agent = _agent(client, tools=[propose_tool], output_schema=Out)

        result = await agent.run("go")

        assert result.termination_reason == "return_direct"
        assert result.output == "PROPOSAL"
        assert result.parsed is None
        # The structured-synthesis call (the only place output_schema is passed)
        # was never made.
        assert len(client.calls) == 1
        assert all(c["output_schema"] is None for c in client.calls)

    async def test_multi_tool_batch_runs_all_and_terminates_on_return_direct(self) -> None:
        # Batch: a non-terminal tool first, the return_direct tool second.
        calls = [
            ToolCall(id="tc1", name="lookup", arguments={}),
            ToolCall(id="tc2", name="propose", arguments={}),
        ]
        client = MockLLMClient([make_response(content="working", tool_calls=calls)])
        agent = _agent(client, tools=[lookup_tool, propose_tool])

        result = await agent.run("go")

        assert result.termination_reason == "return_direct"
        assert result.output == "PROPOSAL"
        # Both tools ran: both tool_result messages are present.
        tool_results = [m for m in result.messages if m.role == "tool_result"]
        assert [m.content for m in tool_results] == ["looked up", "PROPOSAL"]

    async def test_terminate_on_first_return_direct_lowest_index_wins(self) -> None:
        @tool(name="propose_b", description="second proposal", return_direct=True)
        async def propose_b() -> ToolResult:
            return ToolResult(content="SECOND")

        calls = [
            ToolCall(id="tc1", name="propose", arguments={}),
            ToolCall(id="tc2", name="propose_b", arguments={}),
        ]
        client = MockLLMClient([make_response(content="working", tool_calls=calls)])
        agent = _agent(client, tools=[propose_tool, propose_b])

        result = await agent.run("go")

        assert result.output == "PROPOSAL"  # lowest index wins
        assert result.termination_reason == "return_direct"

    async def test_evaluator_not_invoked_on_return_direct(self) -> None:
        class SpyEvaluator:
            def __init__(self) -> None:
                self.calls = 0

            @property
            def max_revisions(self) -> int:
                return 2

            async def evaluate(self, output: str, context: EvaluationContext) -> EvaluationResult:
                self.calls += 1
                return EvaluationResult(verdict=EvaluationVerdict.ACCEPT, evaluator_name="spy")

        spy = SpyEvaluator()
        tc = ToolCall(id="tc1", name="propose", arguments={})
        client = MockLLMClient([make_response(content="proposing", tool_calls=[tc])])
        agent = _agent(client, tools=[propose_tool], output_evaluator=spy)

        result = await agent.run("go")

        assert result.termination_reason == "return_direct"
        assert spy.calls == 0  # evaluator bypassed entirely


class TestRegression:
    async def test_non_return_direct_tool_loops_to_completion(self) -> None:
        tc = ToolCall(id="tc1", name="lookup", arguments={})
        client = MockLLMClient(
            [
                make_response(content="looking", tool_calls=[tc]),
                make_response(content="all done"),
            ]
        )
        agent = _agent(client, tools=[lookup_tool])

        result = await agent.run("go")

        assert result.termination_reason == "complete"
        assert result.output == "all done"
        assert len(client.calls) == 2  # tool turn + closing turn


# --------------------------------------------------------------------------- #
# Step 7: suspend/resume through a return_direct tool
# --------------------------------------------------------------------------- #


async def _suspend_then_resume(agent: ReActAgent, hitl_store: InMemoryHitlRequestStore) -> AgentResult:
    """Drive a run to suspension, approve the pending request, and resume."""
    with pytest.raises(SuspendExecution) as ei:
        await agent.run("go")
    checkpoint = ei.value.checkpoint_data
    assert checkpoint is not None

    pending = await hitl_store.get_pending_requests(_RUN_ID)
    assert pending
    await hitl_store.save_response(
        pending[0].request_id,
        HumanInputResponse(request_id=pending[0].request_id, decision=HumanDecision.APPROVE),
    )

    agent._set_resume_state(checkpoint)
    return await agent.run("go")


class TestResume:
    async def test_resume_reaches_return_direct_in_tail(self) -> None:
        """Suspending tool first, return_direct tool second: resume runs the
        approved tool then the return_direct tool and terminates."""
        hitl_store = InMemoryHitlRequestStore()
        provider = DurableHumanInputProvider(request_store=hitl_store)
        gated = ApprovalWrappedTool(tool=lookup_tool, provider=provider)
        calls = [
            ToolCall(id="tc1", name="lookup", arguments={}),
            ToolCall(id="tc2", name="propose", arguments={}),
        ]
        client = MockLLMClient([make_response(content="working", tool_calls=calls)])
        agent = _agent(client, tools=[gated, propose_tool], run_id=_RUN_ID)

        result = await _suspend_then_resume(agent, hitl_store)

        assert result.termination_reason == "return_direct"
        assert result.output == "PROPOSAL"

    async def test_pre_suspend_return_direct_survives_checkpoint(self) -> None:
        """return_direct tool first, suspending tool second: the hit recorded
        before suspension survives the checkpoint and wins on resume."""
        hitl_store = InMemoryHitlRequestStore()
        provider = DurableHumanInputProvider(request_store=hitl_store)
        gated = ApprovalWrappedTool(tool=lookup_tool, provider=provider)
        calls = [
            ToolCall(id="tc1", name="propose", arguments={}),
            ToolCall(id="tc2", name="lookup", arguments={}),
        ]
        client = MockLLMClient([make_response(content="working", tool_calls=calls)])
        agent = _agent(client, tools=[propose_tool, gated], run_id=_RUN_ID)

        result = await _suspend_then_resume(agent, hitl_store)

        assert result.termination_reason == "return_direct"
        assert result.output == "PROPOSAL"  # the pre-suspension hit, lowest index
