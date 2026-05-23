"""Per-agent integration tests: an in-flight tool call is actually cancelled.

The test scaffolding uses a tool that awaits an unresolved ``asyncio.Event``
and a cancellation triggered via ``loop.call_soon`` so the assertion is
deterministic — no real sleeps and no timing-dependent races.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from nanitics.infrastructure.llm.mock import MockLLMClient
from nanitics.infrastructure.llm.protocol import LLMResponse, ToolCall
from nanitics.infrastructure.observability.emitter import InMemoryEmitter
from nanitics.infrastructure.observability.events import (
    SafetyCancellationEvent,
    Usage,
)
from nanitics.safety.cancellation import CancellationToken
from nanitics.strategies.agents.evaluation import (
    EvaluationResult,
    EvaluationVerdict,
)
from nanitics.strategies.tools import tool


class _AcceptEvaluator:
    """Minimal evaluator for LATS — always accepts."""

    @property
    def max_revisions(self) -> int:
        return 0

    async def evaluate(self, output: str, context: object) -> EvaluationResult:
        return EvaluationResult(
            verdict=EvaluationVerdict.ACCEPT,
            score=0.9,
            evaluator_name="test-evaluator",
        )


def _usage() -> Usage:
    return Usage(input_tokens=1, output_tokens=1)


def _emitter() -> InMemoryEmitter:
    return InMemoryEmitter(trace_id="trace")


def _make_tool_response(tool_name: str, args: dict) -> LLMResponse:
    return LLMResponse(
        content=None,
        tool_calls=[ToolCall(id="tc1", name=tool_name, arguments=args)],
        usage=_usage(),
        model="m",
        stop_reason="tool_use",
    )


def _make_text_response(content: str) -> LLMResponse:
    return LLMResponse(
        content=content,
        tool_calls=[],
        usage=_usage(),
        model="m",
        stop_reason="end_turn",
    )


def _make_slow_tool() -> tuple:
    started = asyncio.Event()

    @tool(name="slow", description="Never resolves until cancelled.")
    async def slow_tool(query: str) -> str:
        started.set()
        await asyncio.Event().wait()  # blocks forever
        return "unreachable"  # pragma: no cover

    return slow_tool, started


def _count_cancellation_events(emitter: InMemoryEmitter) -> int:
    return sum(1 for e in emitter.events if isinstance(e, SafetyCancellationEvent))


class TestReActInFlightCancellation:
    async def test_in_flight_tool_call_is_cancelled(self) -> None:
        from nanitics.strategies.agents.react import ReActAgent

        slow_tool, started = _make_slow_tool()
        token = CancellationToken()
        emitter = _emitter()
        client = MockLLMClient([_make_tool_response("slow", {"query": "x"})])

        agent = ReActAgent(
            name="r",
            llm_client=client,
            emitter=emitter,
            system_prompt="be brief",
            tools=[slow_tool],
            cancellation_token=token,
        )

        async def _run() -> None:
            nonlocal result
            result = await agent.run("go")

        result = None  # type: ignore[assignment]
        task = asyncio.create_task(_run())
        await started.wait()
        token.cancel()
        await asyncio.wait_for(task, timeout=2.0)

        assert result is not None
        assert result.termination_reason == "cancelled"
        # Exactly one safety event from the dispatch-cancellation site.
        assert _count_cancellation_events(emitter) == 1


class TestLATSInFlightCancellation:
    async def test_in_flight_tool_call_is_cancelled(self) -> None:
        from nanitics.strategies.agents.lats import LATSAgent

        slow_tool, started = _make_slow_tool()
        token = CancellationToken()
        emitter = _emitter()

        # LATS expands once per iteration; the first expand call invokes the
        # slow tool, which never resolves.
        client = MockLLMClient([_make_tool_response("slow", {"query": "x"}) for _ in range(10)])

        agent = LATSAgent(
            name="l",
            llm_client=client,
            emitter=emitter,
            system_prompt="be brief",
            tools=[slow_tool],
            cancellation_token=token,
            max_iterations=5,
            max_depth=3,
            node_evaluator=_AcceptEvaluator(),
        )

        result = None  # type: ignore[assignment]

        async def _run() -> None:
            nonlocal result
            result = await agent.run("go")

        task = asyncio.create_task(_run())
        await started.wait()
        token.cancel()
        await asyncio.wait_for(task, timeout=2.0)

        assert result is not None
        assert result.termination_reason == "cancelled"
        assert _count_cancellation_events(emitter) >= 1


class TestReWOOInFlightCancellation:
    async def test_in_flight_tool_call_is_cancelled(self) -> None:
        from nanitics.capabilities.planning import InMemoryPlanStore
        from nanitics.strategies.agents.rewoo import ReWOOAgent

        slow_tool, started = _make_slow_tool()
        token = CancellationToken()
        emitter = _emitter()

        plan_response = json.dumps(
            {
                "steps": [
                    {
                        "step_number": 1,
                        "description": "Slow",
                        "tool_name": "slow",
                        "arguments": {"query": "x"},
                        "depends_on": [],
                    }
                ]
            }
        )
        client = MockLLMClient([_make_text_response(plan_response)])

        agent = ReWOOAgent(
            name="rw",
            llm_client=client,
            emitter=emitter,
            system_prompt="be brief",
            tools=[slow_tool],
            plan_store=InMemoryPlanStore(),
            cancellation_token=token,
        )

        result = None  # type: ignore[assignment]

        async def _run() -> None:
            nonlocal result
            result = await agent.run("go")

        task = asyncio.create_task(_run())
        await started.wait()
        token.cancel()
        await asyncio.wait_for(task, timeout=2.0)

        assert result is not None
        assert result.termination_reason == "cancelled"
        assert _count_cancellation_events(emitter) == 1


class TestCodeActInFlightCancellation:
    async def test_in_flight_sandbox_execute_is_cancelled(self) -> None:
        from nanitics.safety.sandbox.protocol import ExecutionResult
        from nanitics.strategies.agents.codeact import CodeActAgent

        started = asyncio.Event()

        class _SlowSandbox:
            async def start(self) -> None:
                return None

            async def execute(self, code: str) -> ExecutionResult:
                started.set()
                await asyncio.Event().wait()  # blocks forever
                return ExecutionResult(success=True)  # pragma: no cover

            async def reset(self) -> None:
                return None

            async def cleanup(self) -> None:
                return None

            async def __aenter__(self) -> _SlowSandbox:
                return self

            async def __aexit__(self, *args: object) -> None:
                return None

        token = CancellationToken()
        emitter = _emitter()
        client = MockLLMClient(
            [
                _make_tool_response("execute_code", {"code": "import time; time.sleep(99)"}),
            ]
        )

        agent = CodeActAgent(
            name="ca",
            llm_client=client,
            emitter=emitter,
            system_prompt="be brief",
            sandbox=_SlowSandbox(),  # type: ignore[arg-type]
            cancellation_token=token,
        )

        result = None  # type: ignore[assignment]

        async def _run() -> None:
            nonlocal result
            result = await agent.run("go")

        task = asyncio.create_task(_run())
        await started.wait()
        token.cancel()
        await asyncio.wait_for(task, timeout=2.0)

        assert result is not None
        assert result.termination_reason == "cancelled"
        assert _count_cancellation_events(emitter) == 1


# Reflexion and TreeOfThought delegate tool dispatch to an inner agent.
# Their in-flight cancellation comes for free via that inner agent's
# wrapped dispatch path — exercised by ``TestReActInFlightCancellation``.


class TestAgentToolInFlightCancellation:
    """An ``AgentTool`` wrapping a ReAct agent must surface tool-cancellation."""

    async def test_in_flight_tool_call_inside_agent_tool_is_cancelled(self) -> None:
        from nanitics.composition.multi_agent.agent_tool import AgentTool
        from nanitics.strategies.agents.react import ReActAgent

        slow_tool, started = _make_slow_tool()
        token = CancellationToken()
        emitter = _emitter()
        inner_client = MockLLMClient([_make_tool_response("slow", {"query": "x"})])

        inner_agent = ReActAgent(
            name="inner",
            llm_client=inner_client,
            emitter=emitter,
            system_prompt="x",
            tools=[slow_tool],
            cancellation_token=token,
        )

        agent_tool = AgentTool(
            agent=inner_agent,
            emitter=emitter,
            description="An agent",
            cancellation_token=token,
        )

        task = asyncio.create_task(agent_tool.execute(task="go"))
        await started.wait()
        token.cancel()
        result = await asyncio.wait_for(task, timeout=2.0)
        assert "cancel" in result.content.lower() or _count_cancellation_events(emitter) >= 1


@pytest.mark.parametrize(
    "agent_factory_name",
    ["react"],
)
async def test_no_duplicate_safety_emission(agent_factory_name: str) -> None:
    """Cancellation that interrupts a tool emits exactly one safety event."""
    from nanitics.strategies.agents.react import ReActAgent

    slow_tool, started = _make_slow_tool()
    token = CancellationToken()
    emitter = _emitter()
    client = MockLLMClient([_make_tool_response("slow", {"query": "x"})])

    agent = ReActAgent(
        name="r",
        llm_client=client,
        emitter=emitter,
        system_prompt="be brief",
        tools=[slow_tool],
        cancellation_token=token,
    )

    async def _run() -> None:
        await agent.run("go")

    task = asyncio.create_task(_run())
    await started.wait()
    token.cancel()
    await asyncio.wait_for(task, timeout=2.0)
    assert _count_cancellation_events(emitter) == 1
