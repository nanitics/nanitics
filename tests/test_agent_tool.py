"""Tests for AgentTool: schema generation, execution, response shaping, metadata, errors."""

from unittest.mock import AsyncMock, Mock

import pytest

from nanitics.composition.multi_agent.agent_tool import AgentTool
from nanitics.composition.multi_agent.context_transfer import (
    CustomTransfer,
    RawOutputTransfer,
    SummaryTransfer,
    TrajectoryTransfer,
)
from nanitics.infrastructure.llm.mock import MockLLMClient
from nanitics.infrastructure.llm.protocol import (
    ImageContentBlock,
    LLMResponse,
    Message,
    TextContentBlock,
)
from nanitics.infrastructure.observability.emitter import InMemoryEmitter
from nanitics.infrastructure.observability.events import DelegationEvent, Usage
from nanitics.strategies.agents.base import AgentResult
from nanitics.strategies.tools.protocol import Tool


def _make_emitter() -> InMemoryEmitter:
    return InMemoryEmitter(trace_id="test-trace")


def _make_usage() -> Usage:
    return Usage(input_tokens=10, output_tokens=5)


def _make_result(
    output: str | None = "final answer",
    messages: list[Message] | None = None,
    total_steps: int = 2,
    termination_reason: str = "complete",
) -> AgentResult:
    return AgentResult(
        output=output,
        total_steps=total_steps,
        termination_reason=termination_reason,
        messages=messages or [],
        usage=_make_usage(),
    )


def _make_agent(
    name: str = "delegate",
    result: AgentResult | None = None,
) -> AsyncMock:
    agent = AsyncMock()
    agent.name = name
    agent.run = AsyncMock(return_value=result or _make_result())
    # bind() returns a BoundAgent-like handle whose .run dispatches to the
    # current agent.run attribute — tests that swap agent.run still work.
    handle = Mock()

    async def _forward(*a, **kw):
        return await agent.run(*a, **kw)

    handle.run = _forward
    agent.bind = Mock(return_value=handle)
    agent.set_cancellation_token = Mock()
    return agent


# --- Schema ---


class TestAgentToolSchema:
    def test_schema_uses_agent_name_by_default(self):
        emitter = _make_emitter()
        agent = _make_agent(name="researcher")
        tool = AgentTool(
            agent=agent,
            emitter=emitter,
            description="Researches topics",
        )
        assert tool.schema.name == "researcher"

    def test_schema_uses_custom_name(self):
        emitter = _make_emitter()
        agent = _make_agent(name="researcher")
        tool = AgentTool(
            agent=agent,
            emitter=emitter,
            description="Researches topics",
            name="research_agent",
        )
        assert tool.schema.name == "research_agent"

    def test_schema_has_description(self):
        emitter = _make_emitter()
        agent = _make_agent()
        tool = AgentTool(
            agent=agent,
            emitter=emitter,
            description="Finds relevant documents",
        )
        assert tool.schema.description == "Finds relevant documents"

    def test_schema_has_task_parameter(self):
        emitter = _make_emitter()
        agent = _make_agent()
        tool = AgentTool(agent=agent, emitter=emitter, description="Does things")
        params = tool.schema.parameters
        assert params["type"] == "object"
        assert "task" in params["properties"]
        assert params["properties"]["task"]["type"] == "string"
        assert params["required"] == ["task"]

    def test_satisfies_tool_protocol(self):
        emitter = _make_emitter()
        agent = _make_agent()
        tool = AgentTool(agent=agent, emitter=emitter, description="Does things")
        assert isinstance(tool, Tool)


# --- Execution ---


class TestAgentToolExecution:
    async def test_passes_task_to_agent(self):
        emitter = _make_emitter()
        agent = _make_agent()
        tool = AgentTool(agent=agent, emitter=emitter, description="desc")
        await tool.execute(task="find papers on AI")
        agent.run.assert_awaited_once_with("find papers on AI")

    async def test_returns_tool_result_with_content(self):
        emitter = _make_emitter()
        result = _make_result(output="42 papers found")
        agent = _make_agent(result=result)
        tool = AgentTool(agent=agent, emitter=emitter, description="desc")
        tool_result = await tool.execute(task="count papers")
        assert tool_result.content == "42 papers found"

    async def test_metadata_contains_agent_result_fields(self):
        emitter = _make_emitter()
        result = _make_result(total_steps=3, termination_reason="complete")
        agent = _make_agent(result=result)
        tool = AgentTool(agent=agent, emitter=emitter, description="desc")
        tool_result = await tool.execute(task="do something")
        assert tool_result.metadata["total_steps"] == 3
        assert tool_result.metadata["termination_reason"] == "complete"
        assert tool_result.metadata["usage"] == _make_usage().model_dump()

    async def test_emits_delegation_event(self):
        emitter = _make_emitter()
        agent = _make_agent(name="researcher")
        tool = AgentTool(agent=agent, emitter=emitter, description="desc")
        await tool.execute(task="research topic X")
        delegation_events = [e for e in emitter.events if isinstance(e, DelegationEvent)]
        assert len(delegation_events) == 1
        evt = delegation_events[0]
        assert evt.caller_agent == ""
        assert evt.delegate_agent == "researcher"
        assert evt.task == "research topic X"
        assert evt.transfer_strategy == "RawOutputTransfer"

    async def test_emits_delegation_event_with_caller_name(self):
        emitter = _make_emitter()
        agent = _make_agent(name="researcher")
        tool = AgentTool(agent=agent, emitter=emitter, description="desc", caller_name="orchestrator")
        await tool.execute(task="research topic X")
        delegation_events = [e for e in emitter.events if isinstance(e, DelegationEvent)]
        assert len(delegation_events) == 1
        assert delegation_events[0].caller_agent == "orchestrator"

    async def test_error_propagates_from_agent(self):
        emitter = _make_emitter()
        agent = _make_agent()
        agent.run = AsyncMock(side_effect=RuntimeError("agent failed"))
        tool = AgentTool(agent=agent, emitter=emitter, description="desc")
        with pytest.raises(RuntimeError, match="agent failed"):
            await tool.execute(task="do something")


# --- Transfer Strategies ---


class TestAgentToolTransferStrategies:
    async def test_raw_output_transfer(self):
        emitter = _make_emitter()
        result = _make_result(output="raw output")
        agent = _make_agent(result=result)
        tool = AgentTool(
            agent=agent,
            emitter=emitter,
            description="desc",
            transfer_strategy=RawOutputTransfer(),
        )
        tool_result = await tool.execute(task="do it")
        assert tool_result.content == "raw output"

    async def test_trajectory_transfer(self):
        emitter = _make_emitter()
        messages = [
            Message(role="user", content="question"),
            Message(role="assistant", content="answer"),
        ]
        result = _make_result(messages=messages)
        agent = _make_agent(result=result)
        tool = AgentTool(
            agent=agent,
            emitter=emitter,
            description="desc",
            transfer_strategy=TrajectoryTransfer(),
        )
        tool_result = await tool.execute(task="do it")
        assert "USER: question" in tool_result.content
        assert "ASSISTANT: answer" in tool_result.content

    async def test_summary_transfer(self):
        mock_llm = MockLLMClient(
            responses=[
                LLMResponse(
                    content="Summarized: the agent found 42 papers.",
                    usage=_make_usage(),
                    model="mock",
                    stop_reason="end_turn",
                ),
            ]
        )
        emitter = _make_emitter()
        messages = [
            Message(role="user", content="find papers"),
            Message(role="assistant", content="found 42"),
        ]
        result = _make_result(messages=messages)
        agent = _make_agent(result=result)
        tool = AgentTool(
            agent=agent,
            emitter=emitter,
            description="desc",
            transfer_strategy=SummaryTransfer(llm_client=mock_llm),
        )
        tool_result = await tool.execute(task="find papers")
        assert tool_result.content == "Summarized: the agent found 42 papers."

    async def test_custom_transfer(self):
        emitter = _make_emitter()
        result = _make_result(output="raw", total_steps=5)
        agent = _make_agent(result=result)
        tool = AgentTool(
            agent=agent,
            emitter=emitter,
            description="desc",
            transfer_strategy=CustomTransfer(fn=lambda r: f"Steps: {r.total_steps}, Output: {r.output}"),
        )
        tool_result = await tool.execute(task="do it")
        assert tool_result.content == "Steps: 5, Output: raw"

    async def test_delegation_event_records_strategy_name(self):
        emitter = _make_emitter()
        agent = _make_agent()
        tool = AgentTool(
            agent=agent,
            emitter=emitter,
            description="desc",
            transfer_strategy=TrajectoryTransfer(),
        )
        await tool.execute(task="task")
        delegation_events = [e for e in emitter.events if isinstance(e, DelegationEvent)]
        assert delegation_events[0].transfer_strategy == "TrajectoryTransfer"


# --- Content Blocks ---


class TestAgentToolContentBlocks:
    async def test_passes_task_directly_when_no_content_blocks(self):
        emitter = _make_emitter()
        agent = _make_agent()
        tool = AgentTool(agent=agent, emitter=emitter, description="desc")
        await tool.execute(task="find papers on AI")
        agent.run.assert_awaited_once_with("find papers on AI")

    async def test_assembles_multimodal_input_with_content_blocks(self):
        emitter = _make_emitter()
        agent = _make_agent()
        image_block = ImageContentBlock(media_type="image/png", data="base64data")
        tool = AgentTool(
            agent=agent,
            emitter=emitter,
            description="desc",
            content_blocks=[image_block],
        )
        await tool.execute(task="describe this image")
        agent.run.assert_awaited_once_with(
            [
                TextContentBlock(text="describe this image"),
                image_block,
            ]
        )

    async def test_delegation_event_stores_original_task_with_content_blocks(self):
        emitter = _make_emitter()
        agent = _make_agent(name="analyst")
        image_block = ImageContentBlock(media_type="image/jpeg", data="imgdata")
        tool = AgentTool(
            agent=agent,
            emitter=emitter,
            description="desc",
            content_blocks=[image_block],
            caller_name="orchestrator",
        )
        await tool.execute(task="analyze the document")
        delegation_events = [e for e in emitter.events if isinstance(e, DelegationEvent)]
        assert len(delegation_events) == 1
        assert delegation_events[0].task == "analyze the document"
        assert delegation_events[0].caller_agent == "orchestrator"

    async def test_content_blocks_with_transfer_strategy(self):
        emitter = _make_emitter()
        result = _make_result(output="extracted", total_steps=1)
        agent = _make_agent(result=result)
        image_block = ImageContentBlock(media_type="image/png", data="base64data")
        tool = AgentTool(
            agent=agent,
            emitter=emitter,
            description="desc",
            content_blocks=[image_block],
            transfer_strategy=CustomTransfer(fn=lambda r: f"Result: {r.output}"),
        )
        tool_result = await tool.execute(task="extract data")
        # Content blocks affect agent input
        agent.run.assert_awaited_once_with(
            [
                TextContentBlock(text="extract data"),
                image_block,
            ]
        )
        # Transfer strategy affects output extraction
        assert tool_result.content == "Result: extracted"


# --- Integration: ReActAgent with AgentTool ---


class TestAgentToolIntegration:
    """Integration test: ReActAgent caller with AgentTool delegate."""

    async def test_react_agent_uses_agent_tool(self):
        from nanitics.infrastructure.llm.protocol import ToolCall
        from nanitics.infrastructure.observability.events import (
            SpanStartEvent,
        )
        from nanitics.strategies.agents.react import ReActAgent

        emitter = _make_emitter()
        all_events: list[object] = []
        emitter.add_listener(lambda e: all_events.append(e))

        # Set up delegate agent — simple ReasoningAgent that returns a direct answer
        delegate_llm = MockLLMClient(
            responses=[
                LLMResponse(
                    content="The capital of France is Paris.",
                    tool_calls=[],
                    usage=_make_usage(),
                    model="mock",
                    stop_reason="end_turn",
                ),
            ]
        )
        from nanitics.strategies.agents.reasoning import ReasoningAgent

        delegate = ReasoningAgent(
            name="knowledge-agent",
            llm_client=delegate_llm,
            emitter=emitter,
            system_prompt="You answer factual questions.",
        )

        agent_tool = AgentTool(
            agent=delegate,
            emitter=emitter,
            description="Answers factual questions about geography",
        )

        # Set up caller agent — calls the agent tool, then produces final answer
        caller_llm = MockLLMClient(
            responses=[
                # Step 1: caller decides to use the delegate tool
                LLMResponse(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="tc1",
                            name="knowledge-agent",
                            arguments={"task": "What is the capital of France?"},
                        )
                    ],
                    usage=_make_usage(),
                    model="mock",
                    stop_reason="tool_use",
                ),
                # Step 2: caller produces final answer after receiving tool result
                LLMResponse(
                    content="Based on my research, the capital of France is Paris.",
                    tool_calls=[],
                    usage=_make_usage(),
                    model="mock",
                    stop_reason="end_turn",
                ),
            ]
        )

        caller = ReActAgent(
            name="supervisor",
            llm_client=caller_llm,
            emitter=emitter,
            system_prompt="You are a supervisor that delegates to specialists.",
            tools=[agent_tool],
        )

        result = await caller.run("What is the capital of France?")

        # Verify the delegation worked
        assert result.output == "Based on my research, the capital of France is Paris."
        assert result.total_steps == 2

        # Verify delegation event was emitted
        delegation_events = [e for e in all_events if isinstance(e, DelegationEvent)]
        assert len(delegation_events) == 1
        assert delegation_events[0].delegate_agent == "knowledge-agent"
        assert delegation_events[0].task == "What is the capital of France?"

        # Verify nested span structure:
        # supervisor span > step-1 span > knowledge-agent span
        span_starts = [e for e in all_events if isinstance(e, SpanStartEvent)]
        span_names = [s.name for s in span_starts]
        assert "supervisor" in span_names
        assert "knowledge-agent" in span_names
        assert "step-1" in span_names

        # knowledge-agent span should be nested under step-1
        supervisor_span = next(s for s in span_starts if s.name == "supervisor")
        step1_span = next(s for s in span_starts if s.name == "step-1")
        knowledge_span = next(s for s in span_starts if s.name == "knowledge-agent")
        assert step1_span.parent_span_id == supervisor_span.span_id
        # knowledge-agent span's parent is the child emitter's root span,
        # which is linked to the AgentTool's emitter span (step-1's span)
        assert knowledge_span.parent_span_id is not None
