import threading
from unittest.mock import AsyncMock, Mock

from nanitics.composition.multi_agent.agent_tool import AgentTool
from nanitics.composition.multi_agent.orchestrator import create_orchestrator
from nanitics.core.agents.base import AgentResult
from nanitics.core.agents.reasoning import ReasoningAgent
from nanitics.infrastructure.llm.mock import MockLLMClient
from nanitics.infrastructure.llm.protocol import LLMResponse, ToolCall
from nanitics.infrastructure.observability.emitter import InMemoryEmitter
from nanitics.infrastructure.observability.events import Usage
from nanitics.safety.cancellation import CancellationToken


def _make_usage() -> Usage:
    return Usage(input_tokens=10, output_tokens=5)


def _make_emitter() -> InMemoryEmitter:
    return InMemoryEmitter(trace_id="test-trace")


def _make_result(output: str = "done", termination_reason: str = "complete") -> AgentResult:
    return AgentResult(
        output=output,
        total_steps=1,
        termination_reason=termination_reason,
        messages=[],
        usage=_make_usage(),
    )


def _make_response(
    content: str | None = "response",
    tool_calls: list[ToolCall] | None = None,
) -> LLMResponse:
    return LLMResponse(
        content=content,
        tool_calls=tool_calls or [],
        usage=_make_usage(),
        model="test-model",
        stop_reason="end_turn" if not tool_calls else "tool_use",
    )


class TestCancellationTokenInitial:
    def test_starts_uncancelled(self):
        token = CancellationToken()
        assert token.is_cancelled is False


class TestCancellationTokenCancel:
    def test_cancel_sets_cancelled(self):
        token = CancellationToken()
        token.cancel()
        assert token.is_cancelled is True

    def test_cancel_is_idempotent(self):
        token = CancellationToken()
        token.cancel()
        token.cancel()
        token.cancel()
        assert token.is_cancelled is True


class TestCancellationTokenThreadSafety:
    def test_cancel_from_another_thread(self):
        token = CancellationToken()
        assert token.is_cancelled is False

        thread = threading.Thread(target=token.cancel)
        thread.start()
        thread.join()

        assert token.is_cancelled is True


class TestAgentCancellationTokenAccess:
    def test_cancellation_token_property_returns_token(self):
        token = CancellationToken()
        emitter = _make_emitter()
        agent = ReasoningAgent(
            name="test",
            llm_client=MockLLMClient([_make_response()]),
            emitter=emitter,
            system_prompt="You are a test agent.",
            cancellation_token=token,
        )
        assert agent.cancellation_token is token

    def test_cancellation_token_property_returns_none_by_default(self):
        emitter = _make_emitter()
        agent = ReasoningAgent(
            name="test",
            llm_client=MockLLMClient([_make_response()]),
            emitter=emitter,
            system_prompt="You are a test agent.",
        )
        assert agent.cancellation_token is None

    def test_set_cancellation_token(self):
        emitter = _make_emitter()
        agent = ReasoningAgent(
            name="test",
            llm_client=MockLLMClient([_make_response()]),
            emitter=emitter,
            system_prompt="You are a test agent.",
        )
        token = CancellationToken()
        agent.set_cancellation_token(token)
        assert agent.cancellation_token is token


class TestAgentToolCancellationPropagation:
    async def test_agent_tool_propagates_token_to_delegate(self):
        token = CancellationToken()
        emitter = _make_emitter()
        agent = AsyncMock()
        agent.name = "delegate"
        agent.run = AsyncMock(return_value=_make_result())
        handle = Mock()
        handle.run = agent.run
        agent.bind = Mock(return_value=handle)
        agent.set_cancellation_token = Mock()

        tool = AgentTool(
            agent=agent,
            emitter=emitter,
            description="A delegate agent",
            cancellation_token=token,
        )
        await tool.execute(task="do something")
        agent.set_cancellation_token.assert_called_once_with(token)

    async def test_agent_tool_without_token_does_not_call_set(self):
        emitter = _make_emitter()
        agent = AsyncMock()
        agent.name = "delegate"
        agent.run = AsyncMock(return_value=_make_result())
        handle = Mock()
        handle.run = agent.run
        agent.bind = Mock(return_value=handle)
        agent.set_cancellation_token = Mock()

        tool = AgentTool(
            agent=agent,
            emitter=emitter,
            description="A delegate agent",
        )
        await tool.execute(task="do something")
        agent.set_cancellation_token.assert_not_called()


class TestOrchestratorCancellationPropagation:
    def test_create_orchestrator_distributes_token_to_specialists(self):
        token = CancellationToken()
        emitter = _make_emitter()

        s1 = AgentTool(
            agent=ReasoningAgent(
                name="s1",
                llm_client=MockLLMClient([_make_response()]),
                emitter=emitter,
                system_prompt="You are s1.",
            ),
            emitter=emitter,
            description="Specialist 1",
        )
        s2 = AgentTool(
            agent=ReasoningAgent(
                name="s2",
                llm_client=MockLLMClient([_make_response()]),
                emitter=emitter,
                system_prompt="You are s2.",
            ),
            emitter=emitter,
            description="Specialist 2",
        )

        create_orchestrator(
            name="orch",
            llm_client=MockLLMClient([_make_response()]),
            emitter=emitter,
            specialists=[s1, s2],
            cancellation_token=token,
        )

        assert s1.cancellation_token is token
        assert s2.cancellation_token is token

    def test_create_orchestrator_without_token_leaves_specialists_unset(self):
        emitter = _make_emitter()

        s1 = AgentTool(
            agent=ReasoningAgent(
                name="s1",
                llm_client=MockLLMClient([_make_response()]),
                emitter=emitter,
                system_prompt="You are s1.",
            ),
            emitter=emitter,
            description="Specialist 1",
        )

        create_orchestrator(
            name="orch",
            llm_client=MockLLMClient([_make_response()]),
            emitter=emitter,
            specialists=[s1],
        )

        assert s1.cancellation_token is None

    async def test_cancelled_token_stops_delegate_agent(self):
        token = CancellationToken()
        emitter = _make_emitter()
        from nanitics.core.agents.react import ReActAgent

        # ReActAgent checks _is_cancelled at the start of each loop iteration
        delegate = ReActAgent(
            name="delegate",
            llm_client=MockLLMClient([_make_response("I should not finish")]),
            emitter=emitter,
            system_prompt="You are a delegate.",
            tools=[],
        )

        token.cancel()
        delegate.set_cancellation_token(token)

        result = await delegate.run("do work")
        assert result.termination_reason == "cancelled"

    async def test_three_level_hierarchy_cancellation_cascade(self):
        """3-level hierarchy shares one token; cancellation cascades to all agents."""
        token = CancellationToken()
        emitter = _make_emitter()

        # Leaf agents
        leaf1 = ReasoningAgent(
            name="leaf1",
            llm_client=MockLLMClient([_make_response("leaf1 output")]),
            emitter=emitter,
            system_prompt="You are leaf1.",
        )
        leaf2 = ReasoningAgent(
            name="leaf2",
            llm_client=MockLLMClient([_make_response("leaf2 output")]),
            emitter=emitter,
            system_prompt="You are leaf2.",
        )

        # Mid-level orchestrator specialists
        mid_s1 = AgentTool(agent=leaf1, emitter=emitter, description="Leaf 1")
        mid_s2 = AgentTool(agent=leaf2, emitter=emitter, description="Leaf 2")

        mid_orchestrator = create_orchestrator(
            name="mid-orch",
            llm_client=MockLLMClient([_make_response("mid result")]),
            emitter=emitter,
            specialists=[mid_s1, mid_s2],
            cancellation_token=token,
        )

        # Top-level orchestrator
        top_specialist = AgentTool(agent=mid_orchestrator, emitter=emitter, description="Mid orchestrator")

        create_orchestrator(
            name="top-orch",
            llm_client=MockLLMClient([_make_response("top result")]),
            emitter=emitter,
            specialists=[top_specialist],
            cancellation_token=token,
        )

        # Token distributed to mid-level specialists
        assert mid_s1.cancellation_token is token
        assert mid_s2.cancellation_token is token
        # Token distributed to top-level specialist
        assert top_specialist.cancellation_token is token

        # Cancel and verify all agents stop
        token.cancel()

        result = await mid_orchestrator.run("do work")
        assert result.termination_reason == "cancelled"
