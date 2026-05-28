"""Per-agent-type wiring + reset tests for ``ToolResultPolicy``.

The ReAct path is covered by ``test_react_tool_result_policy.py``; this
file pins reset wiring on the three other tool-bearing agents
(``LATSAgent``, ``ReWOOAgent``, ``CodeActAgent``).
"""

from nanitics.capabilities.context.tool_result import ToolResultContext
from nanitics.capabilities.planning.store import InMemoryPlanStore
from nanitics.infrastructure import LLMResponse, MockLLMClient
from nanitics.safety import ExecutionResult, MockSandbox
from nanitics.specialized import ReWOOAgent, ReWOOPlan, ReWOOStep
from nanitics.strategies import tool
from nanitics.strategies.agents.codeact import CodeActAgent
from nanitics.strategies.agents.evaluation import (
    EvaluationContext,
    EvaluationResult,
    EvaluationVerdict,
)
from nanitics.strategies.agents.lats import LATSAgent
from nanitics.strategies.tools.protocol import ToolResult
from tests.testing_helpers import make_emitter, make_response, make_usage


class _ResetSpy:
    def __init__(self) -> None:
        self.reset_calls = 0

    async def apply(self, result: ToolResult, context: ToolResultContext) -> ToolResult:
        return result

    def reset(self) -> None:
        self.reset_calls += 1


class _AcceptEvaluator:
    @property
    def max_revisions(self) -> int:
        return 1

    async def evaluate(self, output: str, context: EvaluationContext) -> EvaluationResult:
        return EvaluationResult(verdict=EvaluationVerdict.ACCEPT, evaluator_name="t")


@tool(name="search", description="Search")
async def search_tool(query: str) -> str:
    return "ok"


class TestLATSWiring:
    async def test_policy_reset_called_on_run(self) -> None:
        spy = _ResetSpy()
        client = MockLLMClient([make_response(content="terminal answer") for _ in range(8)])
        agent = LATSAgent(
            name="lats",
            llm_client=client,
            emitter=make_emitter(),
            system_prompt="s",
            tools=[search_tool],
            node_evaluator=_AcceptEvaluator(),
            max_iterations=1,
            max_depth=3,
            branching_factor=1,
            tool_result_policy=spy,
        )
        await agent.run("go")
        assert spy.reset_calls == 1
        # Registry received the policy + a token counter
        assert agent._tool_registry._tool_result_policy is spy
        assert agent._tool_registry._token_counter is not None

    def test_no_policy_keeps_token_counter_unset(self) -> None:
        agent = LATSAgent(
            name="lats",
            llm_client=MockLLMClient([]),
            emitter=make_emitter(),
            system_prompt="s",
            tools=[search_tool],
            node_evaluator=_AcceptEvaluator(),
        )
        assert agent._tool_registry._tool_result_policy is None
        assert agent._tool_registry._token_counter is None


class TestReWOOWiring:
    async def test_policy_reset_called_on_run(self) -> None:
        spy = _ResetSpy()
        plan_json = ReWOOPlan(
            steps=[
                ReWOOStep(
                    step_number=1,
                    description="Search topic",
                    tool_name="search",
                    arguments={"query": "test"},
                    depends_on=[],
                )
            ]
        ).model_dump_json()
        client = MockLLMClient(
            [
                make_response(content=plan_json),
                make_response(content="final answer"),
            ]
        )
        agent = ReWOOAgent(
            name="rewoo",
            llm_client=client,
            emitter=make_emitter(),
            system_prompt="s",
            tools=[search_tool],
            plan_store=InMemoryPlanStore(),
            tool_result_policy=spy,
        )
        await agent.run("topic")
        assert spy.reset_calls == 1
        assert agent._tool_registry._tool_result_policy is spy

    def test_no_policy_keeps_token_counter_unset(self) -> None:
        agent = ReWOOAgent(
            name="rewoo",
            llm_client=MockLLMClient([]),
            emitter=make_emitter(),
            system_prompt="s",
            tools=[search_tool],
            plan_store=InMemoryPlanStore(),
        )
        assert agent._tool_registry._tool_result_policy is None
        assert agent._tool_registry._token_counter is None


def _llm_response(content: str = "answer") -> LLMResponse:
    return LLMResponse(
        content=content,
        tool_calls=[],
        usage=make_usage(),
        model="m",
        stop_reason="end_turn",
    )


class TestCodeActWiring:
    async def test_policy_reset_called_on_run(self) -> None:
        spy = _ResetSpy()
        # When CodeAct has tools, its first sandbox call injects stub code;
        # subsequent calls correspond to LLM-emitted code blocks. The LLM
        # returns plain text on its first turn here so only the stub
        # injection runs.
        stub_result = ExecutionResult(
            stdout="", stderr="", return_value=None, success=True, error=None, duration_ms=1.0
        )
        agent = CodeActAgent(
            name="codeact",
            llm_client=MockLLMClient([_llm_response("done")]),
            emitter=make_emitter(),
            system_prompt="s",
            sandbox=MockSandbox([stub_result]),
            tools=[search_tool],
            tool_result_policy=spy,
        )
        await agent.run("go")
        assert spy.reset_calls == 1
        assert agent._tool_registry is not None
        assert agent._tool_registry._tool_result_policy is spy

    def test_no_policy_keeps_token_counter_unset(self) -> None:
        agent = CodeActAgent(
            name="codeact",
            llm_client=MockLLMClient([]),
            emitter=make_emitter(),
            system_prompt="s",
            sandbox=MockSandbox([]),
            tools=[search_tool],
        )
        assert agent._tool_registry is not None
        assert agent._tool_registry._tool_result_policy is None
        assert agent._tool_registry._token_counter is None

    def test_codeact_without_tools_has_no_registry(self) -> None:
        agent = CodeActAgent(
            name="codeact",
            llm_client=MockLLMClient([]),
            emitter=make_emitter(),
            system_prompt="s",
            sandbox=MockSandbox([]),
        )
        assert agent._tool_registry is None


class TestBaseAgentDefault:
    """Base ``Agent`` stores ``tool_result_policy`` for non-tool-bearing subclasses."""

    def test_reasoning_agent_default_is_none(self) -> None:
        from nanitics.strategies import ReasoningAgent

        agent = ReasoningAgent(
            name="r",
            llm_client=MockLLMClient([]),
            emitter=make_emitter(),
            system_prompt="s",
        )
        assert agent._tool_result_policy is None
