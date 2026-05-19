"""Tests for agent metadata enrichment.

Verifies that each agent type emits correct `agent_type` and `capabilities`
in `AgentStartEvent`.
"""

import json

from nanitics import (
    ErrorHandler,
    InMemoryEmitter,
    MockEmbeddingClient,
    MockLLMClient,
    ReActAgent,
    ReasoningAgent,
    tool,
)
from nanitics.capabilities.memory.episodic import (
    EpisodicMemoryProvider,
    InMemoryEpisodeStore,
)
from nanitics.capabilities.memory.working_memory import InMemoryWorkingMemory, WorkingMemoryProvider
from nanitics.capabilities.planning.store import InMemoryPlanStore
from nanitics.infrastructure.observability.events import AgentStartEvent
from nanitics.safety.cancellation import CancellationToken
from nanitics.safety.sandbox.protocol import ExecutionResult
from nanitics.specialized import (
    ReflexionAgent,
    ReWOOAgent,
)
from nanitics.strategies.agents.codeact import CodeActAgent
from nanitics.strategies.agents.evaluation import (
    EvaluationContext,
    EvaluationResult,
    EvaluationVerdict,
)
from nanitics.strategies.agents.lats import LATSAgent
from nanitics.strategies.agents.tree_of_thought import (
    SearchStrategy,
    TreeOfThoughtAgent,
    _Candidate,
    _GenerationResponse,
)
from tests.testing_helpers import make_emitter, make_response


class MockSandbox:
    def __init__(self, results: list[ExecutionResult]) -> None:
        self._results = list(results)
        self._index = 0

    async def start(self) -> None:
        pass

    async def execute(self, code: str) -> ExecutionResult:
        if self._index < len(self._results):
            result = self._results[self._index]
            self._index += 1
            return result
        return ExecutionResult(stdout="", stderr="", return_value=None, success=True, error=None, duration_ms=1.0)

    async def reset(self) -> None:
        pass

    async def cleanup(self) -> None:
        pass


class _AcceptEvaluator:
    @property
    def max_revisions(self) -> int:
        return 0

    async def evaluate(self, output: str, context: EvaluationContext) -> EvaluationResult:
        return EvaluationResult(
            verdict=EvaluationVerdict.ACCEPT,
            score=0.9,
            evaluator_name="test-evaluator",
        )


class _StubContextManager:
    async def prepare(self, system_prompt, messages, tools=None, emitter=None):
        return messages

    def reset(self):
        pass


class _StubErrorHandler:
    async def handle_llm_error(self, error, retry_fn, emitter=None):
        raise error

    def handle_tool_error(self, error, attempt, available_tools):
        return None

    def handle_llm_correction(self, error, attempt):
        return None

    def should_degrade(self, error, attempt):
        return False

    def format_degradation_message(self, error):
        return ""

    def reset(self):
        pass

    @property
    def total_corrections(self):
        return 0

    @property
    def max_corrections(self):
        return 1

    def restore(self, total_corrections):
        pass


@tool(name="search", description="Search the web")
async def search_tool(query: str) -> str:
    return f"Results for: {query}"


def _get_start_event(emitter: InMemoryEmitter, agent_name: str | None = None) -> AgentStartEvent:
    """Return the single ``AgentStartEvent`` on ``emitter``.

    Composite agents (e.g., ``ReflexionAgent``) bind their inner agent to the
    outer emitter, so inner-agent start events are forwarded into the outer
    emitter's ``events`` list alongside the outer start event. Tests that
    construct composites must pass ``agent_name`` to disambiguate.
    """
    events = [e for e in emitter.events if isinstance(e, AgentStartEvent)]
    if agent_name is not None:
        events = [e for e in events if e.agent_name == agent_name]
    assert len(events) == 1, f"expected exactly one AgentStartEvent (agent_name={agent_name!r}), got {len(events)}"
    return events[0]


# ──────────────────────────────────────────────────────────
# Agent Type Tests
# ──────────────────────────────────────────────────────────


class TestAgentType:
    async def test_reasoning_agent_type(self) -> None:
        emitter = make_emitter()
        agent = ReasoningAgent(
            name="r",
            llm_client=MockLLMClient([make_response()]),
            emitter=emitter,
            system_prompt="p",
        )
        await agent.run("hi")
        assert _get_start_event(emitter).agent_type == "reasoning"

    async def test_react_agent_type(self) -> None:
        emitter = make_emitter()
        agent = ReActAgent(
            name="r",
            llm_client=MockLLMClient([make_response()]),
            emitter=emitter,
            system_prompt="p",
            tools=[search_tool],
        )
        await agent.run("hi")
        assert _get_start_event(emitter).agent_type == "react"

    async def test_codeact_agent_type(self) -> None:
        emitter = make_emitter()
        agent = CodeActAgent(
            name="c",
            llm_client=MockLLMClient([make_response()]),
            emitter=emitter,
            system_prompt="p",
            sandbox=MockSandbox([]),
        )
        await agent.run("hi")
        assert _get_start_event(emitter).agent_type == "codeact"

    async def test_rewoo_agent_type(self) -> None:
        plan_json = json.dumps(
            {
                "steps": [
                    {
                        "step_number": 1,
                        "description": "s",
                        "tool_name": "search",
                        "arguments": {"query": "q"},
                        "depends_on": [],
                    }
                ]
            }
        )
        emitter = make_emitter()
        agent = ReWOOAgent(
            name="w",
            llm_client=MockLLMClient([make_response(content=plan_json), make_response(content="done")]),
            emitter=emitter,
            system_prompt="p",
            tools=[search_tool],
            plan_store=InMemoryPlanStore(),
        )
        await agent.run("hi")
        assert _get_start_event(emitter).agent_type == "rewoo"

    async def test_tree_of_thought_agent_type(self) -> None:
        gen_response = _GenerationResponse(
            candidates=[_Candidate(reasoning="thought", is_complete=True)]
        ).model_dump_json()
        emitter = make_emitter()
        agent = TreeOfThoughtAgent(
            name="t",
            llm_client=MockLLMClient([make_response(content=gen_response)]),
            emitter=emitter,
            system_prompt="p",
            node_evaluator=_AcceptEvaluator(),
            search_strategy=SearchStrategy.BFS,
        )
        await agent.run("hi")
        assert _get_start_event(emitter).agent_type == "tree_of_thought"

    async def test_lats_agent_type(self) -> None:
        # LATS needs a terminal answer on first expansion
        emitter = make_emitter()
        agent = LATSAgent(
            name="l",
            llm_client=MockLLMClient([make_response(content="answer")]),
            emitter=emitter,
            system_prompt="p",
            tools=[search_tool],
            node_evaluator=_AcceptEvaluator(),
            max_iterations=1,
            branching_factor=1,
        )
        await agent.run("hi")
        assert _get_start_event(emitter).agent_type == "lats"

    async def test_reflexion_agent_type(self) -> None:
        emitter = make_emitter()
        inner = ReasoningAgent(
            name="inner",
            llm_client=MockLLMClient([make_response()]),
            emitter=make_emitter(),
            system_prompt="p",
        )
        agent = ReflexionAgent(
            name="ref",
            llm_client=MockLLMClient([]),
            emitter=emitter,
            system_prompt="p",
            inner_agent=inner,
            evaluator=_AcceptEvaluator(),
            episode_store=InMemoryEpisodeStore(MockEmbeddingClient()),
        )
        await agent.run("hi")
        assert _get_start_event(emitter, agent_name="ref").agent_type == "reflexion"


# ──────────────────────────────────────────────────────────
# Capabilities Tests
# ──────────────────────────────────────────────────────────


class TestCapabilities:
    async def test_reasoning_base_capabilities_includes_error_handling(self) -> None:
        """ReasoningAgent with no extras has error_handling from the default ErrorHandler."""
        emitter = make_emitter()
        agent = ReasoningAgent(
            name="r",
            llm_client=MockLLMClient([make_response()]),
            emitter=emitter,
            system_prompt="p",
        )
        await agent.run("hi")
        assert _get_start_event(emitter).capabilities == ["error_handling"]

    async def test_fail_fast_handler_omits_error_handling_capability(self) -> None:
        """ErrorHandler.fail_fast() does not report error_handling capability."""
        emitter = make_emitter()
        agent = ReasoningAgent(
            name="r",
            llm_client=MockLLMClient([make_response()]),
            emitter=emitter,
            system_prompt="p",
            error_handler=ErrorHandler.fail_fast(),
        )
        await agent.run("hi")
        assert "error_handling" not in _get_start_event(emitter).capabilities

    async def test_evaluation_capability_detected(self) -> None:
        emitter = make_emitter()
        agent = ReasoningAgent(
            name="r",
            llm_client=MockLLMClient([make_response()]),
            emitter=emitter,
            system_prompt="p",
            output_evaluator=_AcceptEvaluator(),
        )
        await agent.run("hi")
        assert "evaluation" in _get_start_event(emitter).capabilities

    async def test_cancellation_capability_detected(self) -> None:
        emitter = make_emitter()
        token = CancellationToken()
        agent = ReasoningAgent(
            name="r",
            llm_client=MockLLMClient([make_response()]),
            emitter=emitter,
            system_prompt="p",
            cancellation_token=token,
        )
        await agent.run("hi")
        assert "cancellation" in _get_start_event(emitter).capabilities

    async def test_context_management_capability_detected(self) -> None:
        emitter = make_emitter()
        agent = ReasoningAgent(
            name="r",
            llm_client=MockLLMClient([make_response()]),
            emitter=emitter,
            system_prompt="p",
            context_manager=_StubContextManager(),
        )
        await agent.run("hi")
        assert "context_management" in _get_start_event(emitter).capabilities

    async def test_error_handling_capability_detected(self) -> None:
        emitter = make_emitter()
        agent = ReasoningAgent(
            name="r",
            llm_client=MockLLMClient([make_response()]),
            emitter=emitter,
            system_prompt="p",
            error_handler=_StubErrorHandler(),
        )
        await agent.run("hi")
        assert "error_handling" in _get_start_event(emitter).capabilities

    async def test_streaming_capability_detected(self) -> None:
        emitter = make_emitter()
        agent = ReActAgent(
            name="r",
            llm_client=MockLLMClient([make_response()]),
            emitter=emitter,
            system_prompt="p",
            tools=[search_tool],
            streaming=True,
        )
        await agent.run("hi")
        assert "streaming" in _get_start_event(emitter).capabilities

    async def test_react_has_tool_use(self) -> None:
        emitter = make_emitter()
        agent = ReActAgent(
            name="r",
            llm_client=MockLLMClient([make_response()]),
            emitter=emitter,
            system_prompt="p",
            tools=[search_tool],
        )
        await agent.run("hi")
        assert "tool_use" in _get_start_event(emitter).capabilities

    async def test_codeact_has_code_execution(self) -> None:
        emitter = make_emitter()
        agent = CodeActAgent(
            name="c",
            llm_client=MockLLMClient([make_response()]),
            emitter=emitter,
            system_prompt="p",
            sandbox=MockSandbox([]),
        )
        await agent.run("hi")
        caps = _get_start_event(emitter).capabilities
        assert "code_execution" in caps

    async def test_codeact_with_tools_has_tool_use(self) -> None:
        emitter = make_emitter()
        agent = CodeActAgent(
            name="c",
            llm_client=MockLLMClient([make_response()]),
            emitter=emitter,
            system_prompt="p",
            sandbox=MockSandbox(
                [
                    ExecutionResult(stdout="", stderr="", return_value=None, success=True, error=None, duration_ms=1.0),
                ]
            ),
            tools=[search_tool],
        )
        await agent.run("hi")
        caps = _get_start_event(emitter).capabilities
        assert "code_execution" in caps
        assert "tool_use" in caps

    async def test_codeact_without_tools_no_tool_use(self) -> None:
        emitter = make_emitter()
        agent = CodeActAgent(
            name="c",
            llm_client=MockLLMClient([make_response()]),
            emitter=emitter,
            system_prompt="p",
            sandbox=MockSandbox([]),
        )
        await agent.run("hi")
        caps = _get_start_event(emitter).capabilities
        assert "tool_use" not in caps

    async def test_rewoo_has_planning_and_tool_use(self) -> None:
        plan_json = json.dumps(
            {
                "steps": [
                    {
                        "step_number": 1,
                        "description": "s",
                        "tool_name": "search",
                        "arguments": {"query": "q"},
                        "depends_on": [],
                    }
                ]
            }
        )
        emitter = make_emitter()
        agent = ReWOOAgent(
            name="w",
            llm_client=MockLLMClient([make_response(content=plan_json), make_response(content="done")]),
            emitter=emitter,
            system_prompt="p",
            tools=[search_tool],
            plan_store=InMemoryPlanStore(),
        )
        await agent.run("hi")
        caps = _get_start_event(emitter).capabilities
        assert "planning" in caps
        assert "tool_use" in caps

    async def test_lats_has_tool_use(self) -> None:
        emitter = make_emitter()
        agent = LATSAgent(
            name="l",
            llm_client=MockLLMClient([make_response(content="answer")]),
            emitter=emitter,
            system_prompt="p",
            tools=[search_tool],
            node_evaluator=_AcceptEvaluator(),
            max_iterations=1,
            branching_factor=1,
        )
        await agent.run("hi")
        assert "tool_use" in _get_start_event(emitter).capabilities

    async def test_lats_with_episode_store_has_episodic_memory(self) -> None:
        emitter = make_emitter()
        agent = LATSAgent(
            name="l",
            llm_client=MockLLMClient([make_response(content="answer")]),
            emitter=emitter,
            system_prompt="p",
            tools=[search_tool],
            node_evaluator=_AcceptEvaluator(),
            max_iterations=1,
            branching_factor=1,
            episode_store=InMemoryEpisodeStore(MockEmbeddingClient()),
        )
        await agent.run("hi")
        caps = _get_start_event(emitter).capabilities
        assert "tool_use" in caps
        assert "episodic_memory" in caps

    async def test_lats_without_episode_store_no_episodic_memory(self) -> None:
        emitter = make_emitter()
        agent = LATSAgent(
            name="l",
            llm_client=MockLLMClient([make_response(content="answer")]),
            emitter=emitter,
            system_prompt="p",
            tools=[search_tool],
            node_evaluator=_AcceptEvaluator(),
            max_iterations=1,
            branching_factor=1,
        )
        await agent.run("hi")
        assert "episodic_memory" not in _get_start_event(emitter).capabilities

    async def test_reflexion_has_episodic_memory(self) -> None:
        emitter = make_emitter()
        inner = ReasoningAgent(
            name="inner",
            llm_client=MockLLMClient([make_response()]),
            emitter=make_emitter(),
            system_prompt="p",
        )
        agent = ReflexionAgent(
            name="ref",
            llm_client=MockLLMClient([]),
            emitter=emitter,
            system_prompt="p",
            inner_agent=inner,
            evaluator=_AcceptEvaluator(),
            episode_store=InMemoryEpisodeStore(MockEmbeddingClient()),
        )
        await agent.run("hi")
        assert "episodic_memory" in _get_start_event(emitter, agent_name="ref").capabilities

    async def test_working_memory_provider_detected(self) -> None:
        emitter = make_emitter()
        wm = InMemoryWorkingMemory()
        provider = WorkingMemoryProvider(memory=wm)
        agent = ReActAgent(
            name="r",
            llm_client=MockLLMClient([make_response()]),
            emitter=emitter,
            system_prompt="p",
            tools=[search_tool],
            context_providers=[provider],
        )
        await agent.run("hi")
        assert "working_memory" in _get_start_event(emitter).capabilities

    async def test_episodic_memory_provider_detected(self) -> None:
        emitter = make_emitter()
        store = InMemoryEpisodeStore(MockEmbeddingClient())
        provider = EpisodicMemoryProvider(store=store)
        agent = ReActAgent(
            name="r",
            llm_client=MockLLMClient([make_response()]),
            emitter=emitter,
            system_prompt="p",
            tools=[search_tool],
            context_providers=[provider],
        )
        await agent.run("hi")
        assert "episodic_memory" in _get_start_event(emitter).capabilities

    async def test_backward_compatible_defaults(self) -> None:
        """New fields have sensible defaults for backward compatibility."""
        event = AgentStartEvent(
            trace_id="t",
            span_id="s",
            agent_name="a",
            task_input="i",
            tools_available=[],
        )
        assert event.agent_type is None
        assert event.capabilities == []
