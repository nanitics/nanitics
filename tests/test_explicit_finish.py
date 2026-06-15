"""Tests for explicit run completion (``require_explicit_finish``).

Covers the ``ToolSchema.human_channel`` marker, capability-aware environment
guidance, the auto-registered ``finish`` tool and its collision guard, the
loop's bare-text-non-terminal / finish-detection / evaluator-gate behavior,
structured output via ``finish`` arguments, and the crash-resume short-circuit.
"""

import json

import pytest
from pydantic import BaseModel

from nanitics.collaboration import (
    CallbackHumanInputProvider,
    HumanDecision,
    HumanInputResponse,
    create_ask_human_tool,
    create_request_approval_tool,
)
from nanitics.infrastructure import LLMResponse, MockLLMClient
from nanitics.infrastructure.llm._openai_format import _to_openai_tools
from nanitics.infrastructure.llm.anthropic import _to_anthropic_tools
from nanitics.infrastructure.llm.protocol import ToolSchema
from nanitics.strategies import ReActAgent, tool
from nanitics.strategies.agents.evaluation import (
    EvaluationContext,
    EvaluationResult,
    EvaluationVerdict,
)
from nanitics.strategies.tools import FunctionTool
from nanitics.tracing import Message, ToolCall

from .testing_helpers import make_emitter, make_response, make_usage


@tool(name="add", description="Add two numbers")
async def add_tool(a: int, b: int) -> str:
    return str(a + b)


def _approve(_request: object) -> HumanInputResponse:
    return HumanInputResponse(decision=HumanDecision.ANSWER, content="ok")


def _ask_human_tool() -> FunctionTool:
    return create_ask_human_tool(CallbackHumanInputProvider(_approve))


# ──────────────────────────────────────────────────────────
# Step 1 — ToolSchema.human_channel marker
# ──────────────────────────────────────────────────────────


class TestHumanChannelMarker:
    def test_default_false(self) -> None:
        @tool(name="t", description="d")
        async def t() -> str:
            return "x"

        assert t.schema.human_channel is False

    def test_ask_human_sets_flag(self) -> None:
        assert _ask_human_tool().schema.human_channel is True

    def test_request_approval_does_not_set_flag(self) -> None:
        approval = create_request_approval_tool(CallbackHumanInputProvider(_approve))
        assert approval.schema.human_channel is False

    def test_decorator_passes_flag(self) -> None:
        @tool(name="ask", description="d", human_channel=True)
        async def ask(question: str) -> str:
            return "x"

        assert ask.schema.human_channel is True

    def test_replace_round_trips_flag(self) -> None:
        @tool(name="t", description="d")
        async def t() -> str:
            return "x"

        assert t.replace(human_channel=True).schema.human_channel is True
        # Replacing an unrelated field preserves an already-set flag.
        marked = t.replace(human_channel=True)
        assert marked.replace(description="d2").schema.human_channel is True

    def test_function_tool_constructor_flag(self) -> None:
        async def fn(x: str) -> str:
            return x

        ft = FunctionTool(
            fn=fn,
            name="t",
            description="d",
            parameters_schema={"type": "object", "properties": {"x": {"type": "string"}}},
            human_channel=True,
        )
        assert ft.schema.human_channel is True

    def test_not_serialized_to_providers(self) -> None:
        schema = ToolSchema(
            name="ask_human",
            description="d",
            parameters={"type": "object", "properties": {}},
            human_channel=True,
        )
        anthropic_json = json.dumps(_to_anthropic_tools([schema]))
        openai_json = json.dumps(_to_openai_tools([schema]))
        assert "human_channel" not in anthropic_json
        assert "human_channel" not in openai_json


# ──────────────────────────────────────────────────────────
# Step 2 — Capability-aware environment guidance
# ──────────────────────────────────────────────────────────

_CURRENT_ENVIRONMENT_TEXT = (
    "You operate autonomously rather than as a conversational "
    "chatbot. Make reasonable assumptions when information is "
    "incomplete and state them explicitly."
)


class TestEnvironmentGuidance:
    def _agent(self, tools: list) -> ReActAgent:
        return ReActAgent(
            name="a",
            llm_client=MockLLMClient([]),
            emitter=make_emitter(),
            system_prompt="Base.",
            tools=tools,
        )

    def test_no_human_channel_keeps_exact_current_text(self) -> None:
        agent = self._agent([add_tool])
        assert _CURRENT_ENVIRONMENT_TEXT in agent._system_prompt

    def test_human_channel_uses_capability_aware_text(self) -> None:
        agent = self._agent([add_tool, _ask_human_tool()])
        prompt = agent._system_prompt
        assert "call `ask_human` rather than" in prompt
        assert "complete result, not a question" in prompt
        # The contradictory "make reasonable assumptions when information is
        # incomplete" instruction is gone, and no one-way assertion is added.
        assert _CURRENT_ENVIRONMENT_TEXT not in prompt
        assert "cannot reply" not in prompt
        assert "one-way" not in prompt

    def test_base_default_unchanged_when_guidance_none(self) -> None:
        # A non-ReAct agent (ReasoningAgent) does not pass environment_guidance,
        # so it keeps the standard text.
        from nanitics.strategies import ReasoningAgent

        agent = ReasoningAgent(
            name="r",
            llm_client=MockLLMClient([]),
            emitter=make_emitter(),
            system_prompt="Base.",
        )
        assert _CURRENT_ENVIRONMENT_TEXT in agent._system_prompt


# ──────────────────────────────────────────────────────────
# Step 4 — flag, finish tool, collision guard
# ──────────────────────────────────────────────────────────


class TestFinishToolRegistration:
    def test_no_finish_tool_by_default(self) -> None:
        agent = ReActAgent(
            name="a",
            llm_client=MockLLMClient([]),
            emitter=make_emitter(),
            system_prompt="Base.",
            tools=[add_tool],
        )
        assert "finish" not in agent._get_tools_available()

    def test_finish_tool_registered_when_enabled(self) -> None:
        agent = ReActAgent(
            name="a",
            llm_client=MockLLMClient([]),
            emitter=make_emitter(),
            system_prompt="Base.",
            tools=[add_tool],
            require_explicit_finish=True,
        )
        assert "finish" in agent._get_tools_available()

    def test_finish_tool_uses_output_schema_shape(self) -> None:
        class Answer(BaseModel):
            value: int
            label: str

        agent = ReActAgent(
            name="a",
            llm_client=MockLLMClient([]),
            emitter=make_emitter(),
            system_prompt="Base.",
            tools=[],
            output_schema=Answer,
            require_explicit_finish=True,
        )
        finish_schema = next(s for s in agent._tool_registry.list_schemas() if s.name == "finish")
        assert set(finish_schema.parameters["properties"]) == {"value", "label"}

    def test_collision_raises(self) -> None:
        @tool(name="finish", description="consumer finish")
        async def consumer_finish(x: str) -> str:
            return x

        with pytest.raises(ValueError, match="reserved"):
            ReActAgent(
                name="a",
                llm_client=MockLLMClient([]),
                emitter=make_emitter(),
                system_prompt="Base.",
                tools=[consumer_finish],
                require_explicit_finish=True,
            )

    def test_collision_allowed_when_mode_off(self) -> None:
        @tool(name="finish", description="consumer finish")
        async def consumer_finish(x: str) -> str:
            return x

        agent = ReActAgent(
            name="a",
            llm_client=MockLLMClient([]),
            emitter=make_emitter(),
            system_prompt="Base.",
            tools=[consumer_finish],
        )
        assert "finish" in agent._get_tools_available()


# ──────────────────────────────────────────────────────────
# Step 5 — loop behavior
# ──────────────────────────────────────────────────────────


def _finish_call(result: str, call_id: str = "f1") -> ToolCall:
    return ToolCall(id=call_id, name="finish", arguments={"result": result})


class TestExplicitFinishLoop:
    def _agent(self, client: MockLLMClient, **kwargs) -> ReActAgent:
        return ReActAgent(
            name="a",
            llm_client=client,
            emitter=make_emitter(),
            system_prompt="Base.",
            tools=kwargs.pop("tools", [add_tool]),
            require_explicit_finish=True,
            **kwargs,
        )

    async def test_finish_terminates_run(self) -> None:
        client = MockLLMClient([make_response(content="here", tool_calls=[_finish_call("the answer")])])
        agent = self._agent(client)
        result = await agent.run("go")
        assert result.output == "the answer"
        assert result.termination_reason == "finished"
        assert result.parsed is None

    async def test_bare_text_is_non_terminal_then_finish(self) -> None:
        client = MockLLMClient(
            [
                make_response(content="Should I use prod or staging?"),
                make_response(content="ok", tool_calls=[_finish_call("done")]),
            ]
        )
        agent = self._agent(client)
        result = await agent.run("go")
        assert result.termination_reason == "finished"
        assert result.output == "done"
        # The bare-text turn produced a nudge user message in history, and the
        # bare text itself never became the output.
        assert any("finishing is the only way" in (m.content or "") for m in result.messages)
        assert result.output != "Should I use prod or staging?"

    async def test_nudge_offers_ask_human_when_present(self) -> None:
        client = MockLLMClient(
            [
                make_response(content="a question"),
                make_response(content="ok", tool_calls=[_finish_call("done")]),
            ]
        )
        agent = self._agent(client, tools=[add_tool, _ask_human_tool()])
        result = await agent.run("go")
        assert any("ask_human" in (m.content or "") for m in result.messages)

    async def test_nudge_text_no_human_channel(self) -> None:
        agent = self._agent(MockLLMClient([]), tools=[add_tool])
        nudge = agent._explicit_finish_nudge()
        assert "ask_human" not in nudge
        assert "finishing is the only way" in nudge

    async def test_nudge_loop_hits_iteration_limit_without_leaking_text(self) -> None:
        # Model never finishes: every turn is bare text. The run must end on the
        # iteration limit with no output, never on the bare text.
        client = MockLLMClient([make_response(content=f"question {i}") for i in range(5)])
        agent = self._agent(client, max_iterations=3)
        result = await agent.run("go")
        assert result.termination_reason == "iteration_limit"
        assert result.output is None

    async def test_finish_with_output_schema_populates_parsed_no_extra_call(self) -> None:
        class Answer(BaseModel):
            value: int

        finish = ToolCall(id="f1", name="finish", arguments={"value": 42})
        client = MockLLMClient([make_response(content="computing", tool_calls=[finish])])
        agent = self._agent(client, tools=[], output_schema=Answer)
        result = await agent.run("go")
        assert result.termination_reason == "finished"
        assert result.parsed == Answer(value=42)
        assert json.loads(result.output) == {"value": 42}
        # Exactly one LLM call: no structured-synthesis follow-up call.
        assert len(client.calls) == 1
        assert all(c["output_schema"] is None for c in client.calls)

    async def test_finish_invalid_args_is_not_terminal(self) -> None:
        # finish called without the required `result` arg: the dispatch errors
        # (correction), the run does not terminate, and the model finishes next.
        bad_finish = ToolCall(id="f1", name="finish", arguments={"wrong": "x"})
        client = MockLLMClient(
            [
                make_response(content="oops", tool_calls=[bad_finish]),
                make_response(content="ok", tool_calls=[_finish_call("recovered", call_id="f2")]),
            ]
        )
        agent = self._agent(client)
        result = await agent.run("go")
        assert result.termination_reason == "finished"
        assert result.output == "recovered"

    async def test_finish_schema_invalid_args_is_not_terminal(self) -> None:
        # output_schema mode: finish called with args that fail schema
        # validation. The dispatch errors (correction) and _finish_outcome
        # returns None, so the run continues and finishes on the next valid call.
        class Answer(BaseModel):
            value: int

        bad = ToolCall(id="f1", name="finish", arguments={"value": "not-an-int"})
        good = ToolCall(id="f2", name="finish", arguments={"value": 9})
        client = MockLLMClient(
            [
                make_response(content="oops", tool_calls=[bad]),
                make_response(content="ok", tool_calls=[good]),
            ]
        )
        agent = self._agent(client, tools=[], output_schema=Answer)
        result = await agent.run("go")
        assert result.termination_reason == "finished"
        assert result.parsed == Answer(value=9)

    async def test_finish_precedence_over_return_direct(self) -> None:
        @tool(name="rd", description="return direct", return_direct=True)
        async def rd_tool() -> str:
            return "return-direct-output"

        batch = [
            ToolCall(id="rd1", name="rd", arguments={}),
            _finish_call("finish-output", call_id="f1"),
        ]
        client = MockLLMClient([make_response(content="both", tool_calls=batch)])
        agent = self._agent(client, tools=[rd_tool])
        result = await agent.run("go")
        assert result.termination_reason == "finished"
        assert result.output == "finish-output"


# ──────────────────────────────────────────────────────────
# Step 5 — evaluator gate on finish
# ──────────────────────────────────────────────────────────


class _AcceptEval:
    @property
    def max_revisions(self) -> int:
        return 2

    async def evaluate(self, output: str, context: EvaluationContext) -> EvaluationResult:
        return EvaluationResult(verdict=EvaluationVerdict.ACCEPT, evaluator_name="t")


class _ReviseOnceEval:
    def __init__(self) -> None:
        self.calls = 0

    @property
    def max_revisions(self) -> int:
        return 3

    async def evaluate(self, output: str, context: EvaluationContext) -> EvaluationResult:
        self.calls += 1
        if self.calls == 1:
            return EvaluationResult(verdict=EvaluationVerdict.REVISE, feedback="try again", evaluator_name="t")
        return EvaluationResult(verdict=EvaluationVerdict.ACCEPT, evaluator_name="t")


class _RejectEval:
    @property
    def max_revisions(self) -> int:
        return 0

    async def evaluate(self, output: str, context: EvaluationContext) -> EvaluationResult:
        return EvaluationResult(verdict=EvaluationVerdict.REJECT, evaluator_name="t")


class TestFinishEvaluatorGate:
    async def test_accept_terminates_finished(self) -> None:
        client = MockLLMClient([make_response(content="x", tool_calls=[_finish_call("answer")])])
        agent = ReActAgent(
            name="a",
            llm_client=client,
            emitter=make_emitter(),
            system_prompt="Base.",
            tools=[add_tool],
            require_explicit_finish=True,
            output_evaluator=_AcceptEval(),
        )
        result = await agent.run("go")
        assert result.termination_reason == "finished"
        assert result.output == "answer"

    async def test_revise_continues_then_finishes(self) -> None:
        client = MockLLMClient(
            [
                make_response(content="x", tool_calls=[_finish_call("draft", call_id="f1")]),
                make_response(content="y", tool_calls=[_finish_call("final", call_id="f2")]),
            ]
        )
        evaluator = _ReviseOnceEval()
        agent = ReActAgent(
            name="a",
            llm_client=client,
            emitter=make_emitter(),
            system_prompt="Base.",
            tools=[add_tool],
            require_explicit_finish=True,
            output_evaluator=evaluator,
        )
        result = await agent.run("go")
        assert result.termination_reason == "finished"
        assert result.output == "final"
        assert evaluator.calls == 2
        # The revision feedback was injected as a user message.
        assert any("try again" in (m.content or "") for m in result.messages)

    async def test_reject_exhausted_is_evaluation_failed(self) -> None:
        client = MockLLMClient([make_response(content="x", tool_calls=[_finish_call("answer")])])
        agent = ReActAgent(
            name="a",
            llm_client=client,
            emitter=make_emitter(),
            system_prompt="Base.",
            tools=[add_tool],
            require_explicit_finish=True,
            output_evaluator=_RejectEval(),
        )
        result = await agent.run("go")
        assert result.termination_reason == "evaluation_failed"
        assert result.output == "answer"

    async def test_finish_truncation_revises(self) -> None:
        truncated = LLMResponse(
            content="x",
            tool_calls=[_finish_call("partial", call_id="f1")],
            usage=make_usage(),
            model="test",
            stop_reason="max_tokens",
        )
        client = MockLLMClient(
            [
                truncated,
                make_response(content="y", tool_calls=[_finish_call("complete", call_id="f2")]),
            ]
        )
        agent = ReActAgent(
            name="a",
            llm_client=client,
            emitter=make_emitter(),
            system_prompt="Base.",
            tools=[add_tool],
            require_explicit_finish=True,
            output_evaluator=_AcceptEval(),
        )
        result = await agent.run("go")
        assert result.termination_reason == "finished"
        assert result.output == "complete"


# ──────────────────────────────────────────────────────────
# Step 6 — crash-resume short-circuit on a trailing finish batch
# ──────────────────────────────────────────────────────────


class TestCrashResumeFinish:
    def _crash_state(self, finish_args: dict) -> dict:
        messages = [
            Message(role="user", content="go"),
            Message(
                role="assistant",
                content="done",
                tool_calls=[ToolCall(id="f1", name="finish", arguments=finish_args)],
            ),
            Message(role="tool_result", content=json.dumps(finish_args), tool_call_id="f1"),
        ]
        return {
            "agent_type": "react",
            "messages": [m.model_dump() for m in messages],
            "step_number": 1,
            "revision_count": 0,
            "working_memory": None,
            "usages": [make_usage().model_dump()],
            "limiter_count": 1,
            "tool_call_limiter_count": 0,
            "error_handler_state": {"total_corrections": 0},
        }

    async def test_resume_finish_terminates_without_redispatch(self) -> None:
        # Empty response list: if the loop were re-entered and the LLM called,
        # MockLLMClient would raise. Success means the short-circuit fired.
        client = MockLLMClient([])
        agent = ReActAgent(
            name="a",
            llm_client=client,
            emitter=make_emitter(),
            system_prompt="Base.",
            tools=[add_tool],
            require_explicit_finish=True,
        )
        agent._set_resume_state(self._crash_state({"result": "resumed-answer"}))
        result = await agent.run("go")
        assert result.termination_reason == "finished"
        assert result.output == "resumed-answer"
        assert client.calls == []

    async def test_resume_finish_with_schema_reconstructs_parsed(self) -> None:
        class Answer(BaseModel):
            value: int

        client = MockLLMClient([])
        agent = ReActAgent(
            name="a",
            llm_client=client,
            emitter=make_emitter(),
            system_prompt="Base.",
            tools=[],
            output_schema=Answer,
            require_explicit_finish=True,
        )
        agent._set_resume_state(self._crash_state({"value": 7}))
        result = await agent.run("go")
        assert result.termination_reason == "finished"
        assert result.parsed == Answer(value=7)

    async def test_resume_without_trailing_finish_re_enters_loop(self) -> None:
        # Trailing batch is a non-finish tool: no short-circuit, the loop runs
        # and the model then finishes.
        messages = [
            Message(role="user", content="go"),
            Message(
                role="assistant",
                content="adding",
                tool_calls=[ToolCall(id="a1", name="add", arguments={"a": 1, "b": 2})],
            ),
            Message(role="tool_result", content="3", tool_call_id="a1"),
        ]
        state = {
            "agent_type": "react",
            "messages": [m.model_dump() for m in messages],
            "step_number": 1,
            "revision_count": 0,
            "working_memory": None,
            "usages": [make_usage().model_dump()],
            "limiter_count": 1,
            "tool_call_limiter_count": 0,
            "error_handler_state": {"total_corrections": 0},
        }
        client = MockLLMClient([make_response(content="ok", tool_calls=[_finish_call("after-resume")])])
        agent = ReActAgent(
            name="a",
            llm_client=client,
            emitter=make_emitter(),
            system_prompt="Base.",
            tools=[add_tool],
            require_explicit_finish=True,
        )
        agent._set_resume_state(state)
        result = await agent.run("go")
        assert result.termination_reason == "finished"
        assert result.output == "after-resume"
        assert len(client.calls) == 1
