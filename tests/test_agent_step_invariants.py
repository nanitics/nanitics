"""Cross-agent invariant test for ``AgentStepEvent`` field contracts.

Asserts a set of invariants that every SDK agent must satisfy on every emitted
``AgentStepEvent``. Each invariant is the positive expression of an
anti-pattern:

1. ``thought`` is either ``None`` or ``str`` — never a dict or dumped JSON.
2. When the LLM response has a structured ``parsed`` payload, ``thought`` is
   **never** the serialized JSON of that payload. Thought is free-text
   reasoning; structured output lives on ``artifact``.
3. ``artifact`` is either ``None`` or ``dict`` — never a string, never a
   Pydantic model.
4. When ``artifact`` is populated, ``thought`` is either ``None`` or the
   scripted reasoning text — never the artifact serialization.

Tree-search agents (``LATSAgent``, ``TreeOfThoughtAgent``) emit pure marker
steps — ``thought``/``action``/``observation``/``artifact`` are all ``None`` by
design. Those rows exist to guard against a future regression that would start
populating them.

Adding a seventh agent is a one-row change: add a factory function and a
parametrize entry.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from pydantic import BaseModel

from nanitics import (
    CodeActAgent,
    LLMResponse,
    MockLLMClient,
    MockSandbox,
    ReActAgent,
    ReasoningAgent,
    ToolCall,
    tool,
)
from nanitics.capabilities.planning.store import InMemoryPlanStore
from nanitics.infrastructure.observability.events import AgentStepEvent
from nanitics.specialized import (
    ReWOOAgent,
    ReWOOPlan,
    ReWOOStep,
)
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
from tests.testing_helpers import make_emitter, make_usage

# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────────


class _Answer(BaseModel):
    value: int


class _AcceptEvaluator:
    """Evaluator that accepts everything."""

    @property
    def max_revisions(self) -> int:
        return 0

    async def evaluate(self, output: str, context: EvaluationContext) -> EvaluationResult:
        return EvaluationResult(
            verdict=EvaluationVerdict.ACCEPT,
            score=1.0,
            evaluator_name="accept",
        )


@tool(name="noop", description="A no-op tool")
async def _noop_tool(x: str = "") -> str:
    return f"noop({x})"


def _gen_response_json(candidates: list[tuple[str, bool]]) -> str:
    """Build the JSON payload for ``TreeOfThoughtAgent`` / ``LATSAgent`` generation."""
    return _GenerationResponse(
        candidates=[_Candidate(reasoning=r, is_complete=c) for r, c in candidates]
    ).model_dump_json()


async def _run_and_collect_steps(
    factory: Callable[[MockLLMClient], Awaitable[tuple[Any, Any]]],
    responses: list[LLMResponse],
    task: str,
) -> tuple[list[AgentStepEvent], list[LLMResponse]]:
    client = MockLLMClient(responses)
    agent, emitter = await factory(client)
    await agent.run(task)
    steps = [e for e in emitter.events if isinstance(e, AgentStepEvent)]
    return steps, responses


# ──────────────────────────────────────────────────────────────────────────────
# Per-agent factories
#
# Each factory scripts at least one LLMResponse with:
#   - a distinct, non-empty ``content`` string
#   - a distinct, non-empty ``reasoning_text`` string
#   - for structured-output paths, a ``parsed`` pydantic model
#
# The factory returns (agent_instance, emitter) — the caller drives ``.run()``.
#
# Each factory's task is a trivial input; the scripted responses are crafted to
# complete the agent's loop in a single pass so the test stays hermetic.
# ──────────────────────────────────────────────────────────────────────────────


async def _reasoning_factory(
    client: MockLLMClient,
) -> tuple[ReasoningAgent[_Answer], Any]:
    emitter = make_emitter()
    agent = ReasoningAgent(
        name="reasoning",
        llm_client=client,
        emitter=emitter,
        system_prompt="Answer.",
        output_schema=_Answer,
    )
    return agent, emitter


def _reasoning_responses() -> list[LLMResponse]:
    return [
        LLMResponse(
            content='{"value": 42}',  # structured output on content
            tool_calls=[],
            usage=make_usage(),
            model="m",
            stop_reason="end_turn",
            reasoning_text="reasoning about the answer",
            parsed=_Answer(value=42),
        ),
    ]


async def _react_factory(client: MockLLMClient) -> tuple[ReActAgent, Any]:
    emitter = make_emitter()
    agent = ReActAgent(
        name="react",
        llm_client=client,
        emitter=emitter,
        system_prompt="Be helpful.",
        tools=[_noop_tool],
    )
    return agent, emitter


def _react_responses() -> list[LLMResponse]:
    tc = ToolCall(id="tc1", name="noop", arguments={"x": "hi"})
    return [
        LLMResponse(
            content="prose before the tool call",
            tool_calls=[tc],
            usage=make_usage(),
            model="m",
            stop_reason="tool_use",
            reasoning_text="reasoning about the tool",
        ),
        LLMResponse(
            content="final answer",
            tool_calls=[],
            usage=make_usage(),
            model="m",
            stop_reason="end_turn",
            reasoning_text="reasoning about the final answer",
        ),
    ]


async def _codeact_factory(client: MockLLMClient) -> tuple[CodeActAgent, Any]:
    emitter = make_emitter()
    sandbox = MockSandbox([])  # Direct answer path — no code execution.
    agent = CodeActAgent(
        name="codeact",
        llm_client=client,
        emitter=emitter,
        system_prompt="Be helpful.",
        sandbox=sandbox,
    )
    return agent, emitter


def _codeact_responses() -> list[LLMResponse]:
    return [
        LLMResponse(
            content="the answer is 42",
            tool_calls=[],
            usage=make_usage(),
            model="m",
            stop_reason="end_turn",
            reasoning_text="reasoning about the direct answer",
        ),
    ]


async def _rewoo_factory(client: MockLLMClient) -> tuple[ReWOOAgent, Any]:
    emitter = make_emitter()
    agent = ReWOOAgent(
        name="rewoo",
        llm_client=client,
        emitter=emitter,
        system_prompt="Plan and solve.",
        tools=[_noop_tool],
        plan_store=InMemoryPlanStore(),
    )
    return agent, emitter


def _rewoo_responses() -> list[LLMResponse]:
    plan_json = ReWOOPlan(
        steps=[
            ReWOOStep(
                step_number=1,
                description="noop call",
                tool_name="noop",
                arguments={"x": "hi"},
                depends_on=[],
            ),
        ]
    ).model_dump_json()
    planner = LLMResponse(
        content=plan_json,
        tool_calls=[],
        usage=make_usage(),
        model="m",
        stop_reason="end_turn",
        reasoning_text="reasoning about the plan",
        parsed=ReWOOPlan.model_validate_json(plan_json),
    )
    solver = LLMResponse(
        content="final rewoo answer",
        tool_calls=[],
        usage=make_usage(),
        model="m",
        stop_reason="end_turn",
        reasoning_text="reasoning about the solver answer",
    )
    return [planner, solver]


async def _lats_factory(client: MockLLMClient) -> tuple[LATSAgent, Any]:
    emitter = make_emitter()
    agent = LATSAgent(
        name="lats",
        llm_client=client,
        emitter=emitter,
        system_prompt="Search.",
        tools=[_noop_tool],
        node_evaluator=_AcceptEvaluator(),
        max_iterations=1,
        branching_factor=1,
        max_depth=1,
    )
    return agent, emitter


def _lats_responses() -> list[LLMResponse]:
    gen = _gen_response_json([("terminal thought", True)])
    return [
        LLMResponse(
            content=gen,
            tool_calls=[],
            usage=make_usage(),
            model="m",
            stop_reason="end_turn",
            reasoning_text="reasoning about the candidate",
        ),
    ]


async def _tot_factory(client: MockLLMClient) -> tuple[TreeOfThoughtAgent, Any]:
    emitter = make_emitter()
    agent = TreeOfThoughtAgent(
        name="tot",
        llm_client=client,
        emitter=emitter,
        system_prompt="Think.",
        node_evaluator=_AcceptEvaluator(),
        search_strategy=SearchStrategy.BFS,
        branching_factor=1,
        max_depth=1,
        max_nodes=5,
    )
    return agent, emitter


def _tot_responses() -> list[LLMResponse]:
    gen = _gen_response_json([("terminal thought", True)])
    return [
        LLMResponse(
            content=gen,
            tool_calls=[],
            usage=make_usage(),
            model="m",
            stop_reason="end_turn",
            reasoning_text="reasoning about the candidate",
        ),
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Parametrized invariant test
# ──────────────────────────────────────────────────────────────────────────────


_CASES: list[tuple[str, Callable[..., Any], Callable[[], list[LLMResponse]], str, bool]] = [
    # (agent_id, factory, responses_builder, task, is_marker_only)
    ("ReasoningAgent", _reasoning_factory, _reasoning_responses, "What is the value?", False),
    ("ReActAgent", _react_factory, _react_responses, "Call noop and answer.", False),
    ("CodeActAgent", _codeact_factory, _codeact_responses, "What is the answer?", False),
    ("ReWOOAgent", _rewoo_factory, _rewoo_responses, "Plan and solve.", False),
    ("LATSAgent", _lats_factory, _lats_responses, "Find a thought.", True),
    ("TreeOfThoughtAgent", _tot_factory, _tot_responses, "Think.", True),
]


@pytest.mark.parametrize(
    ("agent_id", "factory", "responses_builder", "task", "is_marker_only"),
    _CASES,
    ids=[c[0] for c in _CASES],
)
async def test_agent_step_event_invariants(
    agent_id: str,
    factory: Callable[[MockLLMClient], Awaitable[tuple[Any, Any]]],
    responses_builder: Callable[[], list[LLMResponse]],
    task: str,
    is_marker_only: bool,
) -> None:
    """For every emitted ``AgentStepEvent``, all four invariants must hold.

    See the module docstring for the invariants.
    """
    responses = responses_builder()
    step_events, scripted = await _run_and_collect_steps(factory, responses, task)

    assert step_events, f"{agent_id}: expected at least one AgentStepEvent, got zero"

    # Build the set of scripted reasoning texts and scripted content-when-parsed
    # strings for structured-output invariants.
    scripted_reasoning = {r.reasoning_text for r in scripted if r.reasoning_text is not None}
    structured_contents = {r.content for r in scripted if r.parsed is not None and r.content}
    structured_dumps_json = {
        json.dumps(r.parsed.model_dump(), sort_keys=True) if hasattr(r.parsed, "model_dump") else None
        for r in scripted
        if r.parsed is not None
    } - {None}

    for step_event in step_events:
        # Invariant 1: thought is None or str.
        assert step_event.thought is None or isinstance(step_event.thought, str), (
            f"{agent_id}: thought must be None or str, got {type(step_event.thought).__name__}"
        )

        # Invariant 3: artifact is None or dict.
        assert step_event.artifact is None or isinstance(step_event.artifact, dict), (
            f"{agent_id}: artifact must be None or dict, got {type(step_event.artifact).__name__}"
        )

        # Invariant 2: thought never equals the scripted structured-output
        # content (the JSON string) when the response had a parsed payload.
        if step_event.thought is not None:
            assert step_event.thought not in structured_contents, (
                f"{agent_id}: thought must not equal scripted structured-output content; "
                f"got thought={step_event.thought!r}"
            )
            # And never equals the JSON serialization of the artifact dict.
            # If thought is not valid JSON, that is fine — it is free-text.
            thought_sorted = None
            with contextlib.suppress(ValueError, TypeError):
                thought_sorted = json.dumps(json.loads(step_event.thought), sort_keys=True)
            if thought_sorted is not None:
                assert thought_sorted not in structured_dumps_json, (
                    f"{agent_id}: thought must not equal the JSON dump of the artifact"
                )

        # Invariant 4: if artifact is populated, thought is either None or the
        # scripted reasoning text — never the artifact serialization.
        if step_event.artifact is not None:
            if step_event.thought is not None:
                assert step_event.thought in scripted_reasoning, (
                    f"{agent_id}: when artifact is set, thought must be the scripted "
                    f"reasoning_text or None; got thought={step_event.thought!r}"
                )

        # Marker-only invariant: LATS/ToT step events are pure markers.
        if is_marker_only:
            assert step_event.thought is None, (
                f"{agent_id}: marker-only agent must not populate thought; got {step_event.thought!r}"
            )
            assert step_event.action is None, (
                f"{agent_id}: marker-only agent must not populate action; got {step_event.action!r}"
            )
            assert step_event.observation is None, (
                f"{agent_id}: marker-only agent must not populate observation; got {step_event.observation!r}"
            )
            assert step_event.artifact is None, (
                f"{agent_id}: marker-only agent must not populate artifact; got {step_event.artifact!r}"
            )
