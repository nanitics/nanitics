"""HITL tools validation: agent-initiated ``ask_human`` with a real LLM.

A real ``ReActAgent`` is given only the tools returned by
``create_hitl_tools`` and is handed a task it cannot solve without asking
the human for a specific fact. A deterministic provider answers the
agent's question with a pre-scripted value; the agent must observe the
human's reply and weave it into its final answer.

The subjects of the test are (a) that the agent chose to invoke
``ask_human`` (not ``request_approval``), (b) that the deterministic reply
travelled back through the tool into the agent's reasoning, and (c) the
``HumanInputRequestEvent`` / ``HumanInputResponseEvent`` pair.

Acceptance criteria:
  - ``create_hitl_tools(provider)`` returns exactly two FunctionTools whose
    schema names are ``{"request_approval", "ask_human"}``.
  - Trace contains at least one ``HumanInputRequestEvent`` whose
    ``request_type == "question"``, ``agent_name == "planner"`` (the
    agent whose tool_state supplied the name), and whose
    ``metadata["question"]`` is a non-empty string; the first such event's
    ``request_id`` matches a ``HumanInputResponseEvent`` whose
    ``decision == "answer"`` and ``has_content is True``.
  - Trace contains zero ``HumanInputRequestEvent`` with
    ``request_type == "approval"`` — the agent chose ``ask_human`` over
    ``request_approval``.
  - The agent's final answer contains the load-bearing fact from the
    scripted human reply (``eu-west-2``), proving the tool's return value
    was fed back into the reasoning loop. Verbatim echo is not required —
    the model may paraphrase as long as the fact reaches the output.
  - ``result.termination_reason == "complete"``.
"""

from __future__ import annotations

import pytest

from nanitics import (
    CallbackHumanInputProvider,
    HumanDecision,
    HumanInputResponse,
    InMemoryEmitter,
    ReActAgent,
    create_hitl_tools,
)
from nanitics.collaboration.protocol import HumanInputRequest
from nanitics.infrastructure import (
    HumanInputRequestEvent,
    HumanInputResponseEvent,
)
from validation.helpers import (
    assert_trace_contains,
    make_llm_client,
    run_with_retry,
)

_HUMAN_ANSWER = "The preferred deployment region is eu-west-2."


@pytest.mark.quick
async def test_ask_human_roundtrip(traced_emitter: InMemoryEmitter) -> None:
    client = make_llm_client("anthropic")

    def _callback(req: HumanInputRequest) -> HumanInputResponse:
        return HumanInputResponse(
            request_id=req.request_id,
            decision=HumanDecision.ANSWER,
            content=_HUMAN_ANSWER,
        )

    provider = CallbackHumanInputProvider(_callback)
    hitl_tools = create_hitl_tools(provider)

    tool_names = {t.schema.name for t in hitl_tools}
    assert tool_names == {"request_approval", "ask_human"}, (
        f"Expected {{request_approval, ask_human}}; got {tool_names}."
    )

    agent = ReActAgent(
        name="planner",
        llm_client=client,
        emitter=traced_emitter,
        system_prompt=(
            "You are a deployment-planning assistant. You do NOT know the "
            "user's deployment region — you must ask the human using the "
            "``ask_human`` tool before answering. Do not call "
            "``request_approval`` for this task. Once the human answers, "
            "repeat their exact answer in your final reply so the user can "
            "confirm you understood."
        ),
        tools=hitl_tools,
        tool_state={"run_id": "validation-93-planner", "agent_name": "planner"},
        max_iterations=4,
    )

    result = await run_with_retry(
        lambda: agent.run(
            "Tell me which region my application should be deployed to. You do not know the answer — ask the human."
        ),
        max_attempts=2,
    )

    # --- HITL event invariants ---
    question_requests = [
        e for e in traced_emitter.events if isinstance(e, HumanInputRequestEvent) and e.request_type == "question"
    ]
    assert len(question_requests) >= 1, (
        "Expected at least one HumanInputRequestEvent with request_type=='question'; got zero."
    )

    approval_requests = [
        e for e in traced_emitter.events if isinstance(e, HumanInputRequestEvent) and e.request_type == "approval"
    ]
    assert approval_requests == [], (
        "Agent chose request_approval when it was told to use ask_human; "
        f"got {len(approval_requests)} approval request event(s)."
    )

    first_question = question_requests[0]
    # --- Trace-level shape of the request event: agent_name and metadata
    # land on the event itself (no provider-wrapper workaround needed). ---
    assert first_question.agent_name == "planner", (
        "Expected the request event's agent_name to reflect the ambient "
        f"tool_state ('planner'); got {first_question.agent_name!r}."
    )
    event_question = first_question.metadata.get("question")
    assert isinstance(event_question, str), (
        f"Expected event metadata['question'] to be a string; got {first_question.metadata!r}."
    )
    assert event_question.strip(), (
        f"Expected event metadata['question'] to be non-empty; got {first_question.metadata!r}."
    )
    answer_response = assert_trace_contains(
        traced_emitter,
        HumanInputResponseEvent,
        predicate=lambda e: (
            e.request_id == first_question.request_id and e.decision == "answer" and e.has_content is True
        ),
    )
    assert answer_response.wait_duration_ms >= 0, (
        f"wait_duration_ms must be non-negative; got {answer_response.wait_duration_ms}."
    )

    # --- End-to-end: the load-bearing fact from the scripted reply reached the output ---
    assert "eu-west-2" in (result.output or "").lower(), (
        "Expected the load-bearing fact from the scripted human reply "
        f"('eu-west-2') to appear in the agent's final output; got output={result.output!r}."
    )
    assert result.termination_reason == "complete", (
        f"Expected termination_reason=='complete'; got {result.termination_reason!r}."
    )
