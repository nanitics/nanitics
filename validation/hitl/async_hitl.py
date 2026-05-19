"""AsyncHumanInputProvider concurrency: two simultaneous ``ask_human`` calls.

Two real ``ReActAgent`` instances run concurrently under ``asyncio.gather``;
each is configured to ask the human a distinct question via ``ask_human``
tools bound to a shared ``AsyncHumanInputProvider``. A separate resolver
coroutine drains ``provider.get_pending()`` and answers each question with
a deterministic, question-specific reply. The resolver deliberately
answers the second-observed request first, so if the provider mis-routed
responses to futures by order rather than by ``request_id`` the replies
would reach the wrong agent.

The subjects of the test are (a) that the two agents' LLM loops ran
concurrently end-to-end (their step-1 LLM request/response intervals
overlap in wall-clock time), (b) that both agents received their own
matching answer (``request_id`` round-trip is correct), and (c) the HITL
event pair shape.

The deterministic proof that two ``ask_human`` suspensions can coexist on
``AsyncHumanInputProvider._pending`` belongs to ``tests/test_async_provider.py``
(``TestConcurrentRequests::test_two_requests_are_pending_simultaneously``) —
that property is not observable at the validation layer without racing
response-arrival skew from the live LLM, so it is asserted at the unit
layer where the event loop is controllable.

Acceptance criteria:
  - Both agents terminate with ``termination_reason == "complete"``.
  - The two agents' step-1 ``LLMRequestEvent``/``LLMResponseEvent``
    intervals overlap in time — they were dispatched concurrently, not
    serialized.
  - The resolver answered exactly two requests and every resolve call
    returned True (no stale or duplicate request_ids).
  - Trace contains exactly two ``HumanInputRequestEvent`` with
    ``request_type == "question"`` and exactly two
    ``HumanInputResponseEvent`` with ``decision == "answer"``.
  - Each of the two ``request_id`` values appears exactly once in request
    events and exactly once in response events (1:1 pairing, no crosstalk).
  - After resolution ``provider.get_pending() == []``.
  - Each agent's final output contains the answer scripted for its own
    question and does NOT contain the answer scripted for the other
    agent's question — direct evidence responses routed by ``request_id``.
"""

from __future__ import annotations

import asyncio
from collections import Counter

import pytest

from nanitics.hitl import (
    AsyncHumanInputProvider,
    HumanDecision,
    HumanInputResponse,
    create_ask_human_tool,
)
from nanitics.infrastructure import (
    AgentStartEvent,
    HumanInputRequestEvent,
    HumanInputResponseEvent,
    LLMRequestEvent,
    LLMResponseEvent,
)
from nanitics.strategies import ReActAgent
from nanitics.tracing import InMemoryEmitter
from validation.helpers import (
    assert_trace_contains,
    make_llm_client,
    run_with_retry,
)

_CAPITAL_ANSWER = "The capital is Canberra."
_CURRENCY_ANSWER = "The currency is the Japanese yen."

_QUESTION_ANSWERS: dict[str, str] = {
    "capital": _CAPITAL_ANSWER,
    "currency": _CURRENCY_ANSWER,
}


def _classify_question(prompt: str) -> str:
    lowered = prompt.lower()
    if "capital" in lowered:
        return "capital"
    if "currency" in lowered or "yen" in lowered:
        return "currency"
    raise AssertionError(f"Unclassifiable question prompt: {prompt!r}")


@pytest.mark.quick
async def test_async_provider_concurrent_requests(
    traced_emitter: InMemoryEmitter,
) -> None:
    client = make_llm_client("anthropic")
    provider = AsyncHumanInputProvider()

    capital_agent = ReActAgent(
        name="capital-agent",
        llm_client=client,
        emitter=traced_emitter,
        system_prompt=(
            "You are a geography assistant. You do NOT know the answer on "
            "your own — you MUST call ``ask_human`` with a question about "
            "the capital of Australia, then repeat the human's exact reply "
            "in your final sentence."
        ),
        tools=[create_ask_human_tool(provider)],
        tool_state={"run_id": "validation-94-capital", "agent_name": "capital-agent"},
        max_iterations=3,
    )

    currency_agent = ReActAgent(
        name="currency-agent",
        llm_client=client,
        emitter=traced_emitter,
        system_prompt=(
            "You are a finance assistant. You do NOT know the answer on "
            "your own — you MUST call ``ask_human`` with a question about "
            "the currency of Japan, then repeat the human's exact reply in "
            "your final sentence."
        ),
        tools=[create_ask_human_tool(provider)],
        tool_state={"run_id": "validation-94-currency", "agent_name": "currency-agent"},
        max_iterations=3,
    )

    resolver_stats = {
        "resolved": 0,
        "successes": 0,
        "resolved_ids": [],
    }
    agents_done = asyncio.Event()

    async def resolver() -> None:
        while resolver_stats["resolved"] < 2:
            pending = provider.get_pending()
            # Reverse-order resolution: if 2 are pending, answer the later one
            # first so per-request_id routing is exercised, not FIFO order.
            if len(pending) >= 2:
                target = pending[-1]
            elif len(pending) == 1:
                target = pending[0]
            else:
                if agents_done.is_set():
                    return
                await asyncio.sleep(0)
                continue

            kind = _classify_question(target.prompt)
            ok = await provider.resolve(
                target.request_id,
                HumanInputResponse(
                    request_id=target.request_id,
                    decision=HumanDecision.ANSWER,
                    content=_QUESTION_ANSWERS[kind],
                ),
            )
            resolver_stats["resolved"] += 1
            resolver_stats["resolved_ids"].append(target.request_id)
            if ok:
                resolver_stats["successes"] += 1

    async def run_both() -> tuple[object, object]:
        resolver_task = asyncio.create_task(resolver())
        try:
            results = await asyncio.gather(
                capital_agent.run("What is the capital of Australia?"),
                currency_agent.run("What is the currency of Japan?"),
            )
        finally:
            agents_done.set()
            await resolver_task
        return results[0], results[1]

    capital_result, currency_result = await run_with_retry(run_both, max_attempts=2)

    # --- Termination ---
    assert capital_result.termination_reason == "complete", (
        f"capital-agent termination_reason={capital_result.termination_reason!r}; expected 'complete'."
    )
    assert currency_result.termination_reason == "complete", (
        f"currency-agent termination_reason={currency_result.termination_reason!r}; expected 'complete'."
    )

    # --- Concurrency evidence ---
    # Trace-based concurrent-dispatch proof: the two agents' step-1
    # ``llm.request``/``llm.response`` intervals must overlap in wall-clock
    # time. Overlap proves the agent loops were scheduled concurrently — one
    # agent's step-1 LLM call was still in flight when the other agent had
    # already dispatched its own step-1 LLM call. This assertion does NOT
    # attempt to prove that the two ``ask_human`` suspensions coexisted on
    # ``AsyncHumanInputProvider._pending`` at some instant; that stronger
    # property depends on response-arrival skew from the live LLM and is
    # proved deterministically at the unit layer in
    # ``tests/test_async_provider.py::TestConcurrentRequests::test_two_requests_are_pending_simultaneously``.
    agent_starts = {
        e.span_id: e.agent_name
        for e in traced_emitter.events
        if isinstance(e, AgentStartEvent) and e.agent_name in {"capital-agent", "currency-agent"}
    }

    def _agent_for_llm_event(event: LLMRequestEvent | LLMResponseEvent) -> str | None:
        # Step-1 LLM events live inside a step span whose ``parent_span_id``
        # is the agent span (confirmed via trace inspection). For step-1 the
        # LLM event's ``parent_span_id`` itself is the agent's ``span_id``.
        return agent_starts.get(event.parent_span_id)

    per_agent_first_request: dict[str, LLMRequestEvent] = {}
    per_agent_first_response: dict[str, LLMResponseEvent] = {}
    for event in traced_emitter.events:
        if isinstance(event, LLMRequestEvent):
            agent = _agent_for_llm_event(event)
            if agent is not None and agent not in per_agent_first_request:
                per_agent_first_request[agent] = event
        elif isinstance(event, LLMResponseEvent):
            agent = _agent_for_llm_event(event)
            if agent is not None and agent not in per_agent_first_response:
                per_agent_first_response[agent] = event

    expected_agents = {"capital-agent", "currency-agent"}
    assert set(per_agent_first_request) == expected_agents, (
        f"Expected step-1 LLMRequestEvent for each of {expected_agents}; observed {set(per_agent_first_request)}."
    )
    assert set(per_agent_first_response) == expected_agents, (
        f"Expected step-1 LLMResponseEvent for each of {expected_agents}; observed {set(per_agent_first_response)}."
    )

    cap_req_ts = per_agent_first_request["capital-agent"].timestamp
    cap_resp_ts = per_agent_first_response["capital-agent"].timestamp
    cur_req_ts = per_agent_first_request["currency-agent"].timestamp
    cur_resp_ts = per_agent_first_response["currency-agent"].timestamp
    overlap_start = max(cap_req_ts, cur_req_ts)
    overlap_end = min(cap_resp_ts, cur_resp_ts)
    assert overlap_start < overlap_end, (
        "Expected the two agents' step-1 LLM request/response intervals to "
        "overlap (concurrent dispatch); observed no overlap. "
        f"capital-agent=[{cap_req_ts.isoformat()}, {cap_resp_ts.isoformat()}]; "
        f"currency-agent=[{cur_req_ts.isoformat()}, {cur_resp_ts.isoformat()}]."
    )

    assert resolver_stats["resolved"] == 2, (
        f"Expected resolver to answer exactly 2 requests; got {resolver_stats['resolved']}."
    )
    assert resolver_stats["successes"] == 2, (
        f"Every resolve() call must return True; "
        f"got {resolver_stats['successes']} success(es) of "
        f"{resolver_stats['resolved']} attempt(s)."
    )
    assert provider.get_pending() == [], (
        f"After resolution no requests should remain pending; got {provider.get_pending()!r}."
    )

    # --- Trace-shape invariants ---
    question_requests = [
        e for e in traced_emitter.events if isinstance(e, HumanInputRequestEvent) and e.request_type == "question"
    ]
    answer_responses = [
        e for e in traced_emitter.events if isinstance(e, HumanInputResponseEvent) and e.decision == "answer"
    ]
    assert len(question_requests) == 2, (
        f"Expected exactly two question HumanInputRequestEvent; got {len(question_requests)}."
    )
    assert len(answer_responses) == 2, (
        f"Expected exactly two answer HumanInputResponseEvent; got {len(answer_responses)}."
    )

    request_id_counts = Counter(e.request_id for e in question_requests)
    response_id_counts = Counter(e.request_id for e in answer_responses)
    assert set(request_id_counts) == set(response_id_counts), (
        "Question request_ids and answer response_ids must align 1:1; "
        f"requests={set(request_id_counts)}, responses={set(response_id_counts)}."
    )
    assert all(c == 1 for c in request_id_counts.values()), (
        f"Each request_id must appear exactly once among requests; got counts={request_id_counts!r}."
    )
    assert all(c == 1 for c in response_id_counts.values()), (
        f"Each request_id must appear exactly once among responses; got counts={response_id_counts!r}."
    )

    # Pin that at least one response event exists for each request (ordering-
    # independent pairing confirmation via assert_trace_contains).
    for q_event in question_requests:
        assert_trace_contains(
            traced_emitter,
            HumanInputResponseEvent,
            predicate=lambda e, rid=q_event.request_id: e.request_id == rid and e.decision == "answer",
        )

    # --- Per-agent routing: each agent sees only its own answer ---
    capital_output = capital_result.output or ""
    currency_output = currency_result.output or ""
    assert "Canberra" in capital_output, (
        f"capital-agent must receive the Canberra answer; got output={capital_output!r}."
    )
    assert "yen" not in capital_output.lower(), (
        f"capital-agent must NOT receive the currency answer — responses crossed wires; got output={capital_output!r}."
    )
    assert "yen" in currency_output.lower(), (
        f"currency-agent must receive the yen answer; got output={currency_output!r}."
    )
    assert "canberra" not in currency_output.lower(), (
        f"currency-agent must NOT receive the capital answer — responses crossed wires; got output={currency_output!r}."
    )
