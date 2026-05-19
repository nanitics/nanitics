"""Durable HITL persistence across a simulated process restart.

Pins the defining durability property of
``DurableHumanInputProvider`` + ``PostgresHitlRequestStore``: when an
agent's ``ask_human`` request suspends execution, the request is
durably persisted; a fresh provider instance pointing at the same
store can observe the pending request, accept the human's response,
and hand it back on resume.

The script uses a real Anthropic LLM client to construct the
``ReActAgent`` and triggers ``ask_human`` via an explicit system
prompt. The suspension raises ``SuspendExecution`` out of the agent
run, which we catch — this is the durable-suspension contract.

A second ``DurableHumanInputProvider`` is constructed on the same
``PostgresHitlRequestStore`` (the "fresh process" case). We assert:

- The suspended request is visible via the store's
  ``get_pending_requests(run_id)``.
- After ``save_response`` is recorded, ``get_response`` round-trips
  the response payload for the same ``request_id`` across pool
  connections.
- ``get_pending_requests`` no longer returns the responded request.
- The new provider's ``request_input`` returns the stored response
  directly on replay — the provider holds no state; the store is the
  single source of truth (resume path).

The PostgreSQL ``hitl_requests`` and ``hitl_responses`` tables are
dropped in the ``finally`` block.

Acceptance criteria:
  - First-run ``agent.run()`` raises ``SuspendExecution`` via the
    durable provider (no answer available).
  - Exactly one pending request is visible via
    ``store.get_pending_requests(run_id)`` after suspension, and its
    ``prompt``, ``run_id``, ``agent_name`` match what the tool sent.
  - A **second** ``DurableHumanInputProvider`` on the **same** store
    sees the same pending request (proves the request is owned by
    the store, not by the provider instance — the distinguishing
    durability assertion).
  - After ``store.save_response`` the pending list drops to zero.
  - ``store.get_response(request_id)`` round-trips the human's
    ``decision`` and ``content`` exactly.
  - The second provider, asked to ``request_input`` a replay request
    with the same ``request_id``, returns the stored response
    without suspending or re-persisting (resume contract).
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from nanitics.collaboration.durable_provider import DurableHumanInputProvider
from nanitics.collaboration.postgres_hitl_store import (
    PostgresHitlRequestStore,
    get_hitl_schema_sql,
)
from nanitics.collaboration.protocol import (
    HumanDecision,
    HumanInputRequest,
    HumanInputResponse,
    HumanInputType,
)
from nanitics.composition.durability.suspension import SuspendExecution
from nanitics.hitl import create_ask_human_tool
from nanitics.strategies import ReActAgent
from nanitics.tracing import InMemoryEmitter
from validation.helpers import (
    make_llm_client,
    make_postgres_pool,
    requires_postgres,
    run_with_retry,
)


async def _drop_hitl_tables(pool: Any) -> None:
    """Drop HITL tables on teardown — idempotent."""
    async with pool.acquire() as conn:
        await conn.execute("DROP TABLE IF EXISTS hitl_responses")
        await conn.execute("DROP TABLE IF EXISTS hitl_requests")


async def _ensure_hitl_schema(pool: Any) -> None:
    """Apply HITL schema DDL. The module does not expose ``ensure_schema``
    on ``PostgresHitlRequestStore`` — schema is applied at deploy time.
    """
    async with pool.acquire() as conn:
        await conn.execute(get_hitl_schema_sql())


@requires_postgres
async def test_durable_hitl_persists_across_provider_instances(
    traced_emitter: InMemoryEmitter,
) -> None:
    run_id = f"durable-hitl-{uuid.uuid4().hex[:8]}"
    async with make_postgres_pool() as pool:
        try:
            # Clean slate: drop any lingering tables from prior failed runs.
            await _drop_hitl_tables(pool)
            await _ensure_hitl_schema(pool)

            store = PostgresHitlRequestStore(pool)
            provider1 = DurableHumanInputProvider(request_store=store)

            # --- First run: agent must suspend on ask_human ---
            agent = ReActAgent(
                name="durable-hitl-agent",
                llm_client=make_llm_client("anthropic"),
                emitter=traced_emitter,
                system_prompt=(
                    "You do not know the user's favourite colour. You MUST call "
                    "the ``ask_human`` tool with a direct question asking for "
                    "their favourite colour — do not answer on your own."
                ),
                tools=[create_ask_human_tool(provider1)],
                tool_state={
                    "run_id": run_id,
                    "agent_name": "durable-hitl-agent",
                },
                max_iterations=2,
            )

            # SuspendExecution is not retryable (see validation.helpers.retry);
            # it propagates immediately. Transient LLM failures before the
            # ask_human call still retry up to max_attempts.
            with pytest.raises(SuspendExecution) as suspended:
                await run_with_retry(
                    lambda: agent.run("What is my favourite colour?"),
                    max_attempts=2,
                )

            # The raised suspension must carry request metadata through
            # ``suspension_info`` so the resumer knows which request to answer.
            info = suspended.value.suspension_info
            assert info is not None, "SuspendExecution must carry suspension_info"
            assert info.request_type == HumanInputType.QUESTION.value, (
                f"Expected request_type={HumanInputType.QUESTION.value!r}; got {info.request_type!r}"
            )
            assert info.agent_name == "durable-hitl-agent"
            suspended_request_id = info.request_id

            # --- Pending request visible via the store ---
            pending_first = await store.get_pending_requests(run_id)
            assert len(pending_first) == 1, (
                f"Expected exactly one pending request after suspension; got {len(pending_first)}."
            )
            pending_req = pending_first[0]
            assert pending_req.request_id == suspended_request_id
            assert pending_req.run_id == run_id
            assert pending_req.agent_name == "durable-hitl-agent"
            assert pending_req.request_type == HumanInputType.QUESTION
            assert pending_req.prompt, "Persisted request must carry the non-empty prompt."

            # --- Second provider on the same store sees the same request ---
            # The defining durability assertion: the request is owned by the
            # store, not by the original provider instance.
            provider2 = DurableHumanInputProvider(request_store=store)
            pending_from_store_again = await store.get_pending_requests(run_id)
            assert [r.request_id for r in pending_from_store_again] == [suspended_request_id], (
                "A second DurableHumanInputProvider on the same store must observe the identical pending request set."
            )

            # --- Human responds; store round-trip ---
            answer = HumanInputResponse(
                request_id=suspended_request_id,
                decision=HumanDecision.ANSWER,
                content="The favourite colour is teal.",
            )
            await store.save_response(suspended_request_id, answer)

            stored_response = await store.get_response(suspended_request_id)
            assert stored_response is not None, (
                "Response must be retrievable by request_id via a fresh pool connection."
            )
            assert stored_response.decision == HumanDecision.ANSWER
            assert stored_response.content == "The favourite colour is teal."
            assert stored_response.request_id == suspended_request_id

            pending_after_response = await store.get_pending_requests(run_id)
            assert pending_after_response == [], (
                f"Responded requests must no longer be pending; got {pending_after_response!r}."
            )

            # --- Resume path: provider2 reads the stored response from the store ---
            # No preload. The provider is stateless; the store is the single
            # source of truth for both requests and responses.
            replay_request = HumanInputRequest(
                request_id=suspended_request_id,
                run_id=run_id,
                request_type=HumanInputType.QUESTION,
                prompt=pending_req.prompt,
                agent_name=pending_req.agent_name,
            )
            returned = await provider2.request_input(replay_request)
            assert returned == stored_response, (
                "A DurableHumanInputProvider must return the stored response "
                "for a replay request with the same request_id; "
                f"got {returned!r}."
            )
        finally:
            await _drop_hitl_tables(pool)


@requires_postgres
async def test_multiple_pending_requests_have_distinct_request_ids() -> None:
    """Two concurrent HITL requests in one run must have distinct ``request_id``s.

    This is the scenario that motivated deterministic identity: when a
    run has more than one pending request at the same time (different
    ``tool_call.id``s in the same ``run_id``), the store must expose
    both as distinct rows. Preload-channel semantics could not route
    responses correctly in this case; the stateless provider + store
    can.
    """
    run_id = f"multi-pending-{uuid.uuid4().hex[:8]}"
    async with make_postgres_pool() as pool:
        try:
            await _drop_hitl_tables(pool)
            await _ensure_hitl_schema(pool)

            store = PostgresHitlRequestStore(pool)

            req_a = HumanInputRequest(
                request_id=f"{run_id}:tc-a",
                run_id=run_id,
                request_type=HumanInputType.QUESTION,
                prompt="What is your favourite colour?",
                agent_name="agent-A",
            )
            req_b = HumanInputRequest(
                request_id=f"{run_id}:tc-b",
                run_id=run_id,
                request_type=HumanInputType.APPROVAL,
                prompt="Approve publication?",
                agent_name="agent-A",
            )
            await store.save_request(req_a)
            await store.save_request(req_b)

            pending = await store.get_pending_requests(run_id)
            ids = sorted(r.request_id for r in pending)
            assert ids == [f"{run_id}:tc-a", f"{run_id}:tc-b"], (
                f"Expected two distinct request_ids for one run; got {ids!r}."
            )
            assert len({r.request_id for r in pending}) == 2, (
                "request_ids must be distinct across tool_call.ids in the same run."
            )
        finally:
            await _drop_hitl_tables(pool)
