"""Durable resume service end-to-end against real Postgres.

Pins the ``DurableRun`` + ``ResumeService`` contract: a ``ReActAgent``
whose tools force an ``ask_human`` call suspends through ``DurableRun``
to a ``SuspendedRun`` handle; the human's ``HumanInputResponse`` is
routed through ``ResumeService.resume(...)``, which reconstructs the
agent via a factory closure and drives it to completion over the
stored response. Storage is a real ``PostgresHitlRequestStore``
(pgvector/pgvector:pg16 testcontainer, provisioned by
``validation/conftest.py``); checkpoints use
``InMemoryCheckpointStore`` — there is no Postgres checkpoint store
today, and this script's contract is about the HITL-durability round
trip, not checkpoint persistence. The script notes this degradation
explicitly.

Sibling to ``validation/durability/durable_hitl.py``: that script
proves the **stateless-provider / store-as-source-of-truth** property
of ``DurableHumanInputProvider`` across two independent provider
instances. This script proves the **round-trip over the public
service API**: suspend → save response → resume → final output, with
no hand-rolled ``_set_resume_state`` glue.

The PostgreSQL ``hitl_requests`` and ``hitl_responses`` tables are
dropped in the ``finally`` block.

Acceptance criteria:
  - First-run ``DurableRun.start(...)`` returns a ``SuspendedRun``
    with a ``pending_request`` whose ``request_id`` matches
    ``suspension_info.request_id``.
  - The pending request is observable via
    ``store.get_pending_requests(run_id)`` (durable).
  - ``ResumeService.resume(run_id, response)`` completes the run and
    returns a ``ResumeResult`` whose ``output`` is a non-empty string
    (the agent produced a final answer over the stored human
    response).
  - After resume, the responded request is no longer pending.
"""

from __future__ import annotations

import uuid
from typing import Any

from nanitics.collaboration.durable_provider import DurableHumanInputProvider
from nanitics.collaboration.postgres_hitl_store import (
    PostgresHitlRequestStore,
    get_hitl_schema_sql,
)
from nanitics.collaboration.protocol import (
    HumanDecision,
    HumanInputResponse,
)
from nanitics.composition import (
    AgentStep,
    CheckpointStore,
    DurableRun,
    InMemoryCheckpointStore,
    ResumeContext,
    ResumeResult,
    ResumeService,
    Sequential,
    SuspendedRun,
)
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
    """Apply HITL schema DDL."""
    async with pool.acquire() as conn:
        await conn.execute(get_hitl_schema_sql())


def _build_agent_runnable(
    *,
    hitl_store: PostgresHitlRequestStore,
    checkpoint_store: CheckpointStore,
    emitter: InMemoryEmitter,
    run_id: str,
) -> Sequential:
    """Construct the ReAct-agent-in-Sequential runnable.

    The same builder runs both for the initial ``DurableRun`` and
    inside the ``ResumeService`` factory — deterministic reconstruction
    is the production pattern.
    """
    provider = DurableHumanInputProvider(request_store=hitl_store)
    agent = ReActAgent(
        name="durable-resume-service-agent",
        llm_client=make_llm_client("anthropic"),
        emitter=emitter,
        system_prompt=(
            "You do not know the user's favourite colour. You MUST call "
            "the ``ask_human`` tool with a direct question asking for "
            "their favourite colour — do not answer on your own. After "
            "the human answers, produce a one-sentence final answer that "
            "uses their stated colour."
        ),
        tools=[create_ask_human_tool(provider)],
        tool_state={
            "run_id": run_id,
            "agent_name": "durable-resume-service-agent",
        },
        max_iterations=4,
    )
    return Sequential(
        name="durable-resume-service",
        steps=[AgentStep(agent=agent)],
        emitter=emitter,
        run_id=run_id,
        checkpoint_store=checkpoint_store,
    )


@requires_postgres
async def test_durable_resume_service_end_to_end(
    traced_emitter: InMemoryEmitter,
) -> None:
    run_id = f"durable-resume-{uuid.uuid4().hex[:8]}"
    async with make_postgres_pool() as pool:
        try:
            await _drop_hitl_tables(pool)
            await _ensure_hitl_schema(pool)

            hitl_store = PostgresHitlRequestStore(pool)
            # No Postgres checkpoint store exists in the SDK today — use
            # InMemoryCheckpointStore. The HITL-durability property under
            # test is about the hitl_store round-trip, not checkpoint
            # persistence, so this degradation is acceptable here.
            checkpoint_store = InMemoryCheckpointStore()

            # --- Initial run: DurableRun catches the suspension ---
            workflow = _build_agent_runnable(
                hitl_store=hitl_store,
                checkpoint_store=checkpoint_store,
                emitter=traced_emitter,
                run_id=run_id,
            )
            durable = DurableRun(
                workflow,
                hitl_store=hitl_store,
                checkpoint_store=checkpoint_store,
                run_id=run_id,
            )

            suspended = await run_with_retry(
                lambda: durable.start("What is my favourite colour?"),
                max_attempts=2,
            )
            assert isinstance(suspended, SuspendedRun), (
                f"DurableRun.start must return SuspendedRun when the agent "
                f"calls ask_human; got {type(suspended).__name__}."
            )
            assert suspended.run_id == run_id, (
                f"SuspendedRun.run_id must equal the run's run_id; got {suspended.run_id!r}."
            )
            assert suspended.pending_request.request_id == suspended.suspension_info.request_id, (
                "pending_request.request_id must match suspension_info.request_id — same identity across both surfaces."
            )

            # Durability witness: the pending request lives in the store.
            pending = await hitl_store.get_pending_requests(run_id)
            assert [r.request_id for r in pending] == [suspended.pending_request.request_id], (
                f"Exactly one pending request must be visible via the "
                f"Postgres store; got {[r.request_id for r in pending]!r}."
            )

            # --- Resume: ResumeService drives the same abstraction over
            # the human's response. The factory reconstructs the workflow
            # from scratch using only what ResumeContext carries — the
            # production shape.
            def factory(ctx: ResumeContext) -> DurableRun:
                rebuilt = _build_agent_runnable(
                    hitl_store=ctx.hitl_store,  # type: ignore[arg-type]
                    checkpoint_store=ctx.checkpoint_store,
                    emitter=traced_emitter,
                    run_id=ctx.run_id,
                )
                return DurableRun(
                    rebuilt,
                    hitl_store=ctx.hitl_store,
                    checkpoint_store=ctx.checkpoint_store,
                    run_id=ctx.run_id,
                )

            service = ResumeService(
                hitl_store=hitl_store,
                checkpoint_store=checkpoint_store,
                factory=factory,
            )

            response = HumanInputResponse(
                request_id=suspended.pending_request.request_id,
                decision=HumanDecision.ANSWER,
                content="My favourite colour is teal.",
            )
            result = await run_with_retry(
                lambda: service.resume(run_id, response),
                max_attempts=2,
            )
            assert isinstance(result, ResumeResult), (
                f"ResumeService.resume must return ResumeResult for a run "
                f"that completes on resume; got {type(result).__name__}."
            )
            assert result.run_id == run_id
            assert isinstance(result.output, str), (
                f"ResumeResult.output must be a string; got {type(result.output).__name__}."
            )
            assert result.output, (
                "ResumeResult.output must be a non-empty string — the agent "
                f"must have produced a final answer; got {result.output!r}."
            )

            # After resume, no pending requests remain for the run.
            remaining = await hitl_store.get_pending_requests(run_id)
            assert remaining == [], f"Responded requests must no longer be pending after resume; got {remaining!r}."
        finally:
            await _drop_hitl_tables(pool)
