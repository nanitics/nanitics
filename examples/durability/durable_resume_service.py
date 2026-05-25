"""Canonical durable HITL: out-of-process response handoff via ResumeService.

Shows the production-shape API every durable-HITL consumer wants: one
`DurableRun` wraps the suspend side, one `ResumeService` routes every
inbound human response back through a factory — the same abstraction
whether the run completes on first resume or suspends again for
another round.

Four sections:

1. **Initial run.** Build a `Sequential` workflow with two `ApprovalGate`s
   and a trailing `FunctionStep`, wrap in `DurableRun`, call `start`.
   The returned `SuspendedRun` is a plain dataclass whose fields are
   JSON-serializable — this is the payload you ship across the process
   boundary.

2. **Simulated external system.** A stand-in for the API / CLI / UI
   that receives the `SuspendedRun` payload, displays the pending
   request to the human, and produces a `HumanInputResponse`.

3. **Resume — fresh process shape.** Reconstruct stores (same shared
   instances to simulate persistence) and build a `ResumeService` with
   a factory that rebuilds the workflow from scratch using only what
   `ResumeContext` carries. Call `service.resume(run_id, response)`.

4. **Nested suspension.** The second gate suspends again — the same
   `service.resume(...)` call returns another `SuspendedRun`, and a
   second resume completes the run. Demonstrates that `ResumeService`
   is the single entry point for every response, not a one-shot API.

The persistence pair is picked at startup: if ``NANITICS_DURABLE_EXAMPLE_DB_URL`` is set
and the ``postgres`` extra is installed, the example wires
``PostgresHitlRequestStore`` + ``PostgresCheckpointStore`` and applies
both schemas; otherwise it falls back to the in-memory pair so the
example stays runnable in CI without a database.

Related guide: docs/guides/human-in-the-loop.md
"""

from __future__ import annotations

import asyncio
import json
import os

from examples.helpers import make_emitter
from nanitics.composition import (
    CheckpointStore,
    DurableRun,
    FunctionStep,
    InMemoryCheckpointStore,
    PostgresCheckpointStore,
    ResumeContext,
    ResumeResult,
    ResumeService,
    Sequential,
    SuspendedRun,
    get_checkpoint_schema_sql,
)
from nanitics.hitl import (
    ApprovalGate,
    DurableHumanInputProvider,
    HitlRequestStore,
    HumanDecision,
    HumanInputResponse,
    InMemoryHitlRequestStore,
    PostgresHitlRequestStore,
    get_hitl_schema_sql,
)

_RUN_ID = "run-durable-resume-service"


def build_workflow(
    *,
    hitl_store: HitlRequestStore,
    checkpoint_store: CheckpointStore,
    run_id: str,
) -> Sequential:
    """Construct the durable HITL workflow.

    A production application keeps this builder as a single source of
    truth — it runs both at request-start time and inside the
    ``ResumeService`` factory on resume, so reconstruction is
    deterministic.
    """
    provider = DurableHumanInputProvider(request_store=hitl_store)

    async def publish(value: object) -> str:
        return f"Published: {value!r}"

    return Sequential(
        name="durable-publish",
        steps=[
            ApprovalGate(
                provider=provider,
                prompt="Approve initial draft for review?",
                name="initial_approval",
                run_id=run_id,
            ),
            ApprovalGate(
                provider=provider,
                prompt="Final approval to publish?",
                name="final_approval",
                run_id=run_id,
            ),
            FunctionStep(name="publish", fn=publish),
        ],
        emitter=make_emitter(trace_id=run_id),
        checkpoint_store=checkpoint_store,
        run_id=run_id,
    )


def suspended_run_to_payload(suspended: SuspendedRun) -> dict:
    """Serialize a ``SuspendedRun`` to a JSON-compatible dict.

    This is the shape you put on a message queue / HTTP response.
    ``run_id`` is the URL key for routing the response back; the
    ``pending_request`` subtree carries everything a UI needs to
    display.
    """
    return {
        "run_id": suspended.run_id,
        "checkpoint_id": suspended.checkpoint_id,
        "suspension_info": suspended.suspension_info.model_dump(mode="json"),
        "pending_request": suspended.pending_request.model_dump(mode="json"),
    }


async def external_respond(payload: dict) -> HumanInputResponse:
    """Stand-in for an HTTP endpoint / CLI that produces a response.

    A real implementation would render the request to the human, wait
    for their input, and post back through an API route keyed by
    ``run_id``. Here we auto-approve so the example is deterministic.
    """
    request = payload["pending_request"]
    return HumanInputResponse(
        request_id=request["request_id"],
        decision=HumanDecision.APPROVE,
    )


async def _build_stores() -> tuple[HitlRequestStore, CheckpointStore, object | None]:
    """Pick the production Postgres pair if available, else in-memory.

    With ``NANITICS_DURABLE_EXAMPLE_DB_URL`` set and the ``postgres`` extra installed,
    returns ``PostgresHitlRequestStore`` + ``PostgresCheckpointStore``
    backed by a fresh ``asyncpg`` pool (third tuple element), with
    both schemas applied once at startup. Otherwise returns the
    in-memory pair and a ``None`` pool — the example stays runnable
    in CI without a database.
    """
    db_url = os.getenv("NANITICS_DURABLE_EXAMPLE_DB_URL")  # example-scoped to avoid CI / .env collisions
    if db_url and PostgresCheckpointStore is not None and PostgresHitlRequestStore is not None:
        import asyncpg

        pool = await asyncpg.create_pool(db_url)
        async with pool.acquire() as conn:
            await conn.execute(get_hitl_schema_sql())
            await conn.execute(get_checkpoint_schema_sql())
        print("Using PostgresHitlRequestStore + PostgresCheckpointStore")
        return PostgresHitlRequestStore(pool), PostgresCheckpointStore(pool), pool
    print("Using in-memory stores (set NANITICS_DURABLE_EXAMPLE_DB_URL + install 'postgres' extra for Postgres)")
    return InMemoryHitlRequestStore(run_id=_RUN_ID), InMemoryCheckpointStore(), None


async def main() -> None:
    # In production point these at Postgres via NANITICS_DURABLE_EXAMPLE_DB_URL; the
    # in-memory fallback simulates durable storage across the
    # "process boundary" between the initial run and the resume.
    hitl_store, checkpoint_store, pool = await _build_stores()
    try:
        await _run(hitl_store, checkpoint_store)
    finally:
        if pool is not None:
            await pool.close()


async def _run(hitl_store: HitlRequestStore, checkpoint_store: CheckpointStore) -> None:
    # --- Section 1: Initial run ---
    print("--- Section 1: Initial run ---")

    workflow = build_workflow(
        hitl_store=hitl_store,
        checkpoint_store=checkpoint_store,
        run_id=_RUN_ID,
    )
    durable = DurableRun(
        workflow,
        hitl_store=hitl_store,
        checkpoint_store=checkpoint_store,
    )

    first = await durable.start("Q4 report")
    assert isinstance(first, SuspendedRun), "first gate must suspend"
    assert first.run_id == _RUN_ID

    first_payload = suspended_run_to_payload(first)
    # Every field is JSON-serializable — this is what crosses the wire.
    first_json = json.dumps(first_payload)
    assert json.loads(first_json)["run_id"] == _RUN_ID
    print(f"  Suspended at: {first.pending_request.prompt!r}")
    print(f"  Checkpoint:   {first.checkpoint_id}")
    print(f"  Payload size: {len(first_json)} bytes (JSON)")

    # --- Section 2: Simulated external system ---
    print("\n--- Section 2: Simulated external system ---")

    first_response = await external_respond(first_payload)
    assert first_response.request_id == first.pending_request.request_id
    print(f"  Human responded: {first_response.decision.value}")

    # --- Section 3: Resume in "fresh process" ---
    print("\n--- Section 3: Resume via ResumeService ---")

    # The factory is the only piece the resume caller configures — it
    # rebuilds the workflow from ``ResumeContext`` without needing to
    # know how the initial run was constructed.
    def factory(ctx: ResumeContext) -> DurableRun:
        workflow = build_workflow(
            hitl_store=ctx.hitl_store,
            checkpoint_store=ctx.checkpoint_store,
            run_id=ctx.run_id,
        )
        return DurableRun(
            workflow,
            hitl_store=ctx.hitl_store,
            checkpoint_store=ctx.checkpoint_store,
        )

    service = ResumeService(
        hitl_store=hitl_store,
        checkpoint_store=checkpoint_store,
        factory=factory,
    )

    # --- Section 4: Nested suspension ---
    # The second gate suspends again — same abstraction, no branching.
    print("\n--- Section 4: Nested suspension ---")

    second = await service.resume(first.run_id, first_response)
    assert isinstance(second, SuspendedRun), "second gate must suspend"
    assert second.pending_request.request_id != first.pending_request.request_id
    print(f"  Suspended again at: {second.pending_request.prompt!r}")

    second_payload = suspended_run_to_payload(second)
    second_response = await external_respond(second_payload)

    final = await service.resume(second.run_id, second_response)
    assert isinstance(final, ResumeResult), "run must complete after second resume"
    assert final.output == "Published: 'Q4 report'", f"unexpected output: {final.output!r}"
    print(f"  Final output: {final.output!r}")

    print("\nAll sections passed")


if __name__ == "__main__":
    asyncio.run(main())
