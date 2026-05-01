"""Async HITL: suspending on an asyncio.Future until an external caller resolves the request.

Demonstrates AsyncHumanInputProvider — the HTTP-integration HITL provider. Unlike
CallbackHumanInputProvider (in-process) and DurableHumanInputProvider (checkpoint-backed),
AsyncHumanInputProvider suspends on an asyncio.Future and waits for an external caller —
typically a FastAPI endpoint — to invoke ``resolve(request_id, response)``. This is the
shape most production integrations take: the agent runs on one side, an admin UI or API
caller resolves the human input on the other side.

Covers:
  1. In-process async resolution — the basic request/resolve handshake.
  2. HTTP-integration simulation — producer/consumer coroutines model a FastAPI endpoint.
  3. Store-backed persistence — HitlRequestStore round-trips requests/responses.
  4. Timeout via asyncio.wait_for — the caller's responsibility, not the provider's.

Related guide: docs/guides/human-in-the-loop.md
Related example: examples/hitl/approval_gate.py (in-process HITL with ApprovalGate).
"""

import asyncio

from examples.helpers import make_emitter
from nanitics import (
    ApprovalGate,
    AsyncHumanInputProvider,
    HumanDecision,
    HumanInputRequest,
    HumanInputResponse,
    HumanInputType,
    InMemoryHitlRequestStore,
)
from nanitics.infrastructure import (
    HumanInputRequestEvent,
    HumanInputResponseEvent,
)


async def main() -> None:
    # --- Section 1: In-Process Async Resolution ---
    print("--- Section 1: In-Process Async Resolution ---")

    emitter = make_emitter("async-hitl-s1")
    provider = AsyncHumanInputProvider()
    gate = ApprovalGate(
        provider=provider,
        emitter=emitter,
        prompt="Approve publishing this draft?",
        run_id="example-94-s1",
    )

    # Drive the gate in a background task so it suspends on the provider's Future.
    task = asyncio.create_task(gate.execute("draft content"))

    # The gate has scheduled the request; give the event loop a chance to run it
    # up to the suspension point (no sleeps — just yield).
    for _ in range(5):
        await asyncio.sleep(0)
        if provider.get_pending():
            break

    pending = provider.get_pending()
    assert len(pending) == 1, f"Expected 1 pending request, got {len(pending)}"
    pending_request = pending[0]
    assert pending_request.prompt == "Approve publishing this draft?"

    # External "resolver" supplies the human's decision.
    resolved = await provider.resolve(
        pending_request.request_id,
        HumanInputResponse(
            request_id=pending_request.request_id,
            decision=HumanDecision.APPROVE,
        ),
    )
    assert resolved is True

    result = await task
    assert result.output == "draft content", "APPROVE passes the input through unchanged"

    request_events = [e for e in emitter.events if isinstance(e, HumanInputRequestEvent)]
    response_events = [e for e in emitter.events if isinstance(e, HumanInputResponseEvent)]
    assert len(request_events) == 1 and len(response_events) == 1

    print(f"  Gate suspended on {len(pending)} pending request")
    print(f"  External resolve() unblocked the future; output: '{result.output}'")
    print(f"  Trace captured {len(request_events)} request + {len(response_events)} response event")

    # --- Section 2: HTTP-Integration Simulation (Producer/Consumer) ---
    print("\n--- Section 2: HTTP-Integration Simulation ---")

    # Model the production shape without a real HTTP server:
    #   - ``producer`` plays the agent side: kicks off approval-gated steps.
    #   - ``consumer`` plays the admin-UI side: polls for pending requests and resolves them.
    # In a FastAPI app these would be:
    #   - agent runs in a background task on POST /runs,
    #   - UI calls GET /hitl/pending and POST /hitl/{request_id}/resolve.
    provider = AsyncHumanInputProvider()
    http_emitter = make_emitter("async-hitl-s2")

    drafts = ["draft-alpha", "draft-beta", "draft-gamma"]
    decisions: dict[str, HumanDecision] = {
        "draft-alpha": HumanDecision.APPROVE,
        "draft-beta": HumanDecision.REJECT,
        "draft-gamma": HumanDecision.APPROVE,
    }

    async def producer() -> list[str | None]:
        # Each draft runs through its own gate — sequentially here for clarity, though
        # a real agent could gate these concurrently as well. Each draft gets its own
        # run_id so deterministic request_id derivation distinguishes the approvals.
        outputs: list[str | None] = []
        for draft in drafts:
            gate = ApprovalGate(
                provider=provider,
                emitter=http_emitter,
                prompt=lambda d=draft: f"Approve {d}?",
                run_id=f"example-94-s2-{draft}",
            )
            step = await gate.execute(draft)
            outputs.append(step.output)
        return outputs

    async def consumer(expected: int) -> int:
        resolved_count = 0
        while resolved_count < expected:
            pending = provider.get_pending()
            if not pending:
                await asyncio.sleep(0)  # yield to the producer, don't busy-loop
                continue
            req = pending[0]
            # The consumer reads the prompt to derive the decision, just like an admin UI
            # would display the prompt to a human and collect their answer.
            matched = next(d for d in drafts if d in req.prompt)
            await provider.resolve(
                req.request_id,
                HumanInputResponse(request_id=req.request_id, decision=decisions[matched]),
            )
            resolved_count += 1
        return resolved_count

    producer_task = asyncio.create_task(producer())
    consumer_task = asyncio.create_task(consumer(expected=len(drafts)))
    outputs, resolved = await asyncio.gather(producer_task, consumer_task)

    assert resolved == 3
    assert outputs == ["draft-alpha", None, "draft-gamma"], f"Unexpected outputs: {outputs}"

    # Every pending request has been consumed.
    assert provider.get_pending() == []

    print(f"  Resolved {resolved} requests via producer/consumer coroutines")
    print(f"  Producer outputs (None = rejected): {outputs}")
    print("  This is the shape a FastAPI integration takes: agent handler + admin-UI resolver")

    # --- Section 3: Store-Backed Persistence ---
    print("\n--- Section 3: Store-Backed Persistence ---")

    # Passing a store lets the provider persist the request *and* response so an external
    # system can inspect them. In production swap InMemoryHitlRequestStore for
    # PostgresHitlRequestStore — the provider API does not change. Checkpoint-backed
    # suspension across process restarts is demonstrated in examples/durability/checkpoint_suspension.py.
    run_id = "run-s3"
    store = InMemoryHitlRequestStore(run_id=run_id)
    provider = AsyncHumanInputProvider(store=store)

    request = HumanInputRequest(
        run_id=run_id,
        request_type=HumanInputType.APPROVAL,
        prompt="Approve tool invocation?",
    )

    task = asyncio.create_task(provider.request_input(request))
    for _ in range(5):
        await asyncio.sleep(0)
        if provider.get_pending():
            break

    # The store has the pending request visible to any external process.
    pending_in_store = await store.get_pending_requests(run_id=run_id)
    assert len(pending_in_store) == 1
    assert pending_in_store[0].request_id == request.request_id

    await provider.resolve(
        request.request_id,
        HumanInputResponse(
            request_id=request.request_id,
            decision=HumanDecision.APPROVE,
            content="Looks good",
        ),
    )
    response = await task
    assert response.decision == HumanDecision.APPROVE

    # After resolution the response is persisted and retrievable by request_id.
    stored_response = await store.get_response(request.request_id)
    assert stored_response is not None
    assert stored_response.decision == HumanDecision.APPROVE
    assert stored_response.content == "Looks good"

    # And the "pending" list is empty now that a response exists.
    still_pending = await store.get_pending_requests(run_id=run_id)
    assert still_pending == []

    print("  Store captured the request before resolution: 1 pending")
    print(
        f"  Store round-tripped the response: decision={stored_response.decision.value}, "
        f"content='{stored_response.content}'"
    )
    print("  After resolution, store reports 0 pending requests for the run")
    print("  Swap InMemoryHitlRequestStore → PostgresHitlRequestStore for cross-restart durability")

    # --- Section 4: Timeout via asyncio.wait_for ---
    print("\n--- Section 4: Timeout via asyncio.wait_for ---")

    # AsyncHumanInputProvider deliberately does not build in a timeout: the right timeout is
    # policy, not mechanism. The caller wraps request_input() in asyncio.wait_for() and
    # handles asyncio.TimeoutError when the human doesn't respond in time.

    # Happy path: the resolver fires before the timeout.
    provider = AsyncHumanInputProvider()
    happy_request = HumanInputRequest(
        request_type=HumanInputType.QUESTION,
        prompt="Pick a color.",
    )

    async def resolve_quickly() -> None:
        for _ in range(5):
            await asyncio.sleep(0)
            if provider.get_pending():
                break
        await provider.resolve(
            happy_request.request_id,
            HumanInputResponse(
                request_id=happy_request.request_id,
                decision=HumanDecision.ANSWER,
                content="blue",
            ),
        )

    resolver_task = asyncio.create_task(resolve_quickly())
    response = await asyncio.wait_for(provider.request_input(happy_request), timeout=1.0)
    await resolver_task
    assert response.content == "blue"
    print("  Happy path (1s timeout, prompt resolve): answered 'blue'")

    # Timeout path: no resolver.
    timeout_request = HumanInputRequest(
        request_type=HumanInputType.QUESTION,
        prompt="Pick a shape.",
    )
    try:
        await asyncio.wait_for(provider.request_input(timeout_request), timeout=0.05)
    except TimeoutError:
        print("  Timeout path (50ms, no resolver): TimeoutError raised as expected")
    else:
        raise AssertionError("Expected asyncio.TimeoutError; none raised")

    print("\nAll sections passed ✓")


if __name__ == "__main__":
    asyncio.run(main())
