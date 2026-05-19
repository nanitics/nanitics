"""ApprovalWrappedTool: gating tool execution behind human approval.

Demonstrates wrapping a tool with ApprovalWrappedTool so every invocation
requires explicit human approval. Covers schema changes, all three decision
paths (approve, reject, override), and observability through events.

Related guide: docs/guides/human-in-the-loop.md
"""

import asyncio

from examples.helpers import make_emitter
from nanitics.hitl import (
    ApprovalWrappedTool,
    CallbackHumanInputProvider,
    HumanDecision,
    HumanInputResponse,
)
from nanitics.infrastructure import (
    HumanInputRequestEvent,
    HumanInputResponseEvent,
)
from nanitics.strategies import (
    ToolRegistry,
    ToolResult,
    tool,
)
from nanitics.tracing import ToolCall


async def _dispatch(
    wrapped: ApprovalWrappedTool,
    arguments: dict,
    *,
    tool_call_id: str,
    run_id: str = "example-92",
    emitter: object | None = None,
) -> ToolResult:
    """Dispatch ``wrapped`` through a one-tool registry.

    ``ApprovalWrappedTool`` derives ``request_id`` from the ambient
    ``ToolContext`` — this helper populates the context with a ``run_id``
    and a unique ``tool_call_id`` per dispatch.
    """
    registry = ToolRegistry(emitter=emitter, tool_state={"run_id": run_id})
    registry.register(wrapped)
    return await registry.dispatch(ToolCall(id=tool_call_id, name=wrapped.schema.name, arguments=arguments))


# A simple tool to wrap in all sections.
@tool("calculate_discount", "Calculate a discount for a given price")
async def calculate_discount(price: float, percent: int) -> str:
    discount = price * percent / 100
    return f"Discount: ${discount:.2f} (final price: ${price - discount:.2f})"


async def main() -> None:
    # --- Section 1: Wrapping and Schema ---
    print("--- Section 1: Wrapping and Schema ---")

    provider = CallbackHumanInputProvider(
        lambda req: HumanInputResponse(
            request_id=req.request_id,
            decision=HumanDecision.APPROVE,
        )
    )
    wrapped = ApprovalWrappedTool(tool=calculate_discount, provider=provider)

    # Wrapped tool preserves name, description, and parameters
    assert wrapped.schema.name == "calculate_discount"
    assert wrapped.schema.description == "Calculate a discount for a given price"
    assert wrapped.schema.parameters == calculate_discount.schema.parameters
    print("✓ Wrapped tool preserves name, description, and parameters")

    # Only difference: requires_approval is True
    assert wrapped.schema.requires_approval is True
    assert calculate_discount.schema.requires_approval is False
    print("✓ Wrapped schema has requires_approval=True, original unchanged")

    # --- Section 2: Approve Path ---
    print("\n--- Section 2: Approve Path ---")

    provider = CallbackHumanInputProvider(
        lambda req: HumanInputResponse(
            request_id=req.request_id,
            decision=HumanDecision.APPROVE,
        )
    )
    wrapped = ApprovalWrappedTool(tool=calculate_discount, provider=provider)
    result = await _dispatch(wrapped, {"price": 100.0, "percent": 15}, tool_call_id="tc-approve")

    assert result.content == "Discount: $15.00 (final price: $85.00)"
    print(f"✓ Approved execution returns tool output: {result.content}")

    # --- Section 3: Reject Path ---
    print("\n--- Section 3: Reject Path ---")

    call_count = 0

    @tool("tracked_delete", "Delete a record (tracks calls)")
    async def tracked_delete(record_id: str) -> str:
        nonlocal call_count
        call_count += 1
        return f"Deleted {record_id}"

    provider = CallbackHumanInputProvider(
        lambda req: HumanInputResponse(
            request_id=req.request_id,
            decision=HumanDecision.REJECT,
            content="Too risky without backup",
        )
    )
    wrapped = ApprovalWrappedTool(tool=tracked_delete, provider=provider)
    result = await _dispatch(wrapped, {"record_id": "rec-42"}, tool_call_id="tc-reject")

    assert call_count == 0, "Inner tool was never called"
    assert "Action rejected by human" in result.content
    assert "Too risky without backup" in result.content
    print(f"✓ Rejection prevented execution: {result.content}")

    # --- Section 4: Override Path ---
    print("\n--- Section 4: Override Path ---")

    provider = CallbackHumanInputProvider(
        lambda req: HumanInputResponse(
            request_id=req.request_id,
            decision=HumanDecision.OVERRIDE,
            metadata={"modified_params": {"percent": 10}},
        )
    )
    wrapped = ApprovalWrappedTool(tool=calculate_discount, provider=provider)
    result = await _dispatch(wrapped, {"price": 200.0, "percent": 50}, tool_call_id="tc-override")

    # Human changed percent from 50 to 10
    assert result.content == "Discount: $20.00 (final price: $180.00)"
    print(f"✓ Modified execution used altered parameters: {result.content}")

    # --- Section 5: Event Emission ---
    print("\n--- Section 5: Event Emission ---")

    emitter = make_emitter("approval-wrapped-events")

    # Run approve path
    approve_provider = CallbackHumanInputProvider(
        lambda req: HumanInputResponse(
            request_id=req.request_id,
            decision=HumanDecision.APPROVE,
        )
    )
    wrapped = ApprovalWrappedTool(
        tool=calculate_discount,
        provider=approve_provider,
        emitter=emitter,
    )
    await _dispatch(
        wrapped,
        {"price": 50.0, "percent": 20},
        tool_call_id="tc-events-approve",
        emitter=emitter,
    )

    # Run reject path with same emitter
    reject_provider = CallbackHumanInputProvider(
        lambda req: HumanInputResponse(
            request_id=req.request_id,
            decision=HumanDecision.REJECT,
            content="Not authorized",
        )
    )
    wrapped = ApprovalWrappedTool(
        tool=calculate_discount,
        provider=reject_provider,
        emitter=emitter,
    )
    await _dispatch(
        wrapped,
        {"price": 50.0, "percent": 20},
        tool_call_id="tc-events-reject",
        emitter=emitter,
    )

    request_events = [e for e in emitter.events if isinstance(e, HumanInputRequestEvent)]
    response_events = [e for e in emitter.events if isinstance(e, HumanInputResponseEvent)]

    assert len(request_events) == 2, "Two request events (one per execution)"
    assert len(response_events) == 2, "Two response events (one per execution)"

    # Request events carry tool context
    assert request_events[0].tool_name == "calculate_discount"
    assert request_events[0].request_type == "approval"
    assert "Approve tool" in request_events[0].prompt
    print("✓ Request events carry tool_name, request_type, and prompt")

    # Response events carry decision and timing
    assert response_events[0].decision == "approve"
    assert response_events[0].has_content is False
    assert response_events[0].wait_duration_ms >= 0

    assert response_events[1].decision == "reject"
    assert response_events[1].has_content is True
    print("✓ Response events carry decision, has_content, and wait_duration_ms")

    print("\nAll sections passed ✓")


if __name__ == "__main__":
    asyncio.run(main())
