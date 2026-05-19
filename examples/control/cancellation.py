"""Cancellation: cooperative cancellation with CancellationToken, thread safety, and agent integration.

Demonstrates CancellationToken basics (create, cancel, idempotency), thread-safe cancellation
from another thread, pre-cancelled agent (immediate exit), and tool-triggered cancellation
with SafetyCancellationEvent verification.

Related guide: docs/guides/safety.md
"""

import asyncio
import threading

from examples.helpers import make_emitter, make_response
from nanitics.infrastructure import (
    MockLLMClient,
    SafetyCancellationEvent,
)
from nanitics.safety import CancellationToken
from nanitics.strategies import (
    ReActAgent,
    tool,
)
from nanitics.tracing import ToolCall


async def main() -> None:
    # --- Section 1: CancellationToken — Basics ---
    print("--- Section 1: CancellationToken — Basics ---")

    token = CancellationToken()
    assert token.is_cancelled is False

    token.cancel()
    assert token.is_cancelled is True

    # Second cancel is idempotent
    token.cancel()
    assert token.is_cancelled is True

    print("  Before cancel: is_cancelled=False")
    print("  After cancel: is_cancelled=True")
    print("  After second cancel: is_cancelled=True (idempotent)")
    print("✓ CancellationToken starts uncancelled; cancel() is irreversible and idempotent")

    # --- Section 2: Thread-Safe Cancellation ---
    print("\n--- Section 2: Thread-Safe Cancellation ---")

    token = CancellationToken()

    def cancel_from_thread() -> None:
        token.cancel()

    thread = threading.Thread(target=cancel_from_thread)
    thread.start()
    thread.join()

    assert token.is_cancelled is True

    print(f"  Token cancelled from another thread: is_cancelled={token.is_cancelled}")
    print("✓ CancellationToken is thread-safe (backed by threading.Event)")

    # --- Section 3: Pre-Cancelled Agent ---
    print("\n--- Section 3: Pre-Cancelled Agent ---")

    @tool("noop", "A tool that does nothing")
    async def noop() -> str:
        return "done"

    token = CancellationToken()
    token.cancel()  # Cancel before running

    client = MockLLMClient(
        responses=[
            make_response("This should never be reached."),
        ]
    )
    emitter = make_emitter("cancel-pre")

    agent = ReActAgent(
        name="pre-cancelled-agent",
        llm_client=client,
        emitter=emitter,
        system_prompt="You are a helpful assistant.",
        tools=[noop],
        cancellation_token=token,
    )

    result = await agent.run("Do something.")

    assert result.termination_reason == "cancelled"
    assert result.total_steps == 0
    assert result.output is None

    print(f"  Termination reason: {result.termination_reason}")
    print(f"  Steps taken: {result.total_steps}")
    print(f"  Output: {result.output}")
    print("✓ Pre-cancelled token causes immediate exit — no steps executed")

    # --- Section 4: Tool-Triggered Cancellation ---
    print("\n--- Section 4: Tool-Triggered Cancellation ---")

    token = CancellationToken()

    @tool("cancel_self", "A tool that cancels the agent")
    async def cancel_self() -> str:
        token.cancel()
        return "Cancelled!"

    client = MockLLMClient(
        responses=[
            make_response(
                "Let me cancel myself.",
                tool_calls=[ToolCall(id="tc-cancel", name="cancel_self", arguments={})],
                stop_reason="tool_use",
            ),
            make_response("This response won't be consumed."),
        ]
    )
    emitter = make_emitter("cancel-tool")

    agent = ReActAgent(
        name="self-cancelling-agent",
        llm_client=client,
        emitter=emitter,
        system_prompt="You are a helpful assistant.",
        tools=[cancel_self],
        cancellation_token=token,
    )

    result = await agent.run("Cancel yourself.")

    assert result.termination_reason == "cancelled"
    assert result.total_steps == 1

    # Verify SafetyCancellationEvent was emitted
    cancel_events = [e for e in emitter.events if isinstance(e, SafetyCancellationEvent)]
    assert len(cancel_events) == 1
    cancel_event = cancel_events[0]
    assert cancel_event.agent_name == "self-cancelling-agent"

    print(f"  Termination reason: {result.termination_reason}")
    print(f"  Steps taken: {result.total_steps}")
    print(f"  SafetyCancellationEvent: agent={cancel_event.agent_name}")
    print("✓ Tool cancelled the token; agent stopped after 1 step; SafetyCancellationEvent emitted")


if __name__ == "__main__":
    asyncio.run(main())
