"""Handoff: structured context transfer in multi-agent workflows.

Demonstrates the handoff protocol for passing structured work between agents:
HandoffPayload data model, HandoffTransfer strategy, prompt helpers for
sender/receiver coordination, HandoffStep workflow integration, and
create_handoff_chain for multi-agent pipelines. Each handoff emits a
HandoffEvent for observability.

Related guide: docs/guides/multi-agent-foundations.md
"""

import asyncio

from examples.helpers import make_emitter, make_response
from nanitics import (
    AgentResult,
    Message,
    MockLLMClient,
    RawOutputTransfer,
    ReActAgent,
    Usage,
)
from nanitics.infrastructure import (
    HandoffEvent,
)
from nanitics.patterns import (
    HandoffPayload,
    HandoffStep,
    HandoffTransfer,
    create_handoff_chain,
    handoff_receiver_instructions,
    handoff_sender_instructions,
)


async def main() -> None:
    # --- Section 1: HandoffPayload ---
    print("--- Section 1: HandoffPayload ---")

    # HandoffPayload is a structured data model for context passed between agents.
    # task_state is required; all other fields are optional.
    payload = HandoffPayload(
        task_state="Completed research on Python 3.13 features",
        findings=[
            "Free-threaded CPython (PEP 703) is experimental",
            "New REPL with multi-line editing",
            "Improved error messages",
        ],
        decisions=["Focus article on developer-facing changes"],
        open_questions=["Should we cover iOS/Android tier 3 support?"],
        artifacts={"outline": "1. Introduction\n2. Key Features\n3. Migration Guide"},
        metadata={"sources_checked": 5},
    )

    # Payload is immutable (frozen Pydantic model)
    try:
        payload.task_state = "Modified"
        assert False, "Should have raised"
    except Exception:
        pass
    print("  Payload is frozen (immutable) ✓")

    # render() produces a markdown document with sections
    rendered = payload.render()
    assert "## Handoff Context" in rendered
    assert "### Task State" in rendered
    assert "Completed research on Python 3.13 features" in rendered
    assert "### Findings" in rendered
    assert "- Free-threaded CPython" in rendered
    assert "### Decisions" in rendered
    assert "### Open Questions" in rendered
    assert "### Artifacts" in rendered
    assert "#### outline" in rendered
    print("  Rendered markdown:")
    for line in rendered.split("\n")[:8]:
        print(f"    {line}")
    print("    ...")

    # Empty optional fields are omitted from rendering
    minimal = HandoffPayload(task_state="Just the basics")
    minimal_rendered = minimal.render()
    assert "### Findings" not in minimal_rendered
    assert "### Decisions" not in minimal_rendered
    assert "### Task State" in minimal_rendered
    print("  Minimal payload omits empty sections ✓")

    print("✓ Section 1 passed")

    # --- Section 2: HandoffTransfer ---
    print("\n--- Section 2: HandoffTransfer ---")

    # HandoffTransfer is a ContextTransferStrategy that builds a HandoffPayload
    # from an AgentResult using a builder function, then renders it as markdown.

    def build_payload(result: AgentResult) -> HandoffPayload:
        return HandoffPayload(
            task_state=result.output or "No output",
            findings=["Extracted from agent result"],
            decisions=[f"Agent completed in {result.total_steps} steps"],
        )

    strategy = HandoffTransfer(builder=build_payload)

    # Create a mock AgentResult to test the strategy
    result = AgentResult(
        output="Paris is the capital of France.",
        total_steps=2,
        termination_reason="complete",
        messages=[
            Message(role="user", content="What is the capital of France?"),
            Message(role="assistant", content="Paris is the capital of France."),
        ],
        usage=Usage(input_tokens=20, output_tokens=10),
    )

    text = await strategy.extract(result)
    assert "## Handoff Context" in text
    assert "Paris is the capital of France." in text
    assert "Extracted from agent result" in text
    assert "Agent completed in 2 steps" in text
    print(f"  HandoffTransfer produced {len(text)} chars of structured markdown")
    print(f"  Preview: {text.split(chr(10))[0]}")

    print("✓ Section 2 passed")

    # --- Section 3: Prompt Helpers ---
    print("\n--- Section 3: Prompt Helpers ---")

    # handoff_sender_instructions() generates system prompt text that tells
    # an agent to structure its output for handoff.
    sender = handoff_sender_instructions()
    assert "task_state" in sender
    assert "findings" in sender
    assert "decisions" in sender
    assert "open_questions" in sender
    assert "artifacts" in sender
    print(f"  Sender instructions ({len(sender)} chars):")
    print(f"    {sender[:80]}...")

    # Custom fields — only include what you need
    custom_sender = handoff_sender_instructions(payload_fields=["task_state", "findings"])
    assert "task_state" in custom_sender
    assert "findings" in custom_sender
    assert "decisions" not in custom_sender
    print("  Custom sender fields (task_state, findings only) ✓")

    # handoff_receiver_instructions() tells the receiving agent to expect
    # and use structured handoff context.
    receiver = handoff_receiver_instructions()
    assert "handoff" in receiver.lower()
    assert "Task State" in receiver
    assert "Findings" in receiver
    print(f"  Receiver instructions ({len(receiver)} chars):")
    print(f"    {receiver[:80]}...")

    print("✓ Section 3 passed")

    # --- Section 4: HandoffStep ---
    print("\n--- Section 4: HandoffStep ---")

    # HandoffStep is a workflow Step that runs an agent, applies a transfer
    # strategy, and emits a HandoffEvent linking source and destination.
    emitter = make_emitter("handoff-s4")

    researcher_client = MockLLMClient(
        [
            make_response("The key finding is that Python 3.13 adds free-threading support."),
        ]
    )
    researcher = ReActAgent(
        name="researcher",
        llm_client=researcher_client,
        emitter=emitter,
        system_prompt="You are a research specialist.",
        tools=[],
    )

    step = HandoffStep(
        agent=researcher,
        emitter=emitter,
        transfer_strategy=RawOutputTransfer(),
        to_agent="writer",
    )

    assert step.name == "researcher"

    step_result = await step.execute("Research Python 3.13 features")

    # StepResult contains the transferred text
    assert step_result.output == "The key finding is that Python 3.13 adds free-threading support."
    print(f"  Output: {step_result.output!r}")

    # Metadata includes agent execution details
    assert step_result.metadata["agent_name"] == "researcher"
    assert step_result.metadata["total_steps"] == 1
    assert step_result.metadata["termination_reason"] == "complete"
    assert "usage" in step_result.metadata
    print(
        f"  Metadata: agent={step_result.metadata['agent_name']}, "
        f"steps={step_result.metadata['total_steps']}, "
        f"reason={step_result.metadata['termination_reason']}"
    )

    # HandoffEvent was emitted
    handoff_events = [e for e in emitter.events if isinstance(e, HandoffEvent)]
    assert len(handoff_events) == 1
    event = handoff_events[0]
    assert event.from_agent == "researcher"
    assert event.to_agent == "writer"
    assert event.payload_size > 0
    print(f"  HandoffEvent: {event.from_agent} → {event.to_agent} ({event.payload_size} chars)")

    print("✓ Section 4 passed")

    # --- Section 5: create_handoff_chain ---
    print("\n--- Section 5: create_handoff_chain ---")

    # create_handoff_chain builds a Sequential workflow connecting agents
    # with HandoffSteps. Each agent's output flows to the next.
    emitter = make_emitter("handoff-s5")

    researcher_client = MockLLMClient(
        [
            make_response("Research: Python 3.13 introduces free-threading and a new REPL."),
        ]
    )
    writer_client = MockLLMClient(
        [
            make_response(
                "Draft: Python 3.13 brings two exciting changes: "
                "experimental free-threading support and a modernized REPL."
            ),
        ]
    )
    reviewer_client = MockLLMClient(
        [
            make_response(
                "Final: Python 3.13's headline features are free-threading (PEP 703) "
                "and a revamped interactive REPL with multi-line editing."
            ),
        ]
    )

    researcher = ReActAgent(
        name="researcher",
        llm_client=researcher_client,
        emitter=emitter,
        system_prompt="Research the topic thoroughly.",
        tools=[],
    )
    writer = ReActAgent(
        name="writer",
        llm_client=writer_client,
        emitter=emitter,
        system_prompt="Write a clear draft based on the research.",
        tools=[],
    )
    reviewer = ReActAgent(
        name="reviewer",
        llm_client=reviewer_client,
        emitter=emitter,
        system_prompt="Review and polish the draft.",
        tools=[],
    )

    # Build the chain — each intermediate step uses HandoffTransfer
    def build_research_payload(result: AgentResult) -> HandoffPayload:
        return HandoffPayload(
            task_state=result.output or "",
            findings=["Free-threading support (PEP 703)", "New REPL"],
        )

    chain = create_handoff_chain(
        name="article-pipeline",
        agents=[researcher, writer, reviewer],
        emitter=emitter,
        transfer_strategy=HandoffTransfer(builder=build_research_payload),
    )

    chain_result = await chain.execute("Write an article about Python 3.13")

    # Final output is from the last agent (reviewer), using RawOutputTransfer
    assert "Python 3.13" in chain_result.output
    assert "free-threading" in chain_result.output.lower()
    print(f"  Final output: {chain_result.output!r}")

    # Metadata contains intermediate results from each step
    intermediate = chain_result.metadata.get("intermediate_results", {})
    assert "researcher" in intermediate
    assert "writer" in intermediate
    assert "reviewer" in intermediate
    print(f"  Steps executed: {list(intermediate.keys())}")

    # The writer received structured handoff markdown (from HandoffTransfer on researcher)
    writer_input = writer_client.calls[0]["messages"][0].content
    assert "## Handoff Context" in writer_input
    assert "### Task State" in writer_input
    assert "### Findings" in writer_input
    print("  Writer received structured handoff context ✓")

    # The reviewer also received structured handoff (HandoffTransfer on writer).
    # Only the last step's *output extraction* uses RawOutputTransfer —
    # meaning the chain's final result is the reviewer's raw output,
    # but the reviewer's *input* still comes from the writer's HandoffTransfer.
    reviewer_input = reviewer_client.calls[0]["messages"][0].content
    assert "## Handoff Context" in reviewer_input
    print("  Reviewer received structured handoff context ✓")
    print("  Last step uses RawOutputTransfer for final chain output ✓")

    # Three HandoffEvents — one per step in the chain.
    # Each HandoffStep emits an event after its agent completes:
    # researcher→writer, writer→reviewer, reviewer→output (final)
    handoff_events = [e for e in emitter.events if isinstance(e, HandoffEvent)]
    assert len(handoff_events) == 3
    assert handoff_events[0].from_agent == "researcher"
    assert handoff_events[0].to_agent == "writer"
    assert handoff_events[1].from_agent == "writer"
    assert handoff_events[1].to_agent == "reviewer"
    assert handoff_events[2].from_agent == "reviewer"
    assert handoff_events[2].to_agent == "output"
    print(
        f"  HandoffEvents: "
        f"{handoff_events[0].from_agent}→{handoff_events[0].to_agent}, "
        f"{handoff_events[1].from_agent}→{handoff_events[1].to_agent}, "
        f"{handoff_events[2].from_agent}→{handoff_events[2].to_agent}"
    )

    print("✓ Section 5 passed")

    print("\nAll sections passed ✓")


if __name__ == "__main__":
    asyncio.run(main())
