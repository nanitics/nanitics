"""HITL tool factories: agent-initiated human interaction.

Demonstrates `create_request_approval_tool`, `create_ask_human_tool`, and
`create_hitl_tools` — tool factories that give an agent the ability to involve
a human during its reasoning loop. Unlike the gate-based examples where the
*developer* decides where human oversight occurs, here the *agent* decides
when to ask questions or request approval.

Related guide: docs/guides/human-in-the-loop.md
"""

import asyncio

from examples.helpers import make_emitter, make_response
from nanitics.hitl import (
    CallbackHumanInputProvider,
    HumanDecision,
    HumanInputRequest,
    HumanInputResponse,
    HumanInputType,
    create_ask_human_tool,
    create_hitl_tools,
    create_request_approval_tool,
)
from nanitics.infrastructure import (
    HumanInputRequestEvent,
    HumanInputResponseEvent,
    MockLLMClient,
)
from nanitics.strategies import (
    ReActAgent,
    ToolRegistry,
)
from nanitics.tracing import (
    InMemoryEmitter,
    ToolCall,
)


async def main() -> None:
    # --- Section 1: Tool Creation and Schema ---
    print("--- Section 1: Tool Creation and Schema ---")

    provider = CallbackHumanInputProvider(
        lambda req: HumanInputResponse(
            request_id=req.request_id,
            decision=HumanDecision.APPROVE,
        )
    )

    # Individual factories
    approval_tool = create_request_approval_tool(provider)
    ask_tool = create_ask_human_tool(provider)

    assert approval_tool.schema.name == "request_approval"
    assert ask_tool.schema.name == "ask_human"
    print(f"  request_approval: {approval_tool.schema.description[:60]}...")
    print(f"  ask_human: {ask_tool.schema.description[:60]}...")

    # request_approval parameters: action, reason, details
    approval_props = approval_tool.schema.parameters.get("properties", {})
    assert "action" in approval_props
    assert "reason" in approval_props
    assert "details" in approval_props
    print(f"  request_approval parameters: {sorted(approval_props)}")

    # ask_human parameters: question, context_info, options
    ask_props = ask_tool.schema.parameters.get("properties", {})
    assert "question" in ask_props
    assert "context_info" in ask_props
    assert "options" in ask_props
    print(f"  ask_human parameters: {sorted(ask_props)}")

    # Convenience factory returns both
    both = create_hitl_tools(provider)
    assert len(both) == 2
    names = {t.schema.name for t in both}
    assert names == {"request_approval", "ask_human"}
    print("✓ Tool creation and schema inspection")

    # --- Section 2: Approval Responses ---
    print("\n--- Section 2: Approval Responses ---")

    # HITL tools derive request_id from ``{run_id}:{tool_call_id}`` supplied by
    # a ToolRegistry. Build one registry per dispatch in this section — a
    # fresh ToolCall.id per call keeps identities unique within a run.

    async def _run_approval_tool(provider: CallbackHumanInputProvider, tool_call_id: str, **args: object) -> str:
        tool = create_request_approval_tool(provider)
        registry = ToolRegistry(tool_state={"run_id": "example-93-section-2"})
        registry.register(tool)
        call = ToolCall(id=tool_call_id, name="request_approval", arguments=args)
        out = await registry.dispatch(call)
        return out.content

    # Approve
    provider = CallbackHumanInputProvider(
        lambda req: HumanInputResponse(
            request_id=req.request_id,
            decision=HumanDecision.APPROVE,
        )
    )
    content = await _run_approval_tool(provider, "tc-approve", action="Deploy config", reason="New settings ready")
    assert content == "Approved."
    print(f'  APPROVE → "{content}"')

    # Reject
    provider = CallbackHumanInputProvider(
        lambda req: HumanInputResponse(
            request_id=req.request_id,
            decision=HumanDecision.REJECT,
            content="Not authorized for production",
        )
    )
    content = await _run_approval_tool(provider, "tc-reject", action="Delete records", reason="Cleanup needed")
    assert content == "Rejected. Reason: Not authorized for production"
    print(f'  REJECT → "{content}"')

    # Override
    provider = CallbackHumanInputProvider(
        lambda req: HumanInputResponse(
            request_id=req.request_id,
            decision=HumanDecision.OVERRIDE,
            content="Deploy to staging first",
        )
    )
    content = await _run_approval_tool(provider, "tc-override", action="Deploy config", reason="New settings")
    assert content == "Approved with overrides: Deploy to staging first"
    print(f'  OVERRIDE → "{content}"')

    print("✓ All approval response formats verified")

    # --- Section 3: Question Answers ---
    print("\n--- Section 3: Question Answers ---")

    async def _run_ask_tool(provider: CallbackHumanInputProvider, tool_call_id: str, **args: object) -> str:
        tool = create_ask_human_tool(provider)
        registry = ToolRegistry(tool_state={"run_id": "example-93-section-3"})
        registry.register(tool)
        call = ToolCall(id=tool_call_id, name="ask_human", arguments=args)
        out = await registry.dispatch(call)
        return out.content

    # Simple question
    provider = CallbackHumanInputProvider(
        lambda req: HumanInputResponse(
            request_id=req.request_id,
            decision=HumanDecision.ANSWER,
            content="PostgreSQL",
        )
    )
    content = await _run_ask_tool(
        provider,
        "tc-db",
        question="Which database engine should I use?",
        context_info="Setting up the data layer",
    )
    assert content == "Human response: PostgreSQL"
    print(f'  Simple question → "{content}"')

    # Question with options — verify options are passed to the provider
    captured_requests: list[HumanInputRequest] = []

    def capture_and_answer(req: HumanInputRequest) -> HumanInputResponse:
        captured_requests.append(req)
        return HumanInputResponse(
            request_id=req.request_id,
            decision=HumanDecision.ANSWER,
            content="Option B",
        )

    provider = CallbackHumanInputProvider(capture_and_answer)
    content = await _run_ask_tool(
        provider,
        "tc-options",
        question="Which approach?",
        options=["Option A", "Option B", "Option C"],
    )
    assert content == "Human response: Option B"
    assert captured_requests[0].options == ["Option A", "Option B", "Option C"]
    print(f'  With options → "{content}" (options passed to provider)')

    print("✓ All question response formats verified")

    # --- Section 4: Agent Integration ---
    print("\n--- Section 4: Agent Integration ---")

    # One provider that routes by request type
    call_count = 0

    def routing_provider(req: HumanInputRequest) -> HumanInputResponse:
        nonlocal call_count
        call_count += 1
        if req.request_type == HumanInputType.QUESTION:
            return HumanInputResponse(
                request_id=req.request_id,
                decision=HumanDecision.ANSWER,
                content="PostgreSQL",
            )
        # Approval or plan review
        return HumanInputResponse(
            request_id=req.request_id,
            decision=HumanDecision.APPROVE,
        )

    provider = CallbackHumanInputProvider(routing_provider)
    hitl_tools = create_hitl_tools(provider)

    client = MockLLMClient(
        responses=[
            # Step 1: Agent asks which database to use
            make_response(
                "I need to know which database the user prefers.",
                tool_calls=[
                    ToolCall(
                        id="tc-1",
                        name="ask_human",
                        arguments={
                            "question": "Which database engine should I use?",
                            "options": ["PostgreSQL", "MySQL"],
                        },
                    )
                ],
                stop_reason="tool_use",
            ),
            # Step 2: Agent requests approval to proceed
            make_response(
                "The user chose PostgreSQL. Let me get approval to proceed.",
                tool_calls=[
                    ToolCall(
                        id="tc-2",
                        name="request_approval",
                        arguments={
                            "action": "Set up PostgreSQL database",
                            "reason": "User selected PostgreSQL as database engine",
                        },
                    )
                ],
                stop_reason="tool_use",
            ),
            # Step 3: Agent completes the task
            make_response("PostgreSQL database has been configured successfully."),
        ]
    )

    emitter = make_emitter("hitl-agent")
    agent = ReActAgent(
        name="setup-agent",
        llm_client=client,
        emitter=emitter,
        system_prompt="You are a helpful setup assistant.",
        tools=hitl_tools,
        run_id="example-93-setup-agent",
    )
    result = await agent.run("Set up a database for the project.")

    assert "PostgreSQL" in result.output
    assert call_count == 2, f"Provider called {call_count} times, expected 2"
    assert result.total_steps == 3
    print(f"  Agent output: {result.output}")
    print(f"  Steps: {result.total_steps} (ask → approve → respond)")
    print(f"  Provider calls: {call_count}")
    print("✓ Agent-initiated HITL flow complete")

    # --- Section 5: Observability ---
    print("\n--- Section 5: Observability ---")

    emitter = InMemoryEmitter(trace_id="hitl-events")

    # Use ToolRegistry to dispatch — this injects ToolContext with the emitter
    approve_provider = CallbackHumanInputProvider(
        lambda req: HumanInputResponse(
            request_id=req.request_id,
            decision=HumanDecision.APPROVE,
        )
    )
    answer_provider = CallbackHumanInputProvider(
        lambda req: HumanInputResponse(
            request_id=req.request_id,
            decision=HumanDecision.ANSWER,
            content="Yes, proceed",
        )
    )

    approval_tool = create_request_approval_tool(approve_provider)
    ask_tool = create_ask_human_tool(answer_provider)

    registry = ToolRegistry(emitter=emitter, tool_state={"run_id": "example-93-obs"})
    registry.register(approval_tool)
    registry.register(ask_tool)

    await registry.dispatch(
        ToolCall(
            id="obs-1",
            name="request_approval",
            arguments={"action": "Deploy service", "reason": "Ready for release"},
        )
    )
    await registry.dispatch(
        ToolCall(
            id="obs-2",
            name="ask_human",
            arguments={"question": "Should we enable caching?"},
        )
    )

    request_events = [e for e in emitter.events if isinstance(e, HumanInputRequestEvent)]
    response_events = [e for e in emitter.events if isinstance(e, HumanInputResponseEvent)]

    assert len(request_events) == 2, f"Expected 2 request events, got {len(request_events)}"
    assert len(response_events) == 2, f"Expected 2 response events, got {len(response_events)}"

    # Request event fields
    assert request_events[0].request_type == "approval"
    assert "Deploy service" in request_events[0].prompt
    assert request_events[1].request_type == "question"
    assert "caching" in request_events[1].prompt
    print("  Request events: request_type, prompt ✓")

    # Response event fields
    assert response_events[0].decision == "approve"
    assert response_events[0].has_content is False
    assert response_events[0].wait_duration_ms >= 0

    assert response_events[1].decision == "answer"
    assert response_events[1].has_content is True
    assert response_events[1].wait_duration_ms >= 0
    print("  Response events: decision, has_content, wait_duration_ms ✓")

    # Request IDs match between request and response
    assert request_events[0].request_id == response_events[0].request_id
    assert request_events[1].request_id == response_events[1].request_id
    print("  Request IDs match between request and response ✓")

    print("✓ Event pairs emitted with correct fields")

    print("\nAll sections passed ✓")


if __name__ == "__main__":
    asyncio.run(main())
