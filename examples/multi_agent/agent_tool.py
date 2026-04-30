"""Agent-as-tool: wrapping an agent behind the Tool protocol for delegation.

Demonstrates AgentTool — creating a specialist agent, wrapping it as a tool,
inspecting the schema, executing it directly, using it inside a caller agent
for end-to-end delegation, and passing multimodal content blocks to delegates.
Covers ToolResult metadata, DelegationEvent tracing, and the Tool protocol.

Related guide: docs/guides/multi-agent-foundations.md
"""

import asyncio

from examples.helpers import make_emitter, make_response
from nanitics import (
    AgentTool,
    ImageContentBlock,
    MockLLMClient,
    ReActAgent,
    ReasoningAgent,
    Tool,
    ToolCall,
    tool,
)
from nanitics.infrastructure import (
    DelegationEvent,
)

# --- Shared specialist tool ---


@tool("search_docs", "Search documentation for a topic")
async def search_docs(query: str) -> str:
    return f"Documentation result for '{query}': feature X supports use case Y."


async def main() -> None:
    # --- Section 1: Create and Inspect AgentTool ---
    print("--- Section 1: Create and Inspect AgentTool ---")

    emitter = make_emitter("agent-tool-s1")

    # Create a specialist agent with its own tools and system prompt
    specialist_client = MockLLMClient(
        responses=[
            make_response("The docs say feature X supports use case Y."),
        ]
    )
    specialist = ReasoningAgent(
        name="researcher",
        llm_client=specialist_client,
        emitter=emitter,
        system_prompt="You are a documentation research specialist.",
    )

    # Wrap it as a tool — the caller sees a single "task" parameter
    agent_tool = AgentTool(
        agent=specialist,
        emitter=emitter,
        description="Delegate research questions to a documentation specialist.",
    )

    # Schema: tool name defaults to agent.name
    schema = agent_tool.schema
    assert schema.name == "researcher"
    assert schema.description == "Delegate research questions to a documentation specialist."
    print(f"  Name: {schema.name}")
    print(f"  Description: {schema.description}")

    # Single "task" string parameter
    assert schema.parameters["properties"]["task"]["type"] == "string"
    assert schema.parameters["required"] == ["task"]
    print(f"  Parameters: {list(schema.parameters['properties'].keys())}")

    # Custom name override
    renamed_tool = AgentTool(
        agent=specialist,
        emitter=emitter,
        description="Research docs",
        name="doc_researcher",
    )
    assert renamed_tool.schema.name == "doc_researcher"
    print(f"  Custom name: {renamed_tool.schema.name}")

    # Satisfies the Tool protocol
    assert isinstance(agent_tool, Tool)
    print(f"  Tool protocol: {isinstance(agent_tool, Tool)}")

    print("✓ AgentTool wraps agent with schema exposing a single 'task' parameter")

    # --- Section 2: Direct Execution ---
    print("\n--- Section 2: Direct Execution ---")

    emitter = make_emitter("agent-tool-s2")

    # Delegate does multi-step work: tool call → final answer
    delegate_client = MockLLMClient(
        responses=[
            make_response(
                "Let me search the docs.",
                tool_calls=[ToolCall(id="tc-1", name="search_docs", arguments={"query": "feature X"})],
                stop_reason="tool_use",
            ),
            make_response("Based on the docs, feature X supports use case Y."),
        ]
    )

    delegate = ReActAgent(
        name="researcher",
        llm_client=delegate_client,
        emitter=emitter,
        system_prompt="You research documentation.",
        tools=[search_docs],
    )

    agent_tool = AgentTool(
        agent=delegate,
        emitter=emitter,
        description="Research documentation questions.",
        caller_name="coordinator",
    )

    # Execute directly — like calling any tool
    tool_result = await agent_tool.execute(task="How does feature X work?")

    # Content is the delegate's final output (RawOutputTransfer by default)
    assert tool_result.content == "Based on the docs, feature X supports use case Y."
    print(f"  Content: {tool_result.content}")

    # Metadata captures the delegate's execution details
    assert tool_result.metadata["total_steps"] == 2
    assert tool_result.metadata["termination_reason"] == "complete"
    assert "input_tokens" in tool_result.metadata["usage"]
    print(f"  Steps: {tool_result.metadata['total_steps']}")
    print(f"  Termination: {tool_result.metadata['termination_reason']}")
    print(f"  Usage: {tool_result.metadata['usage']}")

    # DelegationEvent links caller and delegate in the trace
    delegation_events = [e for e in emitter.events if isinstance(e, DelegationEvent)]
    assert len(delegation_events) == 1
    evt = delegation_events[0]
    assert evt.caller_agent == "coordinator"
    assert evt.delegate_agent == "researcher"
    assert evt.task == "How does feature X work?"
    assert evt.transfer_strategy == "RawOutputTransfer"
    print(f"  DelegationEvent: {evt.caller_agent} → {evt.delegate_agent}")
    print(f"  Task: {evt.task!r}")
    print(f"  Strategy: {evt.transfer_strategy}")

    print("✓ Direct execution returns ToolResult with content and metadata")

    # --- Section 3: Caller Agent Delegation ---
    print("\n--- Section 3: Caller Agent Delegation ---")

    emitter = make_emitter("agent-tool-s3")

    # Delegate: simple single-turn agent
    delegate_client = MockLLMClient(
        responses=[
            make_response("The capital of France is Paris."),
        ]
    )
    delegate = ReasoningAgent(
        name="knowledge-agent",
        llm_client=delegate_client,
        emitter=emitter,
        system_prompt="You answer factual questions.",
    )

    agent_tool = AgentTool(
        agent=delegate,
        emitter=emitter,
        description="Answers factual questions about geography and history.",
        caller_name="supervisor",
    )

    # Caller: uses the agent tool then synthesizes a final answer
    caller_client = MockLLMClient(
        responses=[
            # Step 1: delegate to the knowledge agent
            make_response(
                content="Let me ask my specialist.",
                tool_calls=[
                    ToolCall(
                        id="tc-k1",
                        name="knowledge-agent",
                        arguments={"task": "What is the capital of France?"},
                    )
                ],
                stop_reason="tool_use",
            ),
            # Step 2: synthesize final answer from the tool result
            make_response("Based on my research, the capital of France is Paris."),
        ]
    )

    caller = ReActAgent(
        name="supervisor",
        llm_client=caller_client,
        emitter=emitter,
        system_prompt="You are a supervisor that delegates to specialists.",
        tools=[agent_tool],
    )

    result = await caller.run("What is the capital of France?")

    # Caller's final output incorporates the delegate's answer
    assert result.output == "Based on my research, the capital of France is Paris."
    assert result.total_steps == 2
    assert result.termination_reason == "complete"
    print(f"  Caller output: {result.output}")
    print(f"  Caller steps: {result.total_steps}")

    # DelegationEvent recorded the delegation
    delegation_events = [e for e in emitter.events if isinstance(e, DelegationEvent)]
    assert len(delegation_events) == 1
    assert delegation_events[0].caller_agent == "supervisor"
    assert delegation_events[0].delegate_agent == "knowledge-agent"
    assert delegation_events[0].task == "What is the capital of France?"
    print(f"  Delegation: {delegation_events[0].caller_agent} → {delegation_events[0].delegate_agent}")

    print("✓ Caller agent delegates via AgentTool and incorporates the result")

    # --- Section 4: Multimodal Content Blocks ---
    print("\n--- Section 4: Multimodal Content Blocks ---")

    emitter = make_emitter("agent-tool-s4")

    # Delegate receives multimodal input: text task + image content block
    delegate_client = MockLLMClient(
        responses=[
            make_response("The document contains an invoice for $500."),
        ]
    )
    delegate = ReasoningAgent(
        name="document-analyzer",
        llm_client=delegate_client,
        emitter=emitter,
        system_prompt="You analyze documents from images.",
    )

    # Provide an image content block — the LLM's string task is prepended automatically
    document_image = ImageContentBlock(media_type="image/png", data="base64encodeddata")

    agent_tool = AgentTool(
        agent=delegate,
        emitter=emitter,
        description="Analyze a document image and extract information.",
        content_blocks=[document_image],
        caller_name="coordinator",
    )

    # Execute — the delegate receives [TextContentBlock(task), ImageContentBlock(image)]
    tool_result = await agent_tool.execute(task="Extract the total amount from this invoice.")

    assert tool_result.content == "The document contains an invoice for $500."
    print(f"  Content: {tool_result.content}")

    # DelegationEvent stores the original string task, not multimodal content
    delegation_events = [e for e in emitter.events if isinstance(e, DelegationEvent)]
    assert len(delegation_events) == 1
    assert delegation_events[0].task == "Extract the total amount from this invoice."
    print(f"  DelegationEvent task: {delegation_events[0].task!r}")

    # The delegate's run() received multimodal input
    # (verified by the mock — it accepted the list[ContentBlock] input)
    print("✓ AgentTool passes multimodal content blocks alongside the task")

    print("\nAll sections passed ✓")


if __name__ == "__main__":
    asyncio.run(main())
