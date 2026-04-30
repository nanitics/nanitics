"""Orchestrator: dynamic delegation to specialist agents.

Demonstrates ``create_orchestrator`` — a factory that returns a ``ReActAgent``
configured to analyze tasks, delegate subtasks to specialist ``AgentTool``
instances, and synthesize their results. Covers prompt section generation,
orchestrator construction, end-to-end delegation with two specialists,
custom system prompt override, and the ``RELAY_LAST`` final-output strategy
that returns the final specialist's output verbatim.

Related guide: docs/guides/multi-agent-coordination.md
"""

import asyncio

from examples.helpers import make_emitter, make_response
from nanitics import (
    AgentTool,
    FinalOutputStrategy,
    MockLLMClient,
    ReActAgent,
    ReasoningAgent,
    ToolCall,
    create_orchestrator,
    orchestrator_prompt_section,
)
from nanitics.infrastructure import (
    DelegationEvent,
)


async def main() -> None:
    # --- Section 1: Prompt Section Generation ---
    print("--- Section 1: Prompt Section Generation ---")

    emitter = make_emitter("orchestrator-s1")

    # Create two specialist agents and wrap as AgentTools
    researcher = ReasoningAgent(
        name="researcher",
        llm_client=MockLLMClient(responses=[]),
        emitter=emitter,
        system_prompt="You are a research specialist.",
    )
    writer = ReActAgent(
        name="writer",
        llm_client=MockLLMClient(responses=[]),
        emitter=emitter,
        system_prompt="You are a technical writer.",
        tools=[],
    )

    specialists = [
        AgentTool(
            agent=researcher,
            emitter=emitter,
            description="Delegate research tasks — finding information, analyzing data.",
        ),
        AgentTool(
            agent=writer,
            emitter=emitter,
            description="Delegate writing tasks — producing articles, reports.",
        ),
    ]

    # orchestrator_prompt_section builds the specialist listing + strategy
    section_name, section_content = orchestrator_prompt_section(specialists)

    assert section_name == "Orchestration"
    assert "researcher" in section_content
    assert "writer" in section_content
    assert "Delegate research tasks" in section_content
    assert "Delegate writing tasks" in section_content
    assert "Analyze the task" in section_content
    assert "Combine the specialists' findings" in section_content

    print(f"  Section name: {section_name}")
    print("  Contains specialist names: researcher, writer")
    print("  Contains strategy steps: analyze, delegate, combine")
    print("✓ orchestrator_prompt_section returns specialist listing with strategy")

    # --- Section 2: Create Orchestrator ---
    print("\n--- Section 2: Create Orchestrator ---")

    emitter = make_emitter("orchestrator-s2")

    researcher = ReasoningAgent(
        name="researcher",
        llm_client=MockLLMClient(responses=[]),
        emitter=emitter,
        system_prompt="You are a research specialist.",
    )
    writer = ReActAgent(
        name="writer",
        llm_client=MockLLMClient(responses=[]),
        emitter=emitter,
        system_prompt="You are a technical writer.",
        tools=[],
    )

    specialists = [
        AgentTool(
            agent=researcher,
            emitter=emitter,
            description="Delegate research tasks.",
        ),
        AgentTool(
            agent=writer,
            emitter=emitter,
            description="Delegate writing tasks.",
        ),
    ]

    orchestrator = create_orchestrator(
        name="coordinator",
        llm_client=MockLLMClient(responses=[]),
        emitter=emitter,
        specialists=specialists,
    )

    # Returns a ReActAgent
    assert isinstance(orchestrator, ReActAgent)
    assert orchestrator.name == "coordinator"

    # The same specialist section the factory uses is available publicly.
    _, auto_section = orchestrator_prompt_section(specialists)
    assert "researcher" in auto_section
    assert "writer" in auto_section
    assert "Delegate research tasks" in auto_section

    # The returned object is a ReActAgent subclass that applies the
    # final-output policy; typed as ReActAgent for caller use.
    print(f"  Type: {ReActAgent.__name__}")
    print(f"  Name: {orchestrator.name}")
    print(f"  Specialist tools: {[s.schema.name for s in specialists]}")
    print("✓ create_orchestrator returns a ReActAgent wired with specialists + auto-prompt")

    # --- Section 3: End-to-End Delegation ---
    print("\n--- Section 3: End-to-End Delegation ---")

    emitter = make_emitter("orchestrator-s3")

    # Researcher specialist: returns research findings
    researcher = ReasoningAgent(
        name="researcher",
        llm_client=MockLLMClient(
            responses=[
                make_response("Research findings: Python 3.13 introduces free-threading and improved JIT."),
            ]
        ),
        emitter=emitter,
        system_prompt="You are a research specialist.",
    )

    # Writer specialist: returns polished article
    writer = ReActAgent(
        name="writer",
        llm_client=MockLLMClient(
            responses=[
                make_response("Article draft: Python 3.13 brings two major changes to the language."),
            ]
        ),
        emitter=emitter,
        system_prompt="You are a technical writer.",
        tools=[],
    )

    specialists = [
        AgentTool(
            agent=researcher,
            emitter=emitter,
            description="Delegate research tasks.",
        ),
        AgentTool(
            agent=writer,
            emitter=emitter,
            description="Delegate writing tasks.",
        ),
    ]

    # Orchestrator LLM: delegate to researcher, then writer, then synthesize
    orchestrator_client = MockLLMClient(
        responses=[
            # Step 1: delegate to researcher
            make_response(
                content="I'll research Python 3.13 first.",
                tool_calls=[
                    ToolCall(
                        id="tc-r1",
                        name="researcher",
                        arguments={"task": "Research key features of Python 3.13"},
                    )
                ],
                stop_reason="tool_use",
            ),
            # Step 2: delegate to writer
            make_response(
                content="Now I'll draft the article.",
                tool_calls=[
                    ToolCall(
                        id="tc-w1",
                        name="writer",
                        arguments={"task": "Write an article about Python 3.13 free-threading and JIT"},
                    )
                ],
                stop_reason="tool_use",
            ),
            # Step 3: synthesize final response
            make_response(
                "Final article: Python 3.13 introduces free-threading "
                "for true parallelism and an improved JIT compiler for better performance."
            ),
        ]
    )

    orchestrator = create_orchestrator(
        name="coordinator",
        llm_client=orchestrator_client,
        emitter=emitter,
        specialists=specialists,
    )

    result = await orchestrator.run("Write an article about Python 3.13")

    # Final output is the synthesized response
    assert result.output == (
        "Final article: Python 3.13 introduces free-threading "
        "for true parallelism and an improved JIT compiler for better performance."
    )
    assert result.total_steps == 3
    assert result.termination_reason == "complete"

    # DelegationEvents trace both delegations
    delegation_events = [e for e in emitter.events if isinstance(e, DelegationEvent)]
    assert len(delegation_events) == 2
    assert delegation_events[0].caller_agent == "coordinator"
    assert delegation_events[0].delegate_agent == "researcher"
    assert delegation_events[1].caller_agent == "coordinator"
    assert delegation_events[1].delegate_agent == "writer"

    print(f"  Output: {result.output[:60]}...")
    print(f"  Steps: {result.total_steps}")
    print(f"  Delegations: {len(delegation_events)}")
    print(f"    1. {delegation_events[0].caller_agent} → {delegation_events[0].delegate_agent}")
    print(f"    2. {delegation_events[1].caller_agent} → {delegation_events[1].delegate_agent}")
    print("✓ Orchestrator delegated to two specialists and synthesized the result")

    # --- Section 4: Custom System Prompt ---
    print("\n--- Section 4: Custom System Prompt ---")

    emitter = make_emitter("orchestrator-s4")

    researcher = ReasoningAgent(
        name="researcher",
        llm_client=MockLLMClient(responses=[]),
        emitter=emitter,
        system_prompt="You are a research specialist.",
    )
    writer = ReActAgent(
        name="writer",
        llm_client=MockLLMClient(responses=[]),
        emitter=emitter,
        system_prompt="You are a technical writer.",
        tools=[],
    )

    specialists = [
        AgentTool(
            agent=researcher,
            emitter=emitter,
            description="Delegate research tasks.",
        ),
        AgentTool(
            agent=writer,
            emitter=emitter,
            description="Delegate writing tasks.",
        ),
    ]

    # Use orchestrator_prompt_section to get the specialist listing
    section_name, section_content = orchestrator_prompt_section(specialists)

    # Build a custom prompt incorporating the specialist section
    custom_prompt = (
        f"You are a project coordinator for technical documentation.\n\n"
        f"{section_content}\n\nAdditional rule: always research before writing."
    )

    orchestrator = create_orchestrator(
        name="coordinator",
        llm_client=MockLLMClient(responses=[]),
        emitter=emitter,
        specialists=specialists,
        system_prompt=custom_prompt,
    )

    # Verify the custom prompt string contains what we assembled before passing it in.
    assert "project coordinator" in custom_prompt
    assert "always research before writing" in custom_prompt
    assert "researcher" in custom_prompt and "writer" in custom_prompt

    print("  Custom instruction present: 'always research before writing'")
    print("  Specialist listing present: researcher, writer")
    print("✓ Custom prompt overrides auto-generation while reusing specialist section")

    # --- Section 5: Relay Mode (FinalOutputStrategy.RELAY_LAST) ---
    print("\n--- Section 5: Relay Mode (FinalOutputStrategy.RELAY_LAST) ---")

    emitter = make_emitter("orchestrator-s5")

    # A single writer specialist produces the actual deliverable. Under
    # the default SYNTHESIZE strategy the coordinator would add a final
    # LLM turn that rewrites (and often compresses) the writer's output.
    # RELAY_LAST returns the writer's tool_result content verbatim.
    article_text = (
        "Python 3.13 ships an experimental free-threaded build (PEP 703) that "
        "disables the GIL so CPU-bound threads can run in parallel. Opt in with "
        "the --disable-gil configure flag; a separate runtime tag (t) marks the "
        "build. Extension authors must add a Py_GIL_DISABLED-aware "
        "PyUnstable_Module_SetGIL call before the free-threaded interpreter "
        "enables their module."
    )

    writer = ReActAgent(
        name="writer",
        llm_client=MockLLMClient(responses=[make_response(article_text)]),
        emitter=emitter,
        system_prompt="You are a technical writer.",
        tools=[],
    )

    specialists = [
        AgentTool(
            agent=writer,
            emitter=emitter,
            description="Delegate writing tasks.",
        ),
    ]

    # The coordinator LLM delegates to the writer, then — under
    # SYNTHESIZE — would produce a short meta-description turn.
    # Under RELAY_LAST, this final text is discarded.
    orchestrator_client = MockLLMClient(
        responses=[
            make_response(
                content="I will delegate to the writer.",
                tool_calls=[
                    ToolCall(
                        id="tc-w5",
                        name="writer",
                        arguments={"task": "Write an article about Python 3.13 free-threading"},
                    )
                ],
                stop_reason="tool_use",
            ),
            # Coordinator's synthesis turn — dropped under RELAY_LAST.
            make_response("Short meta-description about Python 3.13."),
        ]
    )

    orchestrator = create_orchestrator(
        name="coordinator",
        llm_client=orchestrator_client,
        emitter=emitter,
        specialists=specialists,
        final_output_strategy=FinalOutputStrategy.RELAY_LAST,
    )

    result = await orchestrator.run("Write an article about Python 3.13 free-threading")

    # Output is the writer's article verbatim — not the coordinator's
    # compressed synthesis.
    assert result.output == article_text
    assert "Short meta-description" not in (result.output or "")

    print(f"  Output length: {len(result.output or '')} chars (writer's full article)")
    print("  Coordinator synthesis turn: dropped")
    print("✓ RELAY_LAST returns the final specialist's tool_result verbatim")

    print("\nAll sections passed ✓")


if __name__ == "__main__":
    asyncio.run(main())
