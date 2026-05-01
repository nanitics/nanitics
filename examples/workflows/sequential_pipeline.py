"""Sequential and Pipeline workflows: chaining steps with optional type contracts.

Demonstrates the two simplest orchestration patterns. Sequential executes steps in order,
chaining each step's output as the next step's input. Pipeline adds Pydantic type validation
between stages, catching data contract violations early. Both use the same Step/StepResult
protocol and support the same composability primitives (FunctionStep, AgentStep, WorkflowStep).

Related guide: docs/guides/orchestration.md
"""

import asyncio

from pydantic import BaseModel

from examples.helpers import make_emitter, make_response
from nanitics import (
    AgentStep,
    FunctionStep,
    MockLLMClient,
    Pipeline,
    PipelineContractError,
    ReActAgent,
    Sequential,
    Stage,
    WorkflowStep,
)
from nanitics.infrastructure import (
    WorkflowStepCompleteEvent,
)


async def main() -> None:
    # --- Section 1: Sequential with FunctionSteps ---
    print("--- Section 1: Sequential with FunctionSteps ---")

    # Three pure-function steps chained together: uppercase → add prefix → add suffix.
    # Each step receives the previous step's output as input.

    async def uppercase(text: str) -> str:
        return text.upper()

    async def add_prefix(text: str) -> str:
        return f"[PROCESSED] {text}"

    async def add_suffix(text: str) -> str:
        return f"{text} (complete)"

    emitter = make_emitter("seq-s1")

    workflow = Sequential(
        name="text-pipeline",
        steps=[
            FunctionStep("uppercase", uppercase),
            FunctionStep("add-prefix", add_prefix),
            FunctionStep("add-suffix", add_suffix),
        ],
        emitter=emitter,
    )

    result = await workflow.execute("hello world")

    # Final output is the composed transformation
    assert result.output == "[PROCESSED] HELLO WORLD (complete)", f"Got: {result.output}"

    # Intermediate results map step name → StepResult
    intermediates = result.metadata["intermediate_results"]
    assert len(intermediates) == 3
    assert intermediates["uppercase"].output == "HELLO WORLD"
    assert intermediates["add-prefix"].output == "[PROCESSED] HELLO WORLD"
    assert intermediates["add-suffix"].output == "[PROCESSED] HELLO WORLD (complete)"

    # Step count
    assert result.metadata["total_steps_executed"] == 3

    # Events: WorkflowStepCompleteEvent emitted for each step
    step_events = [e for e in emitter.events if isinstance(e, WorkflowStepCompleteEvent)]
    assert len(step_events) == 3, f"Expected 3 step events, got {len(step_events)}"
    assert step_events[0].step_name == "uppercase"
    assert step_events[0].step_index == 0
    assert step_events[1].step_name == "add-prefix"
    assert step_events[2].step_name == "add-suffix"

    print("  Input:  'hello world'")
    print(f"  Output: '{result.output}'")
    print(f"  Steps executed: {result.metadata['total_steps_executed']}")
    print("  Intermediate outputs:")
    for name, step_result in intermediates.items():
        print(f"    {name}: '{step_result.output}'")
    print(f"  WorkflowStepCompleteEvents: {len(step_events)}")
    print("✓ Sequential chains function outputs — each step feeds the next")

    # --- Section 2: Sequential with AgentSteps ---
    print("\n--- Section 2: Sequential with AgentSteps ---")

    # Two agents in sequence: a researcher produces findings, a writer summarizes them.
    # AgentStep converts input to string, runs the agent, wraps output in StepResult.

    researcher_client = MockLLMClient(
        responses=[
            make_response("Key findings: Python 3.13 introduced free-threading and JIT compilation."),
        ]
    )
    writer_response = (
        "Summary: Python 3.13 brings two major features — free-threading "
        "for true parallelism and an experimental JIT compiler for performance."
    )
    writer_client = MockLLMClient(
        responses=[
            make_response(writer_response),
        ]
    )

    emitter = make_emitter("seq-s2")

    researcher = ReActAgent(
        name="researcher",
        llm_client=researcher_client,
        emitter=emitter,
        system_prompt="You are a research assistant. Provide key findings on the given topic.",
        tools=[],
    )
    writer = ReActAgent(
        name="writer",
        llm_client=writer_client,
        emitter=emitter,
        system_prompt="You are a technical writer. Summarize the research findings provided.",
        tools=[],
    )

    workflow = Sequential(
        name="research-pipeline",
        steps=[
            AgentStep(researcher),
            AgentStep(writer),
        ],
        emitter=emitter,
    )

    result = await workflow.execute("What's new in Python 3.13?")

    # Final output is the writer's response
    assert result.output == writer_response

    # Intermediate results contain both agent steps
    intermediates = result.metadata["intermediate_results"]
    assert len(intermediates) == 2
    assert "Python 3.13" in intermediates["researcher"].output

    # AgentStep metadata includes agent-specific fields
    researcher_meta = intermediates["researcher"].metadata
    assert researcher_meta["total_steps"] == 1
    assert researcher_meta["termination_reason"] == "complete"
    assert "input_tokens" in researcher_meta["usage"]

    writer_meta = intermediates["writer"].metadata
    assert writer_meta["total_steps"] == 1
    assert writer_meta["termination_reason"] == "complete"

    print(f"  Researcher output: '{intermediates['researcher'].output}'")
    print(f"  Writer output:     '{result.output}'")
    print(f"  Researcher steps: {researcher_meta['total_steps']}, termination: {researcher_meta['termination_reason']}")
    print(f"  Writer steps:     {writer_meta['total_steps']}, termination: {writer_meta['termination_reason']}")
    print("✓ AgentSteps chain agent outputs — researcher findings flow to writer")

    # --- Section 2b: AgentStep with Structured Output ---
    print("\n--- Section 2b: AgentStep with Structured Output ---")

    # When an agent has output_schema, AgentStep forwards the parsed Pydantic model
    # as step output. The text response is preserved in metadata["text_output"].
    # This enables typed data flow through Sequential and Pipeline.

    class ResearchFindings(BaseModel):
        topic: str
        key_points: list[str]

    findings_json = '{"topic": "Python async", "key_points": ["asyncio", "coroutines", "event loop"]}'
    structured_client = MockLLMClient(
        responses=[
            # First response: the agent's main text completion
            make_response("I researched Python async. Key points: asyncio, coroutines, event loop."),
            # Second response: structured output call — content must be valid JSON for the schema
            make_response(findings_json),
        ]
    )

    emitter = make_emitter("seq-s2b")

    structured_researcher = ReActAgent(
        name="structured-researcher",
        llm_client=structured_client,
        emitter=emitter,
        system_prompt="Research the given topic and produce structured findings.",
        tools=[],
        output_schema=ResearchFindings,
    )

    async def summarize_findings(findings: ResearchFindings) -> str:
        return f"Found {len(findings.key_points)} points about {findings.topic}: {', '.join(findings.key_points)}"

    workflow = Sequential(
        name="structured-pipeline",
        steps=[
            AgentStep(structured_researcher),
            FunctionStep("summarize", summarize_findings),
        ],
        emitter=emitter,
    )

    result = await workflow.execute("Tell me about Python async")

    # Final output is the summarize function's string
    assert result.output == "Found 3 points about Python async: asyncio, coroutines, event loop"

    # Intermediate: the agent step forwarded the parsed model, not the text
    intermediates = result.metadata["intermediate_results"]
    agent_result = intermediates["structured-researcher"]
    assert isinstance(agent_result.output, ResearchFindings)
    assert agent_result.output.topic == "Python async"
    assert agent_result.output.key_points == ["asyncio", "coroutines", "event loop"]

    # The text response is preserved in metadata for observability
    assert agent_result.metadata["text_output"] == findings_json

    print(f"  Agent parsed output: {agent_result.output}")
    print(f"  Agent text output:   '{agent_result.metadata['text_output']}'")
    print(f"  Final output:        '{result.output}'")
    print("✓ AgentStep forwards parsed Pydantic model — typed data flows through pipeline")

    # --- Section 3: Pipeline with Type Contracts ---
    print("\n--- Section 3: Pipeline with Type Contracts ---")

    # Pipeline validates data between stages using Pydantic models.
    # Stage 1 produces RawData → Stage 2 expects RawData, produces ProcessedData.

    class RawData(BaseModel):
        text: str
        source: str

    class ProcessedData(BaseModel):
        summary: str
        word_count: int

    async def extract(input: dict) -> dict:
        return {"text": input["text"], "source": input["source"]}

    async def process(raw: dict) -> dict:
        text = raw["text"]
        words = text.split()
        return {"summary": f"Processed: {text[:50]}", "word_count": len(words)}

    emitter = make_emitter("pipe-s3")

    workflow = Pipeline(
        name="typed-pipeline",
        stages=[
            Stage(
                FunctionStep("extract", extract),
                output_type=RawData,
            ),
            Stage(
                FunctionStep("process", process),
                input_type=RawData,
                output_type=ProcessedData,
            ),
        ],
        emitter=emitter,
    )

    result = await workflow.execute({"text": "The quick brown fox jumps over the lazy dog", "source": "test"})

    assert result.output == {"summary": "Processed: The quick brown fox jumps over the lazy dog", "word_count": 9}
    assert result.metadata["total_steps_executed"] == 2

    print("  Input:  text='The quick brown fox...' source='test'")
    print(f"  Output: {result.output}")
    print(f"  Stages executed: {result.metadata['total_steps_executed']}")
    print("✓ Pipeline validates data contracts between stages — correct data passes through")

    # --- Section 4: Pipeline Contract Violation ---
    print("\n--- Section 4: Pipeline Contract Violation ---")

    # When a stage produces output that doesn't match the next stage's input_type,
    # PipelineContractError is raised with precise diagnostics.

    async def bad_extract(input: dict) -> dict:
        # Missing 'word_count' — violates ProcessedData contract
        return {"summary": "incomplete data"}

    emitter = make_emitter("pipe-s4")

    bad_workflow = Pipeline(
        name="failing-pipeline",
        stages=[
            Stage(
                FunctionStep("bad-extract", bad_extract),
                output_type=ProcessedData,  # Output must match ProcessedData
            ),
            Stage(
                FunctionStep("process", process),
                input_type=ProcessedData,
            ),
        ],
        emitter=emitter,
    )

    try:
        await bad_workflow.execute({"text": "test", "source": "test"})
        assert False, "Should have raised PipelineContractError"
    except PipelineContractError as e:
        assert e.stage_name == "bad-extract"
        assert e.stage_index == 0
        assert e.direction == "output"
        assert e.expected_type == "ProcessedData"
        assert "word_count" in e.validation_error

        print("  Caught PipelineContractError:")
        print(f"    stage_name:    '{e.stage_name}'")
        print(f"    stage_index:   {e.stage_index}")
        print(f"    direction:     '{e.direction}'")
        print(f"    expected_type: '{e.expected_type}'")
        print(f"    validation:    ...{e.validation_error[:80]}...")
    print("✓ Contract violations caught at stage boundary with precise diagnostics")

    # --- Section 5: Composability with WorkflowStep ---
    print("\n--- Section 5: Composability with WorkflowStep ---")

    # WorkflowStep wraps a workflow as a step, enabling nesting.
    # Outer Sequential: prepare → [inner Sequential: double → exclaim] → wrap

    async def prepare(text: str) -> str:
        return text.strip().lower()

    async def double(text: str) -> str:
        return f"{text} {text}"

    async def exclaim(text: str) -> str:
        return f"{text}!"

    async def wrap(text: str) -> str:
        return f"<<{text}>>"

    emitter = make_emitter("compose-s5")

    inner_workflow = Sequential(
        name="inner-transform",
        steps=[
            FunctionStep("double", double),
            FunctionStep("exclaim", exclaim),
        ],
        emitter=emitter,
    )

    outer_workflow = Sequential(
        name="outer-pipeline",
        steps=[
            FunctionStep("prepare", prepare),
            WorkflowStep(inner_workflow),
            FunctionStep("wrap", wrap),
        ],
        emitter=emitter,
    )

    result = await outer_workflow.execute("  Hello  ")

    # prepare: "  Hello  " → "hello"
    # inner (double): "hello" → "hello hello"
    # inner (exclaim): "hello hello" → "hello hello!"
    # wrap: "hello hello!" → "<<hello hello!>>"
    assert result.output == "<<hello hello!>>", f"Got: {result.output}"

    # Outer workflow sees 3 steps (prepare, inner-transform, wrap)
    assert result.metadata["total_steps_executed"] == 3

    # Intermediate results include the WorkflowStep
    intermediates = result.metadata["intermediate_results"]
    assert intermediates["prepare"].output == "hello"
    assert intermediates["inner-transform"].output == "hello hello!"
    assert intermediates["wrap"].output == "<<hello hello!>>"

    # Inner workflow's own intermediates are nested in the WorkflowStep's metadata
    inner_intermediates = intermediates["inner-transform"].metadata["intermediate_results"]
    assert inner_intermediates["double"].output == "hello hello"
    assert inner_intermediates["exclaim"].output == "hello hello!"

    print("  Input:  '  Hello  '")
    print(f"  After prepare:         '{intermediates['prepare'].output}'")
    print(f"  After inner (double):  '{inner_intermediates['double'].output}'")
    print(f"  After inner (exclaim): '{inner_intermediates['exclaim'].output}'")
    print(f"  After wrap:            '{result.output}'")
    print(f"  Outer steps counted: {result.metadata['total_steps_executed']}")
    print("✓ WorkflowStep nests workflows — inner Sequential runs as a single outer step")


if __name__ == "__main__":
    asyncio.run(main())
