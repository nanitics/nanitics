"""Pipeline: typed Stage contracts enforced on both sides of each stage.

Validates the :class:`Pipeline` workflow and the :class:`Stage` contract
mechanism. Three scenarios are covered:

1. **Contract-satisfied round-trip.** A three-stage pipeline —
   ``extractor`` (real :class:`ReActAgent` with ``output_schema`` so its
   output is a Pydantic model) → ``enricher`` (``FunctionStep``) →
   ``publisher`` (``FunctionStep``). Input/output types are declared on
   every stage boundary. A distinctive token (``NANITICS-PIPE-5B2E``) is
   carried through the pipeline and must survive end-to-end, pinning
   **stage-to-stage input piping with typed contracts intact**.

2. **Output-contract violation.** A ``FunctionStep`` that deliberately
   omits a required field; the stage declares the model as ``output_type``.
   The violation is raised as :class:`PipelineContractError` with
   ``direction == "output"`` and ``stage_index == 0``.

3. **Input-contract violation.** Downstream stage declares ``input_type``
   that the (raw dict) pipeline input doesn't satisfy; raised with
   ``direction == "input"``.

Acceptance criteria (round-trip):
  - ``AgentStartEvent`` emitted for the extractor — proves the real agent
    ran inside the pipeline.
  - Exactly three ``WorkflowStepCompleteEvent`` events with ``step_index``
    values ``[0, 1, 2]`` and step names ``[extractor, enricher,
    publisher]`` — pins ordering.
  - ``result.metadata["total_steps_executed"] == 3``.
  - ``result.metadata["intermediate_results"]["extractor"].output`` is a
    ``ProductBrief`` Pydantic instance — proves the structured-output
    contract was honoured.
  - The distinctive token appears in the final output string — proves
    round-trip across all three stages (a broken pipe would drop the
    token or fail the contract).

Acceptance criteria (output-violation):
  - Raises :class:`PipelineContractError`.
  - ``error.stage_name == "bad-extractor"``, ``stage_index == 0``,
    ``direction == "output"``, ``expected_type == "ProductBrief"``.

Acceptance criteria (input-violation):
  - Raises :class:`PipelineContractError`.
  - ``error.stage_index == 0``, ``direction == "input"``,
    ``expected_type == "ProductBrief"``.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from nanitics import (
    AgentStep,
    FunctionStep,
    InMemoryEmitter,
    Pipeline,
    PipelineContractError,
    ReActAgent,
    Stage,
)
from nanitics.infrastructure import (
    AgentStartEvent,
    WorkflowStepCompleteEvent,
)
from validation.helpers import (
    assert_trace_contains,
    make_llm_client,
    run_with_retry,
)

# Distinctive token carried through the whole pipeline. A broken pipe would
# drop it before the publisher stage.
PIPE_TOKEN = "NANITICS-PIPE-5B2E"


class ProductBrief(BaseModel):
    name: str
    tagline: str
    trace_token: str


class PublishedBrief(BaseModel):
    headline: str
    body: str


async def _enrich(brief: ProductBrief) -> ProductBrief:
    return ProductBrief(
        name=brief.name,
        tagline=f"{brief.tagline} — enriched",
        trace_token=brief.trace_token,
    )


async def _publish(brief: ProductBrief) -> PublishedBrief:
    return PublishedBrief(
        headline=f"Announcing {brief.name}",
        body=f"{brief.tagline} [{brief.trace_token}]",
    )


@pytest.mark.quick
async def test_pipeline_contract_round_trip(traced_emitter: InMemoryEmitter) -> None:
    extractor = ReActAgent(
        name="extractor",
        llm_client=make_llm_client("anthropic"),
        emitter=traced_emitter,
        system_prompt=(
            "You extract product briefs from a raw description. Produce a JSON object with keys "
            "'name' (the product name), 'tagline' (a one-sentence tagline), and 'trace_token' "
            f"(which MUST be the exact string {PIPE_TOKEN!r} — downstream systems depend on it)."
        ),
        tools=[],
        max_iterations=2,
        output_schema=ProductBrief,
    )

    workflow = Pipeline(
        name="product-pipeline",
        stages=[
            Stage(AgentStep(extractor), output_type=ProductBrief),
            Stage(FunctionStep("enricher", _enrich), input_type=ProductBrief, output_type=ProductBrief),
            Stage(FunctionStep("publisher", _publish), input_type=ProductBrief, output_type=PublishedBrief),
        ],
        emitter=traced_emitter,
    )

    raw_description = (
        f"Launch brief: 'Aurora', a managed vector database for retrieval apps. "
        f"IMPORTANT: set trace_token to the exact string {PIPE_TOKEN!r}."
    )
    result = await run_with_retry(
        lambda: workflow.execute(raw_description),
        max_attempts=2,
    )

    assert_trace_contains(
        traced_emitter,
        AgentStartEvent,
        predicate=lambda e: e.agent_name == "extractor",
    )

    step_events = [e for e in traced_emitter.events if isinstance(e, WorkflowStepCompleteEvent)]
    assert len(step_events) == 3, f"Expected 3 WorkflowStepCompleteEvent, got: {len(step_events)}"
    assert [e.step_index for e in step_events] == [0, 1, 2], (
        f"Expected step_index sequence [0, 1, 2], got: {[e.step_index for e in step_events]}"
    )
    assert [e.step_name for e in step_events] == ["extractor", "enricher", "publisher"], (
        f"Expected stage ordering [extractor, enricher, publisher], got: {[e.step_name for e in step_events]}"
    )

    assert result.metadata["total_steps_executed"] == 3, (
        f"Expected total_steps_executed == 3, got: {result.metadata['total_steps_executed']}"
    )

    intermediates = result.metadata["intermediate_results"]
    extractor_output = intermediates["extractor"].output
    assert isinstance(extractor_output, ProductBrief), (
        f"Expected extractor output to be a ProductBrief instance (structured-output contract), "
        f"got: {type(extractor_output).__name__}"
    )
    assert extractor_output.trace_token == PIPE_TOKEN, (
        f"Expected extractor to emit trace_token={PIPE_TOKEN!r}, got: {extractor_output.trace_token!r}"
    )

    assert isinstance(result.output, PublishedBrief), (
        f"Expected final output to be PublishedBrief, got: {type(result.output).__name__}"
    )
    assert PIPE_TOKEN in result.output.body, (
        f"Expected token {PIPE_TOKEN!r} to survive the full pipeline; got final body: {result.output.body!r}"
    )
    assert "enriched" in result.output.body, (
        f"Expected enricher output to reach publisher (tagline contains 'enriched'); "
        f"got final body: {result.output.body!r}"
    )


@pytest.mark.quick
async def test_pipeline_contract_violation_on_output(traced_emitter: InMemoryEmitter) -> None:
    async def bad_extract(_input: str) -> dict:
        # Missing required fields tagline and trace_token — ProductBrief rejects it.
        return {"name": "Aurora"}

    workflow = Pipeline(
        name="bad-output-pipeline",
        stages=[
            Stage(FunctionStep("bad-extractor", bad_extract), output_type=ProductBrief),
            Stage(FunctionStep("enricher", _enrich), input_type=ProductBrief),
        ],
        emitter=traced_emitter,
    )

    with pytest.raises(PipelineContractError) as exc_info:
        await workflow.execute("irrelevant")

    error = exc_info.value
    assert error.stage_name == "bad-extractor", f"Expected stage_name == 'bad-extractor', got: {error.stage_name!r}"
    assert error.stage_index == 0, f"Expected stage_index == 0, got: {error.stage_index}"
    assert error.direction == "output", f"Expected direction == 'output', got: {error.direction!r}"
    assert error.expected_type == "ProductBrief", (
        f"Expected expected_type == 'ProductBrief', got: {error.expected_type!r}"
    )


@pytest.mark.quick
async def test_pipeline_contract_violation_on_input(traced_emitter: InMemoryEmitter) -> None:
    async def passthrough(value: str) -> str:
        return value

    workflow = Pipeline(
        name="bad-input-pipeline",
        stages=[
            # Declares input_type=ProductBrief, but the pipeline input is a raw str.
            Stage(FunctionStep("strict-entry", passthrough), input_type=ProductBrief),
        ],
        emitter=traced_emitter,
    )

    with pytest.raises(PipelineContractError) as exc_info:
        await workflow.execute("not a product brief")

    error = exc_info.value
    assert error.stage_index == 0, f"Expected stage_index == 0, got: {error.stage_index}"
    assert error.direction == "input", f"Expected direction == 'input', got: {error.direction!r}"
    assert error.expected_type == "ProductBrief", (
        f"Expected expected_type == 'ProductBrief', got: {error.expected_type!r}"
    )
