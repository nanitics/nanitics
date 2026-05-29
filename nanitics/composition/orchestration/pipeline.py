from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from nanitics.composition.durability.models import RunCheckpoint
from nanitics.composition.durability.store import CheckpointStore
from nanitics.composition.durability.suspension import SuspendExecution
from nanitics.composition.orchestration.protocol import Step, StepResult, _sum_usage
from nanitics.composition.orchestration.workflow import Workflow
from nanitics.infrastructure.errors import NaniticsError
from nanitics.infrastructure.observability.emitter import EventEmitter
from nanitics.infrastructure.observability.events import (
    ExecutionSuspendedEvent,
    Usage,
    WorkflowStepCompleteEvent,
    WorkflowStepDefinition,
)
from nanitics.infrastructure.observability.storage import PersistentTraceStore
from nanitics.safety.cancellation import CancellationToken


class PipelineContractError(NaniticsError):
    """Raised when a stage's input or output violates its type contract.

    Attributes:
        stage_name: Name of the stage that failed validation.
        stage_index: Zero-based position of the stage in the pipeline.
        direction: Whether the violation was on ``"input"`` or ``"output"``.
        expected_type: Name of the Pydantic model that validation was checked against.
        validation_error: Details from the Pydantic validation failure.
    """

    stage_name: str
    stage_index: int
    direction: Literal["input", "output"]
    expected_type: str
    validation_error: str

    def __init__(
        self,
        *,
        stage_name: str,
        stage_index: int,
        direction: Literal["input", "output"],
        expected_type: str,
        validation_error: str,
    ) -> None:
        message = (
            f"Pipeline contract violation at stage '{stage_name}' "
            f"(index {stage_index}, {direction}): "
            f"expected {expected_type}, got validation error: {validation_error}"
        )
        super().__init__(message)
        self.stage_name = stage_name
        self.stage_index = stage_index
        self.direction = direction
        self.expected_type = expected_type
        self.validation_error = validation_error


class Stage:
    """A single stage in a pipeline with optional type validation.

    Wraps a Step and adds input/output validation against Pydantic models.
    Validation is checked before and after step execution; violations raise
    ``PipelineContractError``.

    Args:
        step: The step to execute.
        input_type: Optional Pydantic model to validate stage input against.
        output_type: Optional Pydantic model to validate stage output against.
    """

    def __init__(
        self,
        step: Step,
        *,
        input_type: type[BaseModel] | None = None,
        output_type: type[BaseModel] | None = None,
    ) -> None:
        self._step = step
        self._input_type = input_type
        self._output_type = output_type

    @property
    def name(self) -> str:
        return self._step.name

    async def execute(self, input: Any) -> StepResult:
        return await self._step.execute(input)

    def validate_input(self, input: Any, index: int) -> None:
        """Validate input against the stage's input_type schema.

        Args:
            input: The value to validate.
            index: Stage index, used in error messages.

        Raises:
            PipelineContractError: If validation fails.
        """
        if self._input_type is None:
            return
        try:
            self._input_type.model_validate(input)
        except ValidationError as e:
            raise PipelineContractError(
                stage_name=self.name,
                stage_index=index,
                direction="input",
                expected_type=self._input_type.__name__,
                validation_error=str(e),
            ) from e

    def validate_output(self, output: Any, index: int) -> None:
        """Validate output against the stage's output_type schema.

        Args:
            output: The value to validate.
            index: Stage index, used in error messages.

        Raises:
            PipelineContractError: If validation fails.
        """
        if self._output_type is None:
            return
        try:
            self._output_type.model_validate(output)
        except ValidationError as e:
            raise PipelineContractError(
                stage_name=self.name,
                stage_index=index,
                direction="output",
                expected_type=self._output_type.__name__,
                validation_error=str(e),
            ) from e


class Pipeline(Workflow):
    """Executes stages sequentially with optional type validation between stages.

    Like ``Sequential``, but each stage can declare input and output Pydantic
    models. Violations raise ``PipelineContractError`` with details about which
    stage failed and whether it was an input or output violation. The returned
    ``StepResult.usage`` is the aggregated sum across every stage's
    ``usage`` (``None`` only when every stage contributed ``None``). On a
    cancellation mid-flight, the partial aggregate of the completed stages
    is returned. On resume from a checkpoint, stage usages are reconstructed
    from the checkpoint state and folded into the final sum.

    Args:
        name: Workflow identifier.
        stages: Ordered list of stages to execute. Must contain at least one.
        emitter: Event emitter for observability.
        cancellation_token: Optional cooperative cancellation signal.
        checkpoint_store: Optional store for suspension checkpoints.
        run_id: Run identifier for checkpoint records.

    Raises:
        ValueError: If stages list is empty.
    """

    def __init__(
        self,
        *,
        name: str,
        stages: list[Stage],
        emitter: EventEmitter,
        cancellation_token: CancellationToken | None = None,
        checkpoint_store: CheckpointStore | None = None,
        run_id: str | None = None,
        trace_store: PersistentTraceStore | None = None,
    ) -> None:
        if not stages:
            raise ValueError("Pipeline requires at least one stage")
        super().__init__(
            name=name,
            emitter=emitter,
            cancellation_token=cancellation_token,
            checkpoint_store=checkpoint_store,
            run_id=run_id,
            trace_store=trace_store,
        )
        self._stages = stages

    def _workflow_type(self) -> str:
        return "pipeline"

    def _step_count(self) -> int:
        return len(self._stages)

    def _get_step_definitions(self) -> list[WorkflowStepDefinition]:
        defs = []
        for i, stage in enumerate(self._stages):
            step_type, metadata = self._classify_step(stage._step)
            if stage._input_type is not None:
                metadata["input_type"] = stage._input_type.__name__
            if stage._output_type is not None:
                metadata["output_type"] = stage._output_type.__name__
            defs.append(
                WorkflowStepDefinition(
                    name=stage.name,
                    step_type=step_type,
                    index=i,
                    metadata=metadata,
                )
            )
        return defs

    async def _run(self, input: Any, *, resume_from: RunCheckpoint | None = None) -> StepResult:
        current_input = input
        intermediate_results: dict[str, StepResult] = {}
        start_index = 0

        if resume_from is not None:
            state = resume_from.state
            start_index = state["suspended_stage_index"]
            current_input = state["last_output"]
            restored: dict[str, StepResult] = {}
            for k, d in state["completed_results"].items():
                usage_dict = d.get("usage")
                restored_usage = Usage.model_validate(usage_dict) if usage_dict is not None else None
                restored[k] = StepResult(output=d["output"], metadata=d["metadata"], usage=restored_usage)
            intermediate_results = restored
            self._emit_resumed(resume_from, self._stages[start_index].name)

        for index in range(start_index, len(self._stages)):
            stage = self._stages[index]

            if self._cancellation_token and self._cancellation_token.is_cancelled:
                return StepResult(
                    output=current_input,
                    metadata={
                        "intermediate_results": intermediate_results,
                        "terminated": "cancelled",
                        "total_steps_executed": index,
                    },
                    usage=_sum_usage(r.usage for r in intermediate_results.values()),
                )

            stage.validate_input(current_input, index)

            try:
                with self._emitter.span(stage.name):
                    bound_step = self._bind_step(stage._step)
                    step_start = time.monotonic()
                    result = await bound_step.execute(current_input)
                    step_duration_ms = int((time.monotonic() - step_start) * 1000)
            except SuspendExecution as exc:
                if self._checkpoint_store:
                    checkpoint_state: dict[str, Any] = {
                        "orchestrator_type": "pipeline",
                        "suspended_stage_index": index,
                        "completed_results": {
                            k: {
                                "output": v.output,
                                "metadata": v.metadata,
                                "usage": v.usage.model_dump() if v.usage is not None else None,
                            }
                            for k, v in intermediate_results.items()
                        },
                        "last_output": current_input,
                        "original_input": input,
                    }
                    if exc.checkpoint_data:
                        checkpoint_state["agent_checkpoint"] = exc.checkpoint_data
                    checkpoint = await self._save_checkpoint(exc, checkpoint_state)
                    self._emitter.emit(
                        ExecutionSuspendedEvent(
                            trace_id=self._emitter.trace_id,
                            span_id=self._emitter.span_id,
                            parent_span_id=self._emitter.parent_span_id,
                            suspension_id=exc.suspension_info.suspension_id,
                            suspension_type="hitl",
                            checkpoint_id=checkpoint.checkpoint_id,
                            step_name=stage.name,
                            agent_name=exc.suspension_info.agent_name,
                        )
                    )
                raise

            stage.validate_output(result.output, index)

            intermediate_results[stage.name] = result
            current_input = result.output

            self._emitter.emit(
                WorkflowStepCompleteEvent(
                    trace_id=self._emitter.trace_id,
                    span_id=self._emitter.span_id,
                    parent_span_id=self._emitter.parent_span_id,
                    workflow_name=self._name,
                    step_name=stage.name,
                    step_index=index,
                    step_duration_ms=step_duration_ms,
                    step_output=str(result.output) if result.output is not None else None,
                    step_metadata=result.metadata,
                )
            )

        return StepResult(
            output=current_input,
            metadata={
                "intermediate_results": intermediate_results,
                "total_steps_executed": len(self._stages),
            },
            usage=_sum_usage(r.usage for r in intermediate_results.values()),
        )
