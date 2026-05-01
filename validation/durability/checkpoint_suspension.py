"""Real-provider validation for ``CheckpointStore``, ``SuspendExecution``, and ``RunCheckpoint``.

Drives a ``Sequential`` workflow containing three ``FunctionStep`` nodes
through a suspension/resume cycle backed by ``InMemoryCheckpointStore``.
The first step completes, the second raises ``SuspendExecution`` on its
first invocation, and the workflow reconstructs from the saved
``RunCheckpoint`` and resumes — asserting that the pre-suspension step
is NOT re-executed.

A real Anthropic call runs in the prelude so the validation suite exercises
an actual LLM round-trip (per the validation-suite charter) before driving
the durability-only assertions below.

Acceptance criteria:
  - First-run suspension: ``SuspendExecution`` propagates out of
    ``workflow.execute()``; a ``RunCheckpoint`` is persisted keyed on the
    configured ``run_id``; its ``checkpoint_type == "orchestration"``,
    ``state["suspended_step_index"] == 1``, and
    ``state["completed_results"]`` contains only the pre-suspension step.
  - ``CheckpointSavedEvent`` and ``ExecutionSuspendedEvent`` both emitted
    on the first run; ``suspension_info.suspension_id`` matches on both.
  - Resume: a new ``Sequential`` (simulating a fresh process) calls
    ``execute(input, resume_from=checkpoint)`` and produces the correct
    final output threaded through all three steps.
  - No re-execution of pre-suspension steps on resume: the resume-run
    emitter contains NO ``WorkflowStepCompleteEvent`` for the
    pre-suspension step (proves completed steps were skipped — the
    distinguishing assertion).
  - ``ExecutionResumedEvent`` emitted on the resume run with
    ``checkpoint_id`` matching the loaded checkpoint.
  - Prelude: a real Anthropic ``generate()`` call produces positive usage
    so the script carries a real-provider assertion.
"""

from __future__ import annotations

import pytest

from nanitics import (
    CheckpointStore,
    FunctionStep,
    InMemoryCheckpointStore,
    InMemoryEmitter,
    Message,
    RunCheckpoint,
    Sequential,
    SuspendExecution,
    SuspensionInfo,
)
from nanitics.infrastructure import (
    CheckpointSavedEvent,
    ExecutionResumedEvent,
    ExecutionSuspendedEvent,
    WorkflowStepCompleteEvent,
)
from validation.helpers import (
    assert_trace_contains,
    make_llm_client,
    run_with_retry,
)


@pytest.mark.quick
async def test_checkpoint_suspend_and_resume(traced_emitter: InMemoryEmitter) -> None:
    # --- Prelude: real Anthropic round-trip so the script exercises a real provider. ---
    client = make_llm_client("anthropic")
    prelude = await run_with_retry(
        lambda: client.generate(
            system_prompt="Reply with a single short word.",
            messages=[Message(role="user", content="Say OK.")],
        ),
        max_attempts=2,
    )
    assert prelude.usage.total_tokens > 0, "Prelude real-provider call must produce positive usage."

    # --- Checkpoint store & workflow that suspends at step 2 on the first run ---
    store: CheckpointStore = InMemoryCheckpointStore()
    # Mutable counter shared by the suspending step on both first and resume runs.
    step_2_calls: dict[str, int] = {"count": 0}
    # Track every execution so we can prove step_1 is NOT re-run on resume.
    executed: list[str] = []

    async def step_1(value: str) -> str:
        executed.append("step_1")
        return f"prepared:{value}"

    async def step_2(value: str) -> str:
        step_2_calls["count"] += 1
        if step_2_calls["count"] == 1:
            raise SuspendExecution(
                suspension_info=SuspensionInfo(
                    suspension_id="sus-validation-98",
                    request_id="req-validation-98",
                    request_type="approval",
                    prompt="Approve the prepared value?",
                    agent_name="reviewer",
                ),
            )
        executed.append("step_2")
        return f"approved:{value}"

    async def step_3(value: str) -> str:
        executed.append("step_3")
        return f"final:{value}"

    run_id = "run-validation-98"
    workflow = Sequential(
        name="suspend-pipeline",
        steps=[
            FunctionStep("step_1", step_1),
            FunctionStep("step_2", step_2),
            FunctionStep("step_3", step_3),
        ],
        emitter=traced_emitter,
        checkpoint_store=store,
        run_id=run_id,
    )

    # --- First run: step_1 completes, step_2 suspends, step_3 is never reached ---
    with pytest.raises(SuspendExecution):
        await workflow.execute("raw")

    assert executed == ["step_1"], f"Only step_1 should run before suspension; got {executed!r}"
    assert step_2_calls["count"] == 1

    # --- Checkpoint persisted & structured correctly ---
    checkpoint = await store.load(run_id)
    assert checkpoint is not None, "Expected a checkpoint to be persisted on suspension."
    assert isinstance(checkpoint, RunCheckpoint)
    assert checkpoint.run_id == run_id
    assert checkpoint.checkpoint_type == "orchestration"
    assert checkpoint.state["suspended_step_index"] == 1, (
        f"Expected suspended_step_index == 1 (step_2); got {checkpoint.state['suspended_step_index']}"
    )
    completed_results = checkpoint.state["completed_results"]
    assert "step_1" in completed_results, f"step_1 must be in completed_results; got {list(completed_results)}"
    assert "step_2" not in completed_results, (
        "step_2 must NOT be in completed_results — it suspended before producing output."
    )
    assert checkpoint.suspension_info.suspension_id == "sus-validation-98"
    assert checkpoint.suspension_info.prompt == "Approve the prepared value?"

    # --- First-run events: checkpoint saved + execution suspended, IDs correlate ---
    saved_events = [e for e in traced_emitter.events if isinstance(e, CheckpointSavedEvent)]
    suspended_events = [e for e in traced_emitter.events if isinstance(e, ExecutionSuspendedEvent)]
    assert len(saved_events) == 1, f"Expected 1 CheckpointSavedEvent, got {len(saved_events)}"
    assert len(suspended_events) == 1, f"Expected 1 ExecutionSuspendedEvent, got {len(suspended_events)}"
    assert saved_events[0].checkpoint_id == checkpoint.checkpoint_id
    assert saved_events[0].run_id == run_id
    assert suspended_events[0].suspension_id == "sus-validation-98"
    assert suspended_events[0].checkpoint_id == checkpoint.checkpoint_id

    # --- Resume: a fresh emitter + fresh Sequential simulate resuming in a new process ---
    executed.clear()
    resume_emitter = InMemoryEmitter(trace_id=f"{traced_emitter.trace_id}::resume")
    workflow_resume = Sequential(
        name="suspend-pipeline",
        steps=[
            FunctionStep("step_1", step_1),
            FunctionStep("step_2", step_2),
            FunctionStep("step_3", step_3),
        ],
        emitter=resume_emitter,
        checkpoint_store=store,
        run_id=run_id,
    )

    result = await workflow_resume.execute("raw", resume_from=checkpoint)

    # --- Distinguishing assertion: step_1 MUST NOT run again on resume ---
    assert "step_1" not in executed, (
        f"step_1 must be skipped on resume (it completed pre-suspension); executed={executed!r}"
    )
    assert "step_2" in executed, f"step_2 must run on resume; executed={executed!r}"
    assert "step_3" in executed, f"step_3 must run on resume; executed={executed!r}"

    # --- Per-step event absence: the resume-run emitter holds NO WorkflowStepCompleteEvent for step_1 ---
    resume_step_events = [e for e in resume_emitter.events if isinstance(e, WorkflowStepCompleteEvent)]
    step_1_events_on_resume = [e for e in resume_step_events if e.step_name == "step_1"]
    assert step_1_events_on_resume == [], (
        f"Expected NO WorkflowStepCompleteEvent for step_1 on resume; got {step_1_events_on_resume}"
    )
    step_names_on_resume = {e.step_name for e in resume_step_events}
    assert "step_2" in step_names_on_resume
    assert "step_3" in step_names_on_resume

    # --- Final output threads through all three steps in order ---
    assert result.output == "final:approved:prepared:raw", (
        f"Expected 'final:approved:prepared:raw'; got {result.output!r}"
    )

    # --- ExecutionResumedEvent emitted with matching checkpoint_id ---
    assert_trace_contains(
        resume_emitter,
        ExecutionResumedEvent,
        predicate=lambda e: e.checkpoint_id == checkpoint.checkpoint_id,
    )

    # --- Store invariants: second call to step_2 succeeded ---
    assert step_2_calls["count"] == 2
