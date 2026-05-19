"""Checkpoint and suspension: durable workflow execution with persist-and-resume.

Demonstrates the checkpoint system that enables workflows to suspend execution
(e.g., for human approval), persist their state, and resume later — potentially
in a different process. Six sections progress from low-level primitives to a full
HITL-integrated suspend/resume cycle.

Related guide: docs/guides/orchestration.md
"""

import asyncio

from examples.helpers import make_emitter
from nanitics.composition import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointStore,
    FunctionStep,
    InMemoryCheckpointStore,
    RunCheckpoint,
    Sequential,
    SuspendExecution,
    SuspensionInfo,
)
from nanitics.errors import CheckpointVersionError
from nanitics.hitl import (
    ApprovalGate,
    DurableHumanInputProvider,
    HumanDecision,
    HumanInputResponse,
    InMemoryHitlRequestStore,
)
from nanitics.infrastructure import (
    CheckpointSavedEvent,
    ExecutionResumedEvent,
    ExecutionSuspendedEvent,
    RunSuspendedEvent,
)


async def main() -> None:
    # --- Section 1: Checkpoint Store Basics ---
    print("--- Section 1: Checkpoint Store Basics ---")

    # InMemoryCheckpointStore implements the CheckpointStore protocol.
    # It persists RunCheckpoint objects keyed by checkpoint_id and provides
    # load-by-run_id, delete-by-id, and delete-by-run operations.

    store = InMemoryCheckpointStore()
    assert isinstance(store, CheckpointStore), "Must implement CheckpointStore protocol"

    # Create a checkpoint manually — in practice, workflows do this automatically
    # when a step raises SuspendExecution.
    checkpoint = RunCheckpoint(
        checkpoint_id="cp-1",
        run_id="run-1",
        checkpoint_type="orchestration",
        state={"orchestrator_type": "sequential", "suspended_step_index": 1},
        suspension_info=SuspensionInfo(
            suspension_id="sus-1",
            request_id="req-1",
            request_type="approval",
            prompt="Approve this output?",
            agent_name="reviewer",
        ),
    )

    await store.save(checkpoint)
    loaded = await store.load("run-1")
    assert loaded is not None
    assert loaded.checkpoint_id == "cp-1"
    assert loaded.run_id == "run-1"
    assert loaded.checkpoint_type == "orchestration"
    assert loaded.schema_version == CHECKPOINT_SCHEMA_VERSION
    assert loaded.state["suspended_step_index"] == 1
    assert loaded.suspension_info.request_type == "approval"
    assert loaded.suspension_info.prompt == "Approve this output?"
    assert loaded.created_at is not None

    print(f"  Checkpoint ID: {loaded.checkpoint_id}")
    print(f"  Schema version: {loaded.schema_version}")
    print(f"  Suspension prompt: '{loaded.suspension_info.prompt}'")

    # Load returns None for unknown run IDs
    assert await store.load("unknown-run") is None

    # Delete all checkpoints for a run
    await store.delete_for_run("run-1")
    assert await store.load("run-1") is None

    print("✓ CheckpointStore: save, load, inspect, delete")

    # --- Section 2: SuspendExecution Mechanics ---
    print("\n--- Section 2: SuspendExecution Mechanics ---")

    # SuspendExecution inherits from BaseException, NOT Exception.
    # This ensures it bypasses all `except Exception` blocks and propagates
    # cleanly through tool execution, agent loops, and orchestrators.

    assert issubclass(SuspendExecution, BaseException)
    assert not issubclass(SuspendExecution, Exception)

    # Construction: requires SuspensionInfo, optionally carries checkpoint_data
    info = SuspensionInfo(
        suspension_id="sus-2",
        request_id="req-2",
        request_type="approval",
        prompt="Ready to publish?",
    )
    try:
        raise SuspendExecution(suspension_info=info, checkpoint_data={"draft": "v1"})
    except SuspendExecution as exc:
        assert exc.suspension_info.suspension_id == "sus-2"
        assert exc.checkpoint_data == {"draft": "v1"}
        print(f"  Caught SuspendExecution: {exc.suspension_info.suspension_id}")

    # Demonstrate that `except Exception` does NOT catch SuspendExecution
    async def function_with_broad_except(value: str) -> str:
        try:
            raise SuspendExecution(
                suspension_info=SuspensionInfo(
                    suspension_id="sus-pass",
                    request_id="req-pass",
                    request_type="approval",
                    prompt="This should pass through",
                ),
            )
        except Exception:
            return "caught"  # This should never execute
        return "unreachable"

    try:
        await function_with_broad_except("test")
        assert False, "Should not reach here"
    except SuspendExecution:
        pass  # Expected: it propagated through `except Exception`

    print("✓ SuspendExecution: BaseException inheritance, except Exception passthrough")

    # --- Section 3: Workflow Suspension and Checkpoint Saving ---
    print("\n--- Section 3: Workflow Suspension and Checkpoint Saving ---")

    # When a step raises SuspendExecution inside a Sequential workflow with a
    # checkpoint_store, the workflow: (1) saves a checkpoint, (2) emits events,
    # (3) re-raises the exception. The checkpoint captures completed results
    # and the suspension position.

    store = InMemoryCheckpointStore()
    emitter = make_emitter("suspend-trace")

    async def step_1(input: str) -> str:
        return f"prepared: {input}"

    async def step_2_suspend(input: str) -> str:
        raise SuspendExecution(
            suspension_info=SuspensionInfo(
                suspension_id="sus-1",
                request_id="req-1",
                request_type="approval",
                prompt="Approve step 2 output?",
                agent_name="reviewer",
            ),
        )

    async def step_3(input: str) -> str:
        return f"finalized: {input}"

    workflow = Sequential(
        name="suspend-pipeline",
        steps=[
            FunctionStep("step_1", step_1),
            FunctionStep("step_2", step_2_suspend),
            FunctionStep("step_3", step_3),
        ],
        emitter=emitter,
        checkpoint_store=store,
        run_id="run-suspend",
    )

    # step_1 completes, step_2 suspends, step_3 is never reached
    try:
        await workflow.execute("raw data")
        assert False, "Should have raised SuspendExecution"
    except SuspendExecution:
        pass

    # Inspect the saved checkpoint
    checkpoint = await store.load("run-suspend")
    assert checkpoint is not None
    assert checkpoint.state["orchestrator_type"] == "sequential"
    assert checkpoint.state["suspended_step_index"] == 1  # step_2 is at index 1
    assert checkpoint.state["completed_results"]["step_1"]["output"] == "prepared: raw data"
    assert checkpoint.state["last_output"] == "prepared: raw data"
    assert checkpoint.state["original_input"] == "raw data"
    assert checkpoint.suspension_info.prompt == "Approve step 2 output?"

    print(f"  Checkpoint saved for run: {checkpoint.run_id}")
    print(f"  Suspended at step index: {checkpoint.state['suspended_step_index']}")
    print(f"  Completed results: {list(checkpoint.state['completed_results'].keys())}")

    # Verify events
    saved_events = [e for e in emitter.events if isinstance(e, CheckpointSavedEvent)]
    suspended_events = [e for e in emitter.events if isinstance(e, ExecutionSuspendedEvent)]
    run_suspended_events = [e for e in emitter.events if isinstance(e, RunSuspendedEvent)]

    assert len(saved_events) == 1, f"Expected 1 CheckpointSavedEvent, got {len(saved_events)}"
    assert len(suspended_events) == 1, f"Expected 1 ExecutionSuspendedEvent, got {len(suspended_events)}"
    assert len(run_suspended_events) == 1, f"Expected 1 RunSuspendedEvent, got {len(run_suspended_events)}"
    assert saved_events[0].checkpoint_id == checkpoint.checkpoint_id
    assert suspended_events[0].suspension_id == "sus-1"

    print("✓ Workflow suspension: checkpoint saved, events emitted")

    # --- Section 4: Resuming from a Checkpoint ---
    print("\n--- Section 4: Resuming from a Checkpoint ---")

    # Resume a suspended workflow by loading the checkpoint and calling
    # execute(input, resume_from=checkpoint). The workflow skips completed
    # steps and re-executes from the suspension point.

    store = InMemoryCheckpointStore()
    emitter = make_emitter("resume-trace")

    # Mutable flag pattern: step_2 suspends on first call, succeeds on second.
    # A closure captures a dict so both calls share state.
    call_count: dict[str, int] = {"step_2": 0}

    async def step_2_resumable(input: str) -> str:
        call_count["step_2"] += 1
        if call_count["step_2"] == 1:
            raise SuspendExecution(
                suspension_info=SuspensionInfo(
                    suspension_id="sus-resume",
                    request_id="req-resume",
                    request_type="approval",
                    prompt="Approve?",
                ),
            )
        return f"processed: {input}"

    # Track which steps execute
    executed_steps: list[str] = []

    async def tracked_step_1(input: str) -> str:
        executed_steps.append("step_1")
        return f"prepared: {input}"

    async def tracked_step_3(input: str) -> str:
        executed_steps.append("step_3")
        return f"finalized: {input}"

    workflow = Sequential(
        name="resume-pipeline",
        steps=[
            FunctionStep("step_1", tracked_step_1),
            FunctionStep("step_2", step_2_resumable),
            FunctionStep("step_3", tracked_step_3),
        ],
        emitter=emitter,
        checkpoint_store=store,
        run_id="run-resume",
    )

    # First execution: suspends at step_2
    try:
        await workflow.execute("raw data")
        assert False, "Should have raised SuspendExecution"
    except SuspendExecution:
        pass

    assert "step_1" in executed_steps
    executed_steps.clear()

    # Load checkpoint and resume
    checkpoint = await store.load("run-resume")
    assert checkpoint is not None

    # Reconstruct the workflow — simulates resuming in a different process.
    # Only the checkpoint plus the same step definitions are needed.
    emitter_resume = make_emitter("resume-trace-2")
    workflow_resume = Sequential(
        name="resume-pipeline",
        steps=[
            FunctionStep("step_1", tracked_step_1),
            FunctionStep("step_2", step_2_resumable),
            FunctionStep("step_3", tracked_step_3),
        ],
        emitter=emitter_resume,
        checkpoint_store=store,
        run_id="run-resume",
    )

    result = await workflow_resume.execute("raw data", resume_from=checkpoint)

    # step_1 was skipped on resume, only step_2 and step_3 ran
    assert "step_1" not in executed_steps, "step_1 should be skipped on resume"
    assert "step_3" in executed_steps, "step_3 should execute on resume"
    assert result.output == "finalized: processed: prepared: raw data"

    print(f"  Final output: '{result.output}'")
    print(f"  Steps executed on resume: {executed_steps}")

    # ExecutionResumedEvent emitted
    resumed_events = [e for e in emitter_resume.events if isinstance(e, ExecutionResumedEvent)]
    assert len(resumed_events) == 1, f"Expected 1 ExecutionResumedEvent, got {len(resumed_events)}"
    assert resumed_events[0].checkpoint_id == checkpoint.checkpoint_id

    print("✓ Resume: skipped completed steps, re-executed from suspension point")

    # --- Section 5: Checkpoint Version Validation ---
    print("\n--- Section 5: Checkpoint Version Validation ---")

    # When the SDK's CHECKPOINT_SCHEMA_VERSION changes (e.g., after an upgrade),
    # resuming with an old checkpoint raises CheckpointVersionError. This prevents
    # deserialization errors from incompatible checkpoint formats.

    outdated_checkpoint = RunCheckpoint(
        checkpoint_id="cp-old",
        run_id="run-version",
        checkpoint_type="orchestration",
        schema_version=999,  # Doesn't match current CHECKPOINT_SCHEMA_VERSION
        state={
            "orchestrator_type": "sequential",
            "suspended_step_index": 0,
            "completed_results": {},
            "last_output": "test",
            "original_input": "test",
        },
        suspension_info=SuspensionInfo(
            suspension_id="sus-v",
            request_id="req-v",
            request_type="approval",
            prompt="N/A",
        ),
    )

    version_emitter = make_emitter("version-trace")
    version_workflow = Sequential(
        name="version-check",
        steps=[FunctionStep("s1", step_1)],
        emitter=version_emitter,
        checkpoint_store=InMemoryCheckpointStore(),
        run_id="run-version",
    )

    try:
        await version_workflow.execute("input", resume_from=outdated_checkpoint)
        assert False, "Should have raised CheckpointVersionError"
    except CheckpointVersionError as e:
        assert e.expected_version == CHECKPOINT_SCHEMA_VERSION
        assert e.actual_version == 999
        print(f"  Expected version: {e.expected_version}")
        print(f"  Actual version: {e.actual_version}")

    print("✓ Version validation: incompatible checkpoints rejected")

    # --- Section 6: HITL-Integrated Suspend/Resume Cycle ---
    print("\n--- Section 6: HITL-Integrated Suspend/Resume Cycle ---")

    # The full production-realistic flow: a workflow suspends at an ApprovalGate
    # backed by DurableHumanInputProvider, persists the checkpoint and HITL
    # request, then resumes after the human response is written to the store.

    hitl_store = InMemoryHitlRequestStore(run_id="run-hitl")
    provider = DurableHumanInputProvider(request_store=hitl_store)
    checkpoint_store = InMemoryCheckpointStore()
    emitter_hitl = make_emitter("hitl-trace")

    async def draft_step(input: str) -> str:
        return f"Draft report on {input}"

    async def publish_step(input: str) -> str:
        return f"Published: {input}"

    approval_gate = ApprovalGate(
        provider=provider,
        emitter=emitter_hitl,
        prompt="Approve this draft for publication?",
        name="approval_gate",
        run_id="run-hitl",
    )

    hitl_workflow = Sequential(
        name="publish-pipeline",
        steps=[
            FunctionStep("draft", draft_step),
            approval_gate,
            FunctionStep("publish", publish_step),
        ],
        emitter=emitter_hitl,
        checkpoint_store=checkpoint_store,
        run_id="run-hitl",
    )

    # Step 1: Execute — draft completes, approval gate suspends
    try:
        await hitl_workflow.execute("Q4 results")
        assert False, "Should have raised SuspendExecution"
    except SuspendExecution:
        pass

    print("  Workflow suspended at approval gate")

    # Inspect the saved checkpoint
    hitl_checkpoint = await checkpoint_store.load("run-hitl")
    assert hitl_checkpoint is not None
    assert hitl_checkpoint.state["suspended_step_index"] == 1  # approval_gate index

    # Inspect the stored HITL request
    pending = await hitl_store.get_pending_requests("run-hitl")
    assert len(pending) == 1, f"Expected 1 pending request, got {len(pending)}"
    hitl_request = pending[0]
    assert hitl_request.prompt == "Approve this draft for publication?"
    print(f"  HITL request prompt: '{hitl_request.prompt}'")
    print(f"  HITL request ID: {hitl_request.request_id}")

    # Step 2: Simulate human approval — write the response to the store.
    # The durable provider looks it up by request_id on re-execution and
    # returns it directly (no preload channel, no provider-local state).
    await hitl_store.save_response(
        hitl_request.request_id,
        HumanInputResponse(
            request_id=hitl_request.request_id,
            decision=HumanDecision.APPROVE,
        ),
    )

    # Step 3: Resume the workflow
    # Reconstruct workflow for resume — in production this would be a new process
    emitter_hitl_resume = make_emitter("hitl-resume-trace")
    hitl_workflow_resume = Sequential(
        name="publish-pipeline",
        steps=[
            FunctionStep("draft", draft_step),
            ApprovalGate(
                provider=provider,
                emitter=emitter_hitl_resume,
                prompt="Approve this draft for publication?",
                name="approval_gate",
                run_id="run-hitl",
            ),
            FunctionStep("publish", publish_step),
        ],
        emitter=emitter_hitl_resume,
        checkpoint_store=checkpoint_store,
        run_id="run-hitl",
    )

    result = await hitl_workflow_resume.execute("Q4 results", resume_from=hitl_checkpoint)

    assert result.output == "Published: Draft report on Q4 results", f"Got: {result.output}"
    print(f"  Final output: '{result.output}'")

    # Verify resume event
    resumed_events = [e for e in emitter_hitl_resume.events if isinstance(e, ExecutionResumedEvent)]
    assert len(resumed_events) == 1
    print(f"  Resumed from checkpoint: {resumed_events[0].checkpoint_id}")

    print("✓ HITL-integrated suspend/resume: full lifecycle complete")

    print("\nAll sections passed ✓")


if __name__ == "__main__":
    asyncio.run(main())
