from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from nanitics.composition.durability.models import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointVersionError,
    RunCheckpoint,
    StepRecord,
    SuspensionInfo,
)


class TestSuspensionInfo:
    def test_creates_with_required_fields(self) -> None:
        info = SuspensionInfo(
            suspension_id="sus-1",
            request_id="req-1",
            request_type="approval",
            prompt="Approve this?",
        )
        assert info.suspension_id == "sus-1"
        assert info.suspension_type == "hitl"
        assert info.request_id == "req-1"
        assert info.request_type == "approval"
        assert info.prompt == "Approve this?"
        assert info.agent_name is None

    def test_frozen(self) -> None:
        info = SuspensionInfo(
            suspension_id="sus-1",
            request_id="req-1",
            request_type="approval",
            prompt="Approve this?",
        )
        with pytest.raises(ValidationError):
            info.suspension_id = "changed"


class TestRunCheckpoint:
    def test_creates_with_defaults(self) -> None:
        info = SuspensionInfo(
            suspension_id="sus-1",
            request_id="req-1",
            request_type="approval",
            prompt="Approve?",
        )
        cp = RunCheckpoint(
            run_id="run-1",
            checkpoint_type="orchestration",
            state={"step": 0},
            suspension_info=info,
        )
        assert cp.checkpoint_id  # auto-generated
        assert cp.run_id == "run-1"
        assert cp.checkpoint_type == "orchestration"
        assert cp.schema_version == CHECKPOINT_SCHEMA_VERSION
        assert cp.state == {"step": 0}
        assert cp.suspension_info is info
        assert cp.checkpoint_reason == "hitl_suspend"
        assert isinstance(cp.created_at, datetime)

    def test_hitl_construction_defaults_reason_to_hitl_suspend(self) -> None:
        """Existing HITL construction is unaffected by the additive changes.

        ``suspension_info`` is still set and ``checkpoint_reason`` defaults to
        ``"hitl_suspend"`` — the pre-step-durability shape.
        """
        info = SuspensionInfo(
            suspension_id="sus-1",
            request_id="req-1",
            request_type="approval",
            prompt="Approve?",
        )
        cp = RunCheckpoint(
            run_id="run-1",
            checkpoint_type="orchestration",
            state={"step": 0},
            suspension_info=info,
        )
        assert cp.suspension_info is info
        assert cp.checkpoint_reason == "hitl_suspend"

    def test_step_checkpoint_has_no_suspension_info(self) -> None:
        """A step/crash cursor checkpoint is not a suspension.

        ``suspension_info`` is optional and ``None`` for a ``"step"``
        checkpoint; the discriminator distinguishes it from a HITL suspension
        without inferring from a null ``suspension_info``.
        """
        cp = RunCheckpoint(
            run_id="run-1",
            checkpoint_type="agent",
            state={"cursor": 2},
            suspension_info=None,
            checkpoint_reason="step",
        )
        assert cp.suspension_info is None
        assert cp.checkpoint_reason == "step"

    def test_crash_safe_reason_accepted(self) -> None:
        cp = RunCheckpoint(
            run_id="run-1",
            checkpoint_type="agent",
            state={},
            checkpoint_reason="crash_safe",
        )
        assert cp.checkpoint_reason == "crash_safe"
        assert cp.suspension_info is None

    def test_rejects_unknown_checkpoint_reason(self) -> None:
        with pytest.raises(ValidationError):
            RunCheckpoint(
                run_id="run-1",
                checkpoint_type="agent",
                state={},
                checkpoint_reason="bogus",  # type: ignore[arg-type]
            )

    def test_step_checkpoint_json_roundtrip(self) -> None:
        cp = RunCheckpoint(
            run_id="run-1",
            checkpoint_type="agent",
            state={"cursor": 2},
            checkpoint_reason="step",
        )
        restored = RunCheckpoint.model_validate_json(cp.model_dump_json())
        assert restored == cp
        assert restored.suspension_info is None
        assert restored.checkpoint_reason == "step"

    def test_serialization_roundtrip(self) -> None:
        info = SuspensionInfo(
            suspension_id="sus-1",
            request_id="req-1",
            request_type="approval",
            prompt="Approve?",
            agent_name="test-agent",
        )
        cp = RunCheckpoint(
            checkpoint_id="cp-1",
            run_id="run-1",
            checkpoint_type="agent",
            schema_version=1,
            state={"messages": [{"role": "user", "content": "hi"}], "step": 3},
            suspension_info=info,
        )
        data = cp.model_dump()
        restored = RunCheckpoint.model_validate(data)
        assert restored == cp

    def test_json_roundtrip(self) -> None:
        info = SuspensionInfo(
            suspension_id="sus-1",
            request_id="req-1",
            request_type="approval",
            prompt="Approve?",
        )
        cp = RunCheckpoint(
            run_id="run-1",
            checkpoint_type="orchestration",
            state={"nested": {"data": [1, 2, 3]}},
            suspension_info=info,
        )
        json_str = cp.model_dump_json()
        restored = RunCheckpoint.model_validate_json(json_str)
        assert restored == cp

    def test_frozen(self) -> None:
        info = SuspensionInfo(
            suspension_id="sus-1",
            request_id="req-1",
            request_type="approval",
            prompt="Approve?",
        )
        cp = RunCheckpoint(
            run_id="run-1",
            checkpoint_type="orchestration",
            state={},
            suspension_info=info,
        )
        with pytest.raises(ValidationError):
            cp.run_id = "changed"


class TestStepRecord:
    def test_creates_with_defaults(self) -> None:
        rec = StepRecord(
            run_id="run-1",
            step_path="seq#2/agent/turn#3/tool#1:send_email",
            step_kind="tool_call",
            result={"content": "ok", "tool_call_id": "tc-1"},
        )
        assert rec.run_id == "run-1"
        assert rec.step_path == "seq#2/agent/turn#3/tool#1:send_email"
        assert rec.step_kind == "tool_call"
        assert rec.result == {"content": "ok", "tool_call_id": "tc-1"}
        assert rec.schema_version == CHECKPOINT_SCHEMA_VERSION
        assert isinstance(rec.created_at, datetime)

    def test_frozen(self) -> None:
        rec = StepRecord(
            run_id="run-1",
            step_path="seq#0",
            step_kind="orchestration_step",
            result={},
        )
        with pytest.raises(ValidationError):
            rec.run_id = "changed"

    def test_json_roundtrip(self) -> None:
        rec = StepRecord(
            run_id="run-1",
            step_path="seq#0/agent/turn#1",
            step_kind="agent_turn",
            result={"messages": [{"role": "assistant", "content": "hi"}]},
            created_at=datetime(2025, 1, 1, tzinfo=UTC),
        )
        restored = StepRecord.model_validate_json(rec.model_dump_json())
        assert restored == rec

    def test_rejects_unknown_step_kind(self) -> None:
        with pytest.raises(ValidationError):
            StepRecord(
                run_id="run-1",
                step_path="seq#0",
                step_kind="not_a_kind",  # type: ignore[arg-type]
                result={},
            )


class TestCheckpointVersionError:
    def test_inherits_nanitics_error(self) -> None:
        from nanitics.infrastructure.errors import NaniticsError

        err = CheckpointVersionError(
            "Version mismatch",
            expected_version=1,
            actual_version=2,
        )
        assert isinstance(err, NaniticsError)
        assert err.expected_version == 1
        assert err.actual_version == 2
        assert "Version mismatch" in str(err)

    def test_to_dict(self) -> None:
        err = CheckpointVersionError(
            "Version mismatch",
            expected_version=1,
            actual_version=2,
        )
        d = err.to_dict()
        assert d["expected_version"] == 1
        assert d["actual_version"] == 2
        assert d["message"] == "Version mismatch"


class TestSchemaVersion:
    def test_constant_is_int(self) -> None:
        assert isinstance(CHECKPOINT_SCHEMA_VERSION, int)
        assert CHECKPOINT_SCHEMA_VERSION == 4
