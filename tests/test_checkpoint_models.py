from datetime import datetime

import pytest
from pydantic import ValidationError

from nanitics.composition.durability.models import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointVersionError,
    RunCheckpoint,
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
        assert isinstance(cp.created_at, datetime)

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
        assert CHECKPOINT_SCHEMA_VERSION == 1
