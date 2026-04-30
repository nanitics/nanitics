import pytest

from nanitics.composition.durability.models import SuspensionInfo
from nanitics.composition.durability.suspension import SuspendExecution


class TestSuspendExecution:
    def test_is_base_exception(self) -> None:
        exc = SuspendExecution(
            suspension_info=SuspensionInfo(
                suspension_id="test-id",
                request_id="test-request",
                request_type="approval",
                prompt="Approve?",
            ),
        )
        assert isinstance(exc, BaseException)
        assert not isinstance(exc, Exception)

    def test_not_caught_by_except_exception(self) -> None:
        with pytest.raises(SuspendExecution):
            try:
                raise SuspendExecution(
                    suspension_info=SuspensionInfo(
                        suspension_id="test-id",
                        request_id="test-request",
                        request_type="approval",
                        prompt="Approve?",
                    ),
                )
            except Exception:
                pytest.fail("SuspendExecution was caught by except Exception")

    def test_fields(self) -> None:
        info = SuspensionInfo(
            suspension_id="sus-123",
            request_id="req-123",
            request_type="approval",
            prompt="Approve?",
        )
        exc = SuspendExecution(
            suspension_info=info,
            checkpoint_data={"key": "value"},
        )
        assert exc.suspension_info is info
        assert exc.suspension_info.suspension_id == "sus-123"
        assert exc.checkpoint_data == {"key": "value"}

    def test_checkpoint_data_defaults_to_none(self) -> None:
        exc = SuspendExecution(
            suspension_info=SuspensionInfo(
                suspension_id="test-id",
                request_id="test-request",
                request_type="approval",
                prompt="Approve?",
            ),
        )
        assert exc.checkpoint_data is None

    def test_message(self) -> None:
        exc = SuspendExecution(
            suspension_info=SuspensionInfo(
                suspension_id="sus-456",
                request_id="req-456",
                request_type="approval",
                prompt="Approve?",
            ),
        )
        assert "sus-456" in str(exc)
