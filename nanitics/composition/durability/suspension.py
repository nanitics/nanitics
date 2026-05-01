from __future__ import annotations

from typing import Any

from nanitics.composition.durability.models import SuspensionInfo


class SuspendExecution(BaseException):
    """Control flow signal indicating durable suspension.

    Inherits BaseException to bypass all `except Exception` blocks,
    ensuring clean propagation through tool execution, agent loops,
    and orchestrators.
    """

    def __init__(
        self,
        *,
        suspension_info: SuspensionInfo,
        checkpoint_data: dict[str, Any] | None = None,
    ) -> None:
        self.suspension_info = suspension_info
        self.checkpoint_data = checkpoint_data
        super().__init__(f"Execution suspended: {suspension_info.suspension_id}")
