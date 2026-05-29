from __future__ import annotations

from typing import Any

from nanitics.composition.durability.models import SuspensionInfo


class SuspendExecution(BaseException):
    """Control flow signal indicating durable suspension.

    Inherits BaseException to bypass all `except Exception` blocks,
    ensuring clean propagation through tool execution, agent loops,
    and orchestrators.

    Two optional resume-state carriers travel with the signal as it
    propagates up the orchestration tree:

    - ``checkpoint_data`` — the *leaf* agent's per-agent resume state. Set
      by the suspending agent; consumed by the orchestrator directly
      containing that agent (stored under ``state["agent_checkpoint"]``),
      which then clears it.
    - ``orchestration_state`` — the checkpoint-state dict of the workflow
      frame currently re-raising. Each orchestrator on the suspend path
      sets this to its own state before re-raising; its *parent* reads it
      and embeds it under ``state["nested_checkpoint"]``. This builds a
      single recursive checkpoint that captures the full suspension path
      through nested workflows.
    """

    def __init__(
        self,
        *,
        suspension_info: SuspensionInfo,
        checkpoint_data: dict[str, Any] | None = None,
        orchestration_state: dict[str, Any] | None = None,
    ) -> None:
        self.suspension_info = suspension_info
        self.checkpoint_data = checkpoint_data
        self.orchestration_state = orchestration_state
        super().__init__(f"Execution suspended: {suspension_info.suspension_id}")
