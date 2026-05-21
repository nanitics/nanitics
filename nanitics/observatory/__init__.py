"""Observatory — backend API and embedded SPA for trace visualization.

Most adopters call :func:`mount_observatory` to mount the API and UI
under a single prefix in one line::

    from nanitics.observatory import mount_observatory

    mount_observatory(app, store, prefix="/observatory")

The wheel ships the prebuilt SPA, so visiting ``/observatory/`` works
without any frontend toolchain. Consumers that need different middleware
on the data endpoints and the UI — or that serve only one of the two —
use :func:`create_observatory_api_router` and
:func:`create_observatory_ui_router` directly.
"""

from nanitics.observatory.models import (
    AgentDetailResponse,
    AgentInfoResponse,
    AgentListResponse,
    AgentStatsResponse,
    EventListResponse,
    RunCreateRequest,
    RunDetailResponse,
    RunListItem,
    RunListResponse,
    RunResponse,
    RunStatusUpdateRequest,
    SpanEventsResponse,
    SpanSummary,
    SpanTreeNodeResponse,
    SpanTreeResponse,
    TraceEventResponse,
    TraceSummaryResponse,
    WorkflowDAGResponse,
    WorkflowStepResponse,
    WorkflowStepStatus,
)
from nanitics.observatory.router import (
    create_observatory_api_router,
    create_observatory_ui_router,
    mount_observatory,
)
from nanitics.observatory.service import ObservatoryService

__all__ = [
    "AgentDetailResponse",
    "AgentInfoResponse",
    "AgentListResponse",
    "AgentStatsResponse",
    "EventListResponse",
    "ObservatoryService",
    "RunCreateRequest",
    "RunDetailResponse",
    "RunListItem",
    "RunListResponse",
    "RunResponse",
    "RunStatusUpdateRequest",
    "SpanEventsResponse",
    "SpanSummary",
    "SpanTreeNodeResponse",
    "SpanTreeResponse",
    "TraceEventResponse",
    "TraceSummaryResponse",
    "WorkflowDAGResponse",
    "WorkflowStepResponse",
    "WorkflowStepStatus",
    "create_observatory_api_router",
    "create_observatory_ui_router",
    "mount_observatory",
]
