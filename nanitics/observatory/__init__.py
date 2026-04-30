"""Observatory — backend API for the Nanitics trace visualization frontend.

Usage::

    from nanitics.observatory import create_observatory_router

    router = create_observatory_router(store)
    app.include_router(router, prefix="/api/observatory")
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
from nanitics.observatory.router import create_observatory_router
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
    "create_observatory_router",
]
