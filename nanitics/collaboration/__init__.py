from nanitics.collaboration.approval_gate import ApprovalGate
from nanitics.collaboration.approval_wrapped import ApprovalWrappedTool
from nanitics.collaboration.async_provider import AsyncHumanInputProvider
from nanitics.collaboration.durable_provider import DurableHumanInputProvider
from nanitics.collaboration.hitl_store import (
    DuplicateHitlRequestError,
    HitlRequestStore,
    InMemoryHitlRequestStore,
)

try:
    from nanitics.collaboration.postgres_hitl_store import (
        PostgresHitlRequestStore,
        get_hitl_schema_sql,
    )
except ImportError:
    PostgresHitlRequestStore = None  # type: ignore[assignment,misc]
    get_hitl_schema_sql = None  # type: ignore[assignment]

from nanitics.collaboration.protocol import (
    CallbackHumanInputProvider,
    HumanDecision,
    HumanInputProvider,
    HumanInputRequest,
    HumanInputResponse,
    HumanInputType,
)
from nanitics.collaboration.revision_gate import RevisionGate
from nanitics.collaboration.tools import (
    create_ask_human_tool,
    create_hitl_tools,
    create_request_approval_tool,
)

__all__ = [
    "ApprovalGate",
    # Approval
    "ApprovalWrappedTool",
    # Async HITL
    "AsyncHumanInputProvider",
    "CallbackHumanInputProvider",
    "DuplicateHitlRequestError",
    # Durable HITL
    "DurableHumanInputProvider",
    "HitlRequestStore",
    "HumanDecision",
    # Protocol
    "HumanInputProvider",
    # Models
    "HumanInputRequest",
    "HumanInputResponse",
    # Enums
    "HumanInputType",
    "InMemoryHitlRequestStore",
    # Postgres HITL
    "PostgresHitlRequestStore",
    "RevisionGate",
    "create_ask_human_tool",
    # Tool factories
    "create_hitl_tools",
    "create_request_approval_tool",
    "get_hitl_schema_sql",
]
