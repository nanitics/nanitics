"""Error classes and the error-handling capability surface."""

from nanitics.capabilities.errors import (
    ErrorCategory,
    ErrorClassifier,
    ErrorHandler,
    RetryPolicy,
    classify_error,
    format_correction_prompt,
)
from nanitics.collaboration.hitl_store import DuplicateHitlRequestError
from nanitics.composition.durability.models import CheckpointVersionError
from nanitics.composition.multi_agent.peer_network import PeerBudgetExceededError
from nanitics.composition.orchestration.pipeline import PipelineContractError
from nanitics.composition.orchestration.workflow import WorkflowCancelledError
from nanitics.infrastructure.errors import (
    AgentBudgetExceededError,
    AgentError,
    AgentEscalationError,
    AgentIterationLimitError,
    AgentToolCallLimitError,
    ApprovalTimeoutError,
    ApprovalUnavailableError,
    EmbeddingError,
    EmbeddingProviderError,
    EmbeddingRateLimitError,
    HumanInputProviderError,
    LLMContextLengthError,
    LLMError,
    LLMProviderError,
    LLMRateLimitError,
    LLMSchemaViolationError,
    NaniticsError,
    ToolError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolParameterError,
    ToolTimeoutError,
)
from nanitics.infrastructure.observability import MalformedStoredEventError

__all__ = [
    "AgentBudgetExceededError",
    "AgentError",
    "AgentEscalationError",
    "AgentIterationLimitError",
    "AgentToolCallLimitError",
    "ApprovalTimeoutError",
    "ApprovalUnavailableError",
    "CheckpointVersionError",
    "DuplicateHitlRequestError",
    "EmbeddingError",
    "EmbeddingProviderError",
    "EmbeddingRateLimitError",
    "ErrorCategory",
    "ErrorClassifier",
    "ErrorHandler",
    "HumanInputProviderError",
    "LLMContextLengthError",
    "LLMError",
    "LLMProviderError",
    "LLMRateLimitError",
    "LLMSchemaViolationError",
    "MalformedStoredEventError",
    "NaniticsError",
    "PeerBudgetExceededError",
    "PipelineContractError",
    "RetryPolicy",
    "ToolError",
    "ToolExecutionError",
    "ToolNotFoundError",
    "ToolParameterError",
    "ToolTimeoutError",
    "WorkflowCancelledError",
    "classify_error",
    "format_correction_prompt",
]
