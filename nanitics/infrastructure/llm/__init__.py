from nanitics.infrastructure.llm.anthropic import AnthropicLLMClient
from nanitics.infrastructure.llm.instrumented import InstrumentedLLMClient
from nanitics.infrastructure.llm.mock import MockLLMClient
from nanitics.infrastructure.llm.openai import OpenAILLMClient
from nanitics.infrastructure.llm.protocol import (
    ContentBlock,
    ImageContentBlock,
    LLMClient,
    LLMResponse,
    Message,
    SystemPromptSection,
    TextContentBlock,
    ToolCall,
    ToolSchema,
)
from nanitics.infrastructure.llm.routing import (
    CostBudgetRouting,
    RoutingContext,
    RoutingLLMClient,
    RoutingStrategy,
    RuleBasedRouting,
)

try:
    from nanitics.infrastructure.llm.litellm import LiteLLMClient
except ImportError:
    LiteLLMClient = None  # type: ignore[assignment,misc]

try:
    from nanitics.infrastructure.llm.mistral import MistralLLMClient
except ImportError:
    MistralLLMClient = None  # type: ignore[assignment,misc]

__all__ = [
    "AnthropicLLMClient",
    "ContentBlock",
    "CostBudgetRouting",
    "ImageContentBlock",
    "InstrumentedLLMClient",
    "LLMClient",
    "LLMResponse",
    "LiteLLMClient",
    "Message",
    "MistralLLMClient",
    "MockLLMClient",
    "OpenAILLMClient",
    "RoutingContext",
    "RoutingLLMClient",
    "RoutingStrategy",
    "RuleBasedRouting",
    "SystemPromptSection",
    "TextContentBlock",
    "ToolCall",
    "ToolSchema",
]
