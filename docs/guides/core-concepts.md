# Core Concepts

> For API details — signatures, fields, constraints — read the docstrings in your editor, in the source tree under [`nanitics/`](../../nanitics/), or browse them at [docs.nanitics.dev](https://docs.nanitics.dev/). `nanitics.__all__` is the authoritative public surface.

Everything in the SDK builds on a few foundational abstractions: the agent loop, messages, LLM clients, system prompts, and extension points. This guide covers how they fit together. For API details (fields, constructors, parameters), see the docstrings on each class.

## The Agent Loop

Every agent follows the same high-level cycle:

1. **Observe** — receive a task or new information (user input, tool results, context injections)
2. **Think** — send the conversation to an LLM to decide what to do next
3. **Act** — execute tool calls if the LLM requested them
4. **Repeat** — continue until the LLM produces a final text response (no tool calls)

The loop terminates when:
- The LLM responds with text only (no tool calls) → `"complete"`
- The iteration limit is reached → `"iteration_limit"`
- The cancellation token is triggered → `"cancelled"`
- Output evaluation rejects the result after max revisions → `"evaluation_failed"`

Different agent types implement variations of this loop (tree search, plan-then-execute, reflexion, etc.), but the core mechanisms — LLM calls, tool dispatch, context management — are shared across all of them.

> **See also:** [`examples/agents/react_agent.py`](../../examples/agents/react_agent.py) for a complete ReAct loop with multi-turn tool use and event tracing. See [Agent Types](agent-types.md) for how each agent type varies the loop.

## Messages

All communication flows through `Message` objects. There are three roles:

- **`user`** — input from the user or injected context (context providers insert user messages)
- **`assistant`** — LLM responses, which may contain text, tool calls, or both
- **`tool_result`** — output from a tool execution, linked back to its `ToolCall` via `tool_call_id`

Messages are immutable. The full conversation history — including injected context and tool results — is available on `AgentResult.messages` after a run. Messages can carry `metadata` for internal bookkeeping (e.g., `{"protected": True}` prevents context managers from truncating them).

> **See also:** `Message` docstring in `nanitics/core/messages.py` for all fields.

## AgentResult

Every `agent.run()` call returns an `AgentResult` containing the agent's final output, termination reason, full message history, and aggregated token usage. When `output_schema` is provided, the parsed structured output is available on `result.parsed`. `AgentStep` preserves this — when used in workflows the parsed model becomes the step output, enabling typed data flow through Sequential and Pipeline.

> **See also:** `AgentResult` docstring in `nanitics/core/agent.py`. See [Getting Started](getting-started.md) for end-to-end usage.

## LLM Clients

The SDK communicates with language models through the `LLMClient` protocol — any object with a `generate()` method works, no inheritance required. This lets you swap backends without changing agent code.

### AnthropicLLMClient

The production client for calling Claude via the Anthropic API. Handles message formatting, tool call serialization, structured output via tool-use, and maps API errors to the SDK's error hierarchy (`LLMRateLimitError`, `LLMContextLengthError`, `LLMProviderError`).

Note: `output_schema` and `tools` are mutually exclusive on a single `generate()` call.

> **See also:** `AnthropicLLMClient` docstring in `nanitics/infrastructure/llm/anthropic_client.py`.

### OpenAILLMClient

The production client for calling OpenAI models via the Chat Completions API. Handles message formatting, tool serialization, structured output via the tool-use pattern, multimodal input (text and images), streaming via `on_token`, and maps API errors to the same SDK error hierarchy as `AnthropicLLMClient`. Supports custom `base_url` for Azure, proxies, and OpenAI-compatible endpoints.

Structured output is implemented as a forced tool call rather than OpenAI's native `response_format={"type": "json_schema"}` so that agents written against any provider behave identically when switched between providers.

Note: `output_schema` and `tools` are mutually exclusive on a single `generate()` call (same as the other clients).

> **See also:** `OpenAILLMClient` docstring in `nanitics/infrastructure/llm/openai.py`. [`examples/providers/openai_client.py`](../../examples/providers/openai_client.py).

### LiteLLMClient

An adapter client that wraps [LiteLLM](https://github.com/BerriAI/litellm)'s `acompletion()`. A single `LiteLLMClient` routes to 100+ providers — Bedrock, Vertex, Gemini, Cohere, Together, Groq, Ollama, vLLM, Azure, and many more — via LiteLLM's translation layer. The `model` string is provider-prefixed per LiteLLM's convention: `"openai/gpt-4o-mini"`, `"anthropic/claude-haiku-4-5"`, `"bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0"`, `"gemini/gemini-2.0-flash"`, `"ollama/llama3"`, etc. Provider-specific parameters (Bedrock region, Vertex project, Azure deployment id) pass through via the `extra_kwargs` escape hatch.

> **Trade-offs:** Prefer the native `AnthropicLLMClient`, `OpenAILLMClient`, or `MistralLLMClient` when they exist for your provider. Native clients offer stronger error classification (LiteLLM normalizes errors but `Retry-After` headers are not always surfaced), native cache-token reporting (LiteLLM does not normalize cache tokens consistently across providers), and a lighter dependency footprint. Use `LiteLLMClient` for everything else — it is the long-tail catch-all, not a replacement for native clients.

Note: `output_schema` and `tools` are mutually exclusive on a single `generate()` call (same as the other clients). Install with `pip install nanitics[litellm]`.

> **See also:** `LiteLLMClient` docstring in `nanitics/infrastructure/llm/litellm.py`. [`examples/providers/litellm_adapter.py`](../../examples/providers/litellm_adapter.py).

### MistralLLMClient

The production client for calling Mistral models. Same error mapping and schema semantics as the other native clients. Install with `pip install nanitics[mistral]`.

> **See also:** `MistralLLMClient` docstring in `nanitics/infrastructure/llm/mistral.py`.

### MockLLMClient

Returns scripted responses in sequence — essential for testing agents without API calls. Records every call in `MockLLMClient.calls` for assertions on what system prompts and messages were sent. See [Testing](testing.md) for the full testing pattern.

> **See also:** `MockLLMClient` docstring in `nanitics/infrastructure/llm/mock_client.py`.

### RoutingLLMClient

Wraps multiple LLM clients and routes each request to one based on a strategy. Implements `LLMClient`, so it's a drop-in replacement anywhere an LLM client is expected. Use cases include routing to cheaper models for simple tasks, switching to stronger models when tools are involved, or enforcing token budgets across a run.

Built-in strategies:
- **`RuleBasedRouting`** — route via a custom function that inspects the request context
- **`CostBudgetRouting`** — track token usage and switch models when budget thresholds are crossed

You can implement custom strategies by providing any object with a `select(context: RoutingContext) -> str` method.

> **See also:** [`examples/providers/llm_routing.py`](../../examples/providers/llm_routing.py). `RoutingLLMClient` docstring in `nanitics/infrastructure/llm/routing.py`.

## System Prompts

System prompts tell the LLM who it is and how to behave. Rather than hardcoding a single string, the SDK uses `SystemPromptBuilder` to compose system prompts from named sections:

```python
builder = SystemPromptBuilder()
builder.add_section("base", "You are a research assistant.")
builder.add_section("tools", "Use tools to gather information before answering.")
prompt = builder.build()
```

Sections are joined with double newlines in insertion order.

### SystemPromptContributor

Many SDK features need to inject instructions into the system prompt — working memory, planning strategies, episodic memory, and more. They do this by implementing the `SystemPromptContributor` protocol, which returns a `(section_name, content)` tuple.

Pass contributors to an agent via `prompt_contributors`. The agent combines the base `system_prompt` with all contributor sections automatically.

This is the primary composition mechanism for system prompts: rather than building one monolithic string, you compose behavior from independent contributors that each own their section.

> **See also:** [`examples/tools/system_prompt_builder.py`](../../examples/tools/system_prompt_builder.py) for the builder API, custom contributors, and assembly pattern.

## Extension Points

The agent loop integrates several optional capabilities through well-defined protocols. Each is covered in its own guide — this section shows how they compose together.

### Context Providers

Inject information into the conversation before each LLM call. The agent calls every provider's `provide()` method and inserts the results as user messages. Context providers power working memory, episodic memory, shared memory, and custom information injection.

Key design decisions:
- `priority` controls ordering — lower values appear earlier in context
- `protected` content survives context manager truncation
- Providers see the current message history, enabling reactive context injection

> **See also:** [Memory](memory.md) guide. [`examples/memory/working_memory.py`](../../examples/memory/working_memory.py).

### Output Evaluators

Judge the agent's final output and optionally trigger revision. If the verdict is `REVISE` and revision attempts remain, the agent loops back with the feedback. This creates a self-correction cycle without external intervention.

> **See also:** [Evaluation](evaluation.md) guide. [`examples/evaluation/evaluation.py`](../../examples/evaluation/evaluation.py).

### Error Handlers

Recover from LLM and tool errors through retry, correction prompts, or graceful degradation. All agents use `ErrorHandler()` by default, providing resilience out of the box. Pass `ErrorHandler.fail_fast()` to disable retry and correction during development or testing.

> **See also:** [Error Handling](error-handling.md) guide. [`examples/control/error_handling.py`](../../examples/control/error_handling.py).

### Context Managers

Manage the context window when conversations grow long — truncate old messages, summarize history, track token usage.

> **See also:** [Context Management](context-management.md) guide. [`examples/context/context_management.py`](../../examples/context/context_management.py).

## The EventEmitter

Every agent requires an `EventEmitter` to record its execution. The emitter collects all events, supports real-time listeners, and manages hierarchical spans for structured tracing. `InMemoryEmitter` is the default implementation for development and testing.

The `EventEmitter` protocol lets you implement custom emitters (e.g., streaming events over SSE, writing to a database).

> **See also:** [Observability](observability.md) guide. [`examples/tools/event_emitter.py`](../../examples/tools/event_emitter.py), [`examples/observability/trace_collection.py`](../../examples/observability/trace_collection.py).
