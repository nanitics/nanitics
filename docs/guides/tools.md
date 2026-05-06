# Tools

> For API details — signatures, fields, constraints — read the docstrings in your editor, in the source tree under [`nanitics/`](../../nanitics/), or browse them at [docs.nanitics.dev](https://docs.nanitics.dev/). `nanitics.__all__` is the authoritative public surface.

Tools give agents the ability to take actions — query databases, call APIs, read files, perform calculations. Without tools, an agent can only reason over its existing context. With tools, it can interact with the world.

> **See also:** [`examples/tools/tool_basics.py`](../../examples/tools/tool_basics.py) — runnable example covering decorator, protocol, registry, context, and error handling.

## When to Use Tools

**Use tools when** the agent needs to retrieve information, perform side effects, or execute computations it can't do through reasoning alone.

**Don't create a tool when** a context provider would serve better. If the agent always needs certain information, inject it automatically via a `ContextProvider` rather than requiring the agent to decide to call a tool. See [Context Management](context-management.md) and [Core Concepts](core-concepts.md#extension-points).

## Creating Tools

There are three ways to create tools, each suited to different situations:

### `@tool` Decorator

The fastest path. Decorate an async function with a name and description. The decorator inspects the function signature, generates a JSON Schema, and wraps it as a `FunctionTool`. Supports optional `parameters_model` for Pydantic validation. Use this for the majority of tools.

> **See also:** [`examples/tools/tool_basics.py`](../../examples/tools/tool_basics.py) — decorator usage with simple and Pydantic-validated parameters.

### `FunctionTool` Direct Construction

The class behind `@tool`. Use it when you need to create tools programmatically — for example, generating tools from a config file or database schema at runtime. Accepts either a `parameters_model` (Pydantic) or raw `parameters_schema` (JSON Schema dict).

> **See also:** [`examples/tools/tool_basics.py`](../../examples/tools/tool_basics.py) — `FunctionTool` with raw JSON Schema.

### Tool Protocol (Class-Based)

Any object with a `schema` property and an async `execute()` method satisfies the `Tool` protocol. Use this when a tool needs its own state (database connection, API client) or lifecycle management. Structural typing — no inheritance required.

> **See also:** [`examples/tools/tool_basics.py`](../../examples/tools/tool_basics.py) — class-based tool with encapsulated state.

### Which to Choose

| Approach | Use when |
|----------|----------|
| `@tool` decorator | Most tools. Simple functions, fast iteration. |
| `FunctionTool` | Programmatic tool creation, dynamic schemas. |
| Tool protocol class | Tool needs its own state or lifecycle. |
| Reference tool (`nanitics.tools.*`) | A shipped factory (`create_web_search_tool`, `create_http_tool`, `create_file_read_tool`, `create_code_execution_tool`) already covers the capability. See [Built-in Tools](built-in-tools.md). |
| MCP tool | An external MCP server exposes the capability (filesystem, git, Postgres, Slack, etc.). See [MCP Tools](#mcp-tools). |

## Parameter Validation

The two parameter paths offer different validation guarantees:

| Path | Validation | Error on invalid input |
|------|-----------|----------------------|
| `parameters_model` (Pydantic, used by `@tool`) | Full type checking, constraints, detailed messages | `ToolParameterError` before execution |
| `parameters_schema` (raw JSON Schema dict) | Required-parameter presence only | `ToolExecutionError` during execution |

Use `@tool` or `parameters_model` for full validation. Use `parameters_schema` only when constructing tools dynamically and you're confident the LLM will send well-formed parameters.

## Tool State and Context

Tools sometimes need access to the runtime environment — the event emitter for tracing, or shared state between tools. `ToolContext` provides this without globals.

Add a parameter with a `ToolContext` type annotation to any tool function. It's automatically detected and injected — it doesn't appear in the tool schema the LLM sees. The parameter name doesn't matter; only the type annotation matters.

**Tool state** lets you inject per-run data (database sessions, user context, accumulators) into tools via `tool_state` on the agent constructor. All tools in the same run share the same state dict, accessible through `ctx.state`.

> **See also:** [`examples/tools/tool_basics.py`](../../examples/tools/tool_basics.py) — `ToolContext` injection and shared tool state.

## Tool Registries

The `ToolRegistry` manages a collection of tools, dispatches tool calls from the LLM, and handles validation and event emission. You rarely create one directly — the agent builds one internally from the `tools` list you provide.

Registries are useful when you need to:
- **Group tools** into logical sets and swap them between runs
- **Share an emitter** across all tools for consistent tracing
- **Inject tool state** that all tools in a run can access

> **See also:** [`examples/tools/tool_basics.py`](../../examples/tools/tool_basics.py) — manual registry creation and dispatch.

## MCP Tools

[MCP](https://modelcontextprotocol.io) (the Model Context Protocol) is a standard way for external servers to expose tools to LLM-driven clients. Nanitics acts as an MCP *client*: it connects to any MCP-compatible server — filesystem, git, Postgres, Slack, GitHub, the Fetch server, Sequential Thinking, and many more — and exposes the server's tools as ordinary `Tool` instances. The agent loop, `ToolRegistry`, event emission, and error handling are unchanged. MCP tools are FunctionTools that happen to be backed by a remote process.

Install the optional extra:

```bash
pip install nanitics[mcp]
```

### Connecting

`MCPClient` is an async context manager. Entering the context spawns the transport (stdio subprocess or SSE connection), runs the MCP handshake, and discovers tools. Exiting the context closes the session and disposes of the transport.

<!-- verify: skip — illustrative usage; `llm` and `emitter` are caller-supplied and the `async with` / `await` run inside an async context -->
```python
from nanitics import MCPClient, MCPStdioParameters, ReActAgent

params = MCPStdioParameters(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
)

async with MCPClient.stdio(params) as client:
    tools = await client.list_tools()
    agent = ReActAgent(
        name="fs-agent",
        llm_client=llm,
        emitter=emitter,
        system_prompt="Use the filesystem tools to help the user.",
        tools=tools,
    )
    result = await agent.run("Show me the first few files in /tmp.")
```

For SSE servers, use `MCPClient.sse(url=..., headers=...)` with the same `async with` shape.

### Name Management

Multiple MCP servers can expose tools with the same name (e.g., two servers with a `search` tool). Two knobs resolve this at discovery time:

- **`name_prefix`** — prepended to every discovered tool's name. Use `name_prefix="fs_"` and `name_prefix="git_"` on two clients to get `fs_search` and `git_search` without collision. The prefix is transparent to the MCP server — calls use the original server-side name.
- **`name_filter`** — a predicate over the server-side name; tools for which it returns `False` are skipped during discovery. Use this to whitelist (or blacklist) specific tools from a server that exposes more than you want to offer the agent.

<!-- verify: skip — illustrative fragment; `params` is from the previous snippet and the `async with` / `await` run inside an async context -->
```python
async with MCPClient.stdio(
    params,
    name_prefix="fs_",
    name_filter=lambda name: name in {"read_file", "list_directory"},
) as client:
    tools = await client.list_tools()
```

### Lifecycle

Tools returned by `list_tools()` are bound to the session owned by the enclosing `async with` block. Calling `.execute()` on an `MCPTool` after the block exits raises `LLMProviderError` — the underlying subprocess is gone and the streams are closed. If your agent needs the tools across multiple runs, keep the `async with` open around the full agent lifetime.

Two timeouts bound the connection:

- **`discovery_timeout`** (default 30s) — bounds the MCP initialization handshake and `tools/list` combined. On timeout, `LLMProviderError(provider="mcp")` is raised.
- **`default_call_timeout`** (default 60s) — bounds each `execute()` call. Server-declared per-tool timeouts (rare) in `ToolSchema.timeout_seconds` override this. On timeout, `ToolTimeoutError` is raised.

### Scope

This integration is client-only. Out of scope for now:

- MCP server mode (Nanitics agents are not exposed as MCP tools).
- MCP resources and prompts (only tools are supported).
- MCP sampling / elicitation callbacks.
- Dynamic `tools/list_changed` re-discovery — the tool list is cached after the first call.
- Streamable-HTTP transport — only stdio and SSE are supported.

> **See also:** [`examples/tools/mcp_tools.py`](../../examples/tools/mcp_tools.py) — runnable demo using an in-process MCP server; includes a commented real-stdio section. See the `MCPClient` docstring for the full API surface.

## Reference Tools

Nanitics ships four curated tools in `nanitics.tools` for capabilities most agents need — web search, HTTP calls, file reading, and code execution. They satisfy the same `Tool` protocol as `FunctionTool` and dispatch through `ToolRegistry` identically. No new registry, no new event types; `ToolInvokeEvent` / `ToolResultEvent` are emitted through the standard path.

Each tool ships with a safe default posture: explicit allow-lists, bounded response sizes, configurable timeouts, no network unless opted in. Misuse fails closed — a construction-time `ValueError` if security prerequisites are missing, or a per-call `ToolParameterError` that the LLM can see and correct.

Install everything needed by the HTTP-based reference tools with the umbrella extra:

```bash
pip install nanitics[tools]
```

### `create_web_search_tool`

```python
from nanitics import create_web_search_tool

tool = create_web_search_tool(api_key="...", provider="tavily")  # or provider="brave"
```

Requires `pip install nanitics[http-tools]`. Supports Tavily (default) and Brave backends selected by the `provider` argument. Results are normalized into a common `{title, url, snippet, score}` shape; the LLM sees a bulleted markdown rendering and application code sees the structured result via `ToolResult.metadata` (see `WebSearchResult`). Defaults: `max_results_default=5`, `request_timeout=30.0`. The factory requires a non-empty `api_key` — there is no environment-variable fallback.

> **See also:** [`examples/tools/web_search_tool.py`](../../examples/tools/web_search_tool.py).

### `create_http_tool`

```python
from nanitics import create_http_tool

tool = create_http_tool(allowed_domains=["api.example.com"])
```

Requires `pip install nanitics[http-tools]`. Construction requires either a non-empty `allowed_domains` list or the explicit `allow_any_domain=True` escape hatch — host matching is exact and case-insensitive. 4xx and 5xx statuses are NOT raised; they are surfaced via `metadata["status"]` so the LLM can read and adapt. Only transport failures, timeouts, and disallowed domains raise. Defaults: `request_timeout=30.0`, `max_response_bytes=1_048_576` (1 MiB) with truncation flagged in metadata; redirects are followed.

> **See also:** [`examples/tools/http_file_tools.py`](../../examples/tools/http_file_tools.py).

### `create_file_read_tool`

```python
from nanitics import create_file_read_tool

tool = create_file_read_tool(allowed_paths=["/srv/data"])
```

No optional extras required. Construction requires a non-empty `allowed_paths` list; every entry is resolved once via `pathlib.Path.resolve()` and compared against the resolved request path (symlinks are followed). Requests that resolve outside every allowed root raise `ToolParameterError`. UTF-8-decodable files return as text; non-UTF-8 payloads come back as a base64 string with `metadata["encoding"] == "base64"`. Defaults: `max_bytes=1_048_576` (1 MiB), ceiling `104_857_600` (100 MiB).

> **See also:** [`examples/tools/http_file_tools.py`](../../examples/tools/http_file_tools.py).

### `create_code_execution_tool`

<!-- verify: skip — illustrative sketch with `...` placeholder; the `async with` runs inside an async context -->
```python
from nanitics import DockerSandbox, SandboxConfig, create_code_execution_tool

sandbox = DockerSandbox(config=SandboxConfig())
async with sandbox:
    tool = create_code_execution_tool(sandbox=sandbox)
    # ... run agent ...
```

No optional extras required by the factory itself — the user's choice of sandbox is what pulls in Docker (via the existing `code_execution` extra when using `DockerSandbox`). The tool does NOT own the sandbox lifecycle; enter the sandbox's async context manager around the agent run. Sandbox-level failures (`ExecutionResult.success is False`) do NOT raise — they are surfaced through `metadata["success"]` and an `error:` prefix in `content` so the LLM can read stderr and try again. Only unexpected exceptions raised by the sandbox implementation are wrapped in `ToolExecutionError`.

> **See also:** [`examples/tools/code_execution_tool.py`](../../examples/tools/code_execution_tool.py).

See [Built-in Tools](built-in-tools.md) for the full catalog, security-defaults table, and the decision frame for picking a reference tool vs MCP vs a custom tool.

## Best Practices

### Naming

Tool names should be verb-noun: `get_weather`, `search_documents`, `create_ticket`. The LLM uses the name to decide what the tool does. Avoid generic names like `do_thing` or `helper`.

### Descriptions

Write descriptions for the LLM, not for humans. Be specific about what the tool does, when to use it, what parameters mean, and what output looks like.

<!-- verify: skip — illustrative good-vs-bad contrast; bare `@tool(...)` decorators without a following function body -->
```python
# Bad — too vague
@tool("search", "Search for things")

# Good — specific and actionable
@tool("search_docs", "Search the documentation knowledge base. Returns matching documents ranked by relevance. Use this when the user asks about how something works.")
```

### Parameter Design

- Use descriptive parameter names (`query` not `q`, `max_results` not `n`)
- Provide defaults for optional parameters
- Use Pydantic models with `Field(description=...)` for complex parameters
- Keep the number of parameters small — the LLM has to figure out what to pass
- **Constrain string parameters to known values with `Literal` types** — when a parameter accepts a fixed set of values, use `Literal` instead of `str`. This emits a JSON Schema `enum` constraint so the LLM sees the valid options directly in the tool schema, and Pydantic rejects invalid values before the function body runs (raising `ToolParameterError`).

```python
from typing import Literal

# Define the Literal type
ComparableField = Literal["name", "address", "email"]

# Derive a set for use elsewhere (lookups, iteration)
COMPARABLE_FIELDS = set(ComparableField.__args__)

@tool("compare", "Compare a field between two records")
async def compare(field: ComparableField, source: str, target: str) -> str:
    # No runtime validation needed — Pydantic rejects invalid values
    ...
```

The resulting JSON Schema includes `"enum": ["name", "address", "email"]` on the `field` parameter, giving the LLM explicit guidance on valid values. This replaces the pattern of accepting `str` and validating at runtime with an `if field not in ALLOWED_VALUES` guard.

### Return Values

- Return structured, readable text — the LLM parses this to decide what to do next
- Include relevant context ("Found 3 results" is better than just listing them)
- Keep output concise — large tool results consume context window tokens
- For errors, return a clear error message as a string rather than raising (unless truly unrecoverable)

### Structured Tool Errors

When a tool needs to raise — for example, to surface a typed, structured failure that the agent should reason about — subclass `ToolError` (`from nanitics import ToolError`) rather than raising bare `Exception` or `RuntimeError`. The default classifier in `nanitics.capabilities.errors.classification.classify_error` treats every `ToolError` subclass as `CORRECTABLE` by default, so the correction loop receives the error and the agent gets a chance to self-correct on the next iteration. App-defined typed errors carrying domain fields (entity ids, validation reasons, retry hints) inherit this behavior without per-class registration. The one documented exception is `ToolTimeoutError`, which is classified as `RETRYABLE`; refer to each error class's docstring for the authoritative per-class category. See [Error Handling](error-handling.md) for the full hierarchy and recovery model.

## Pitfalls

**Tools with side effects:** The LLM may call a tool multiple times or in unexpected order. Design tools to be safe for repeated calls, or use approval wrapping. See [Human-in-the-Loop](human-in-the-loop.md).

**Tools that return too much data:** The full tool result goes into the conversation history. If a tool returns 10,000 lines, the context window fills up fast. Truncate or summarize within the tool.

**Ambiguous tool names:** If two tools have similar names or descriptions, the LLM may pick the wrong one. Make names and descriptions clearly distinct.

**Missing ToolContext type annotation:** If you forget the `ToolContext` type hint, the parameter will appear in the tool schema as a regular parameter, and the LLM will try to pass a value for it.

## See Also

- [`examples/tools/tool_basics.py`](../../examples/tools/tool_basics.py) — runnable example covering the full tool system
- [Agent Types](agent-types.md) — different agent types use tools in different ways
- [Error Handling](error-handling.md) — how the agent recovers when tools fail
- [Human-in-the-Loop](human-in-the-loop.md) — approval wrapping for sensitive tools
- [Core Concepts](core-concepts.md) — how tools fit into the agent loop
