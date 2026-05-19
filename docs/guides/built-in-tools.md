# Built-in Tools

> For API details — signatures, fields, constraints — read the docstrings in your editor, in the source tree under [`nanitics/`](../../nanitics/), or browse them at [docs.nanitics.dev](https://docs.nanitics.dev/). `nanitics.__all__` is the authoritative public surface.

The SDK ships four curated tools in `nanitics.tools` — web search, HTTP requests, file reading, and code execution. They exist so you don't have to rebuild the things every agent needs. Use them when they fit; reach for [MCP](tools.md#mcp-tools) for ecosystem breadth; write a [custom tool](tools.md#creating-tools) when your domain has something the other two can't express. The decision order matters: a well-scoped reference tool beats a one-off custom wrapper every time.

Install the optional dependencies for the HTTP-backed tools with:

```bash
pip install nanitics[tools]
```

## Reference tool catalog

### `create_web_search_tool`

Backed by Tavily or Brave. Pass `api_key` (required) and optionally `provider="tavily"` / `"brave"`. Results normalize to a `{title, url, snippet, score}` shape the LLM sees as a bulleted markdown list; structured results round-trip via `ToolResult.metadata` as `WebSearchResult`. Defaults: five results, 30-second timeout. There is no environment-variable fallback — the empty-key case raises `ValueError` at construction. See [`examples/tools/web_search_tool.py`](../../examples/tools/web_search_tool.py).

### `create_http_tool`

Generic HTTP client with an explicit domain allow-list. Supply `allowed_domains=[...]` (exact, case-insensitive hosts) or the explicit `allow_any_domain=True` escape hatch for trusted environments. Responses are bounded by `max_response_bytes` (default 1 MiB) and truncated with metadata flags. 4xx and 5xx responses are NOT raised — status is surfaced via `metadata["status"]` so the LLM can read and adapt. Only transport failures, timeouts, and disallowed domains raise. See [`examples/tools/http_file_tools.py`](../../examples/tools/http_file_tools.py).

### `create_file_read_tool`

Filesystem reader gated by a resolved path allow-list. `allowed_paths` is required and non-empty; every entry is resolved once at construction and compared against the resolved request path (symlinks are followed). UTF-8 text is returned as-is; binary payloads are base64-encoded with `metadata["encoding"] == "base64"`. Default `max_bytes=1_048_576` with a hard ceiling of 100 MiB. See [`examples/tools/http_file_tools.py`](../../examples/tools/http_file_tools.py).

### `create_code_execution_tool`

Runs Python code inside any `Sandbox` — `DockerSandbox` for production isolation, `MockSandbox` in tests. The factory does NOT own the sandbox lifecycle; enter the sandbox's `async with` block around the agent run. Sandbox-level failures (`ExecutionResult.success is False`) do NOT raise — they are surfaced through `metadata["success"]` and an `error:` prefix in `content` so the LLM reads stderr and iterates. See [`examples/tools/code_execution_tool.py`](../../examples/tools/code_execution_tool.py).

## Security defaults

| Tool | Required arg | Timeout | Body cap | Allow-list policy |
|------|--------------|---------|----------|-------------------|
| `create_web_search_tool` | `api_key` | 30s | provider-bounded | N/A |
| `create_http_tool` | `allowed_domains` or `allow_any_domain=True` | 30s | 1 MiB | exact host, case-insensitive |
| `create_file_read_tool` | `allowed_paths` | N/A | 1 MiB (per call, `max_bytes`) | resolved-path containment |
| `create_code_execution_tool` | `sandbox` | sandbox-owned | sandbox-owned | sandbox-owned |

Misuse fails closed — missing or empty allow-lists raise `ValueError` at construction; disallowed paths or domains at call time raise `ToolParameterError` so the LLM sees actionable feedback and can correct.

## Parsing result metadata

Each tool's `ToolResult.metadata` is a dict that round-trips through a frozen Pydantic model — `WebSearchResult`, `HttpResponse`, `FileReadResult`, or `CodeExecutionResult`. Application code validates the shape in one line:

```python
from nanitics.strategies import ToolResult
from nanitics.tools import HttpResponse

def render(result: ToolResult) -> str:
    parsed = HttpResponse.model_validate(result.metadata)
    return f"{parsed.status} {parsed.url}"
```

The LLM never sees these models — it reads `ToolResult.content`. The typed metadata is for your code.

## Reference tool vs MCP vs custom

| When | Pick |
|------|------|
| One of the four tools fits — web search, HTTP, file read, Python execution | Reference tool |
| You need breadth the reference set doesn't cover — filesystem ops, git, Postgres, Slack, GitHub, etc. | [MCP](tools.md#mcp-tools) |
| Your domain has something nothing else expresses — a specific API, a proprietary database, an internal RPC | [Custom tool](tools.md#creating-tools) |

Prefer the reference tool when one fits because its security posture is already vetted. Prefer MCP when you need the ecosystem breadth; prefer a custom tool when the behavior is domain-specific. Don't reach for a custom tool just to avoid a dependency — the allow-list defaults and result models of the reference tools are worth the `pip install`.

## Writing your own

`tools.md` covers the three creation methods — `@tool` decorator, `FunctionTool` direct construction, and the class-based `Tool` protocol. This guide does not restate them. If the four reference tools don't fit and no MCP server exists for your case, go there.

## See also

- [Tools](tools.md) — creation methods, parameter validation, `ToolContext`, MCP integration
- [Safety](safety.md) — sandbox selection for `create_code_execution_tool`
- [`examples/tools/web_search_tool.py`](../../examples/tools/web_search_tool.py) — runnable web search with Tavily
- [`examples/tools/http_file_tools.py`](../../examples/tools/http_file_tools.py) — HTTP and file-read tools in one agent
- [`examples/tools/code_execution_tool.py`](../../examples/tools/code_execution_tool.py) — Python execution via `MockSandbox` and `DockerSandbox`
- [`examples/tools/mcp_tools.py`](../../examples/tools/mcp_tools.py) — MCP alternative when the reference set doesn't fit
