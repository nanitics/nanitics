"""Tool system fundamentals: creating, validating, registering, dispatching, and observing tools.

Covers the @tool decorator, ToolResult with metadata, Pydantic validation, the Tool protocol
for stateful tools, ToolRegistry dispatch, ToolContext injection, and error handling.

Related guide: docs/guides/tools.md
"""

import asyncio

from pydantic import BaseModel, Field

from nanitics import (
    InMemoryEmitter,
    ToolCall,
    ToolContext,
    ToolError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolParameterError,
    ToolRegistry,
    ToolResult,
    ToolSchema,
    tool,
)
from nanitics.infrastructure import (
    ToolInvokeEvent,
    ToolResultEvent,
)

# --- Section 1: Minimal Tool (@tool decorator) ---


@tool("get_weather", "Get the current weather for a city")
async def get_weather(city: str) -> str:
    return f"Sunny, 22°C in {city}"


# --- Section 2: ToolResult with Metadata ---


@tool("lookup_user", "Look up a user by their ID")
async def lookup_user(user_id: str) -> ToolResult:
    # Simulate a user lookup
    users = {
        "u-123": {"name": "Alice", "email": "alice@example.com", "role": "admin"},
        "u-456": {"name": "Bob", "email": "bob@example.com", "role": "viewer"},
    }
    user = users.get(user_id)
    if user is None:
        return ToolResult(content=f"No user found with ID '{user_id}'")

    return ToolResult(
        content=f"User {user['name']} ({user['role']})",
        metadata={"email": user["email"], "role": user["role"]},
    )


# --- Section 3: Pydantic Model Validation ---


class SearchParams(BaseModel):
    query: str = Field(description="Search query")
    max_results: int = Field(default=5, ge=1, le=20, description="Number of results to return")


@tool("search_documents", "Search the document index", parameters_model=SearchParams)
async def search_documents(query: str, max_results: int = 5) -> str:
    return f"Found {max_results} results for '{query}'"


# --- Section 4: Tool Protocol (Stateful Tools) ---


class HitCounter:
    """A stateful tool that counts page visits. Demonstrates the Tool protocol."""

    def __init__(self) -> None:
        self._counts: dict[str, int] = {}

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="count_hit",
            description="Record and return the visit count for a page",
            parameters={
                "type": "object",
                "properties": {
                    "page": {"type": "string", "description": "Page path to count"},
                },
                "required": ["page"],
            },
            timeout_seconds=2.0,
        )

    async def execute(self, **params: str) -> ToolResult:
        page = params["page"]
        self._counts[page] = self._counts.get(page, 0) + 1
        return ToolResult(
            content=f"Page '{page}' has {self._counts[page]} visit(s)",
            metadata={"page": page, "count": self._counts[page]},
        )


# --- Section 5: Error Handling Tools ---


@tool("restricted_action", "Perform a restricted action")
async def restricted_action(action: str) -> str:
    raise ToolError(f"Not authorized to perform: {action}")


@tool("buggy_tool", "A tool with a bug")
async def buggy_tool() -> str:
    raise ValueError("unexpected internal error")


# --- Section 6: ToolContext Tools ---


@tool("add_to_cart", "Add an item to the shopping cart")
async def add_to_cart(item: str, ctx: ToolContext) -> str:
    cart = ctx.state["cart"]
    cart.append(item)
    return f"Added '{item}'. Cart now has {len(cart)} item(s)."


async def main() -> None:
    # --- Section 1: Minimal Tool ---
    print("--- Section 1: Minimal Tool (@tool decorator) ---")

    # The @tool decorator auto-generates a schema from the function signature
    assert get_weather.schema.name == "get_weather"
    assert get_weather.schema.description == "Get the current weather for a city"
    params = get_weather.schema.parameters
    assert "city" in params["properties"]
    assert params["properties"]["city"]["type"] == "string"
    assert "city" in params.get("required", [])
    print(f"  Schema: {get_weather.schema.name}({', '.join(params.get('required', []))})")

    # Execute directly — plain strings are auto-wrapped in ToolResult
    result = await get_weather.execute(city="Tokyo")
    assert isinstance(result, ToolResult)
    assert result.content == "Sunny, 22°C in Tokyo"
    print(f"  Result: {result.content}")

    # --- Section 2: ToolResult with Metadata ---
    print("\n--- Section 2: ToolResult with Metadata ---")

    result = await lookup_user.execute(user_id="u-123")
    # content is what the LLM sees
    assert "Alice" in result.content
    assert "admin" in result.content
    # metadata is for application logic — not sent to the LLM
    assert result.metadata["email"] == "alice@example.com"
    assert result.metadata["role"] == "admin"
    print(f"  Content (LLM sees): {result.content}")
    print(f"  Metadata (app only): {result.metadata}")

    # Not-found case returns content only, no metadata
    result = await lookup_user.execute(user_id="u-999")
    assert "No user found" in result.content
    print(f"  Not found: {result.content}")

    # --- Section 3: Pydantic Model Validation ---
    print("\n--- Section 3: Pydantic Model Validation ---")

    # Schema includes Field descriptions and constraints
    schema = search_documents.schema
    props = schema.parameters["properties"]
    assert "description" in props["query"]
    assert props["max_results"]["default"] == 5
    print(f"  Schema params: {list(props.keys())}")

    # Valid parameters work
    result = await search_documents.execute(query="agents", max_results=3)
    assert "3 results" in result.content
    print(f"  Valid call: {result.content}")

    # Invalid parameters raise ToolParameterError before the function runs
    try:
        await search_documents.execute(query="agents", max_results=100)
        assert False, "Should have raised ToolParameterError"
    except ToolParameterError as e:
        assert "search_documents" in str(e)
        print(f"  Invalid params caught: {type(e).__name__}")

    # --- Section 4: Tool Protocol (Stateful Tools) ---
    print("\n--- Section 4: Tool Protocol (Stateful Tools) ---")

    counter = HitCounter()

    # Schema supports features not available via @tool (e.g., timeout_seconds)
    assert counter.schema.timeout_seconds == 2.0
    assert counter.schema.name == "count_hit"
    print(f"  Schema: {counter.schema.name} (timeout={counter.schema.timeout_seconds}s)")

    # State persists across calls
    result = await counter.execute(page="/home")
    assert result.metadata["count"] == 1
    print(f"  Hit 1: {result.content}")
    result = await counter.execute(page="/home")
    assert result.metadata["count"] == 2
    print(f"  Hit 2: {result.content}")

    # Different pages have independent counts
    result = await counter.execute(page="/about")
    assert result.metadata["count"] == 1

    # --- Section 5: ToolRegistry — Registration and Dispatch ---
    print("\n--- Section 5: ToolRegistry ---")

    emitter = InMemoryEmitter(trace_id="example-trace")
    registry = ToolRegistry(emitter=emitter)
    registry.register(get_weather)
    registry.register(lookup_user)
    registry.register(counter)

    # List schemas — all registered tools appear
    schemas = registry.list_schemas()
    names = {s.name for s in schemas}
    assert names == {"get_weather", "lookup_user", "count_hit"}
    print(f"  Registered: {', '.join(sorted(names))}")

    # Dispatch a ToolCall (the same way the agent loop does)
    result = await registry.dispatch(ToolCall(id="call-1", name="get_weather", arguments={"city": "Paris"}))
    assert result.content == "Sunny, 22°C in Paris"
    print(f"  Dispatch result: {result.content}")

    # Dispatching an unknown tool raises ToolNotFoundError
    try:
        await registry.dispatch(ToolCall(id="call-2", name="nonexistent", arguments={}))
        assert False, "Should have raised ToolNotFoundError"
    except ToolNotFoundError as e:
        assert e.tool_name == "nonexistent"
        print(f"  Unknown tool caught: {type(e).__name__}(tool_name={e.tool_name!r})")

    # Events were emitted for observability
    invoke_events = [e for e in emitter.events if isinstance(e, ToolInvokeEvent)]
    result_events = [e for e in emitter.events if isinstance(e, ToolResultEvent)]
    assert len(invoke_events) >= 1
    assert invoke_events[0].tool_name == "get_weather"
    assert any(e.tool_name == "get_weather" and e.success for e in result_events)
    print(f"  Events: {len(invoke_events)} invoke, {len(result_events)} result")

    # --- Section 6: ToolContext — Emitter and Shared State ---
    print("\n--- Section 6: ToolContext ---")

    cart: list[str] = []
    ctx_emitter = InMemoryEmitter(trace_id="cart-trace")
    ctx_registry = ToolRegistry(emitter=ctx_emitter, tool_state={"cart": cart})
    ctx_registry.register(add_to_cart)

    # ToolContext parameter is excluded from the schema the LLM sees
    param_names = set(add_to_cart.schema.parameters.get("properties", {}).keys())
    assert "ctx" not in param_names
    assert "item" in param_names
    print(f"  Schema params: {sorted(param_names)} (ctx excluded)")

    # Dispatch two calls — shared state accumulates across calls
    await ctx_registry.dispatch(ToolCall(id="c-1", name="add_to_cart", arguments={"item": "eggs"}))
    await ctx_registry.dispatch(ToolCall(id="c-2", name="add_to_cart", arguments={"item": "milk"}))
    assert cart == ["eggs", "milk"]
    print(f"  Shared state after 2 calls: cart={cart}")

    # ToolContext also provides the emitter for tool-level observability
    ctx_tool_events = [e for e in ctx_emitter.events if isinstance(e, ToolInvokeEvent)]
    assert len(ctx_tool_events) == 2, "Emitter captured tool invocations"
    assert ctx_tool_events[0].tool_name == "add_to_cart"
    print(f"  ctx.emitter captured {len(ctx_tool_events)} tool invocations ✓")

    # --- Section 7: Error Handling ---
    print("\n--- Section 7: Error Handling ---")

    err_emitter = InMemoryEmitter(trace_id="error-trace")
    err_registry = ToolRegistry(emitter=err_emitter)
    err_registry.register(restricted_action)
    err_registry.register(buggy_tool)

    # ToolError (expected, handleable) — re-raised as-is
    try:
        await err_registry.dispatch(ToolCall(id="e-1", name="restricted_action", arguments={"action": "delete"}))
        assert False, "Should have raised ToolError"
    except ToolError as e:
        print(f"  Expected error: {type(e).__name__}: {e}")

    # The failure was recorded in events
    err_results = [e for e in err_emitter.events if isinstance(e, ToolResultEvent)]
    assert any(not e.success and "Not authorized" in (e.error or "") for e in err_results)

    # Unexpected exceptions are wrapped in ToolExecutionError
    try:
        await err_registry.dispatch(ToolCall(id="e-2", name="buggy_tool", arguments={}))
        assert False, "Should have raised ToolExecutionError"
    except ToolExecutionError as e:
        assert e.tool_name == "buggy_tool"
        assert isinstance(e.__cause__, ValueError)
        print(f"  Unexpected wrapped: {type(e).__name__}(cause={type(e.__cause__).__name__})")

    print("\nAll sections passed ✓")


if __name__ == "__main__":
    asyncio.run(main())
