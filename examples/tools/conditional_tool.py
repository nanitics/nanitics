"""ConditionalTool: state-driven tool visibility.

Demonstrates wrapping tools with ConditionalTool so their visibility in the
schema list depends on runtime tool_state. Covers basic wrapping, registry
filtering, and state changes causing tools to appear and disappear.

Related guide: docs/guides/tools.md
"""

import asyncio

from nanitics.specialized import ConditionalTool
from nanitics.strategies import (
    ToolRegistry,
    tool,
)


@tool("search_records", "Search for records by keyword")
async def search_records(keyword: str) -> str:
    return f"Found 3 records matching '{keyword}'"


@tool("delete_record", "Delete a record by ID")
async def delete_record(record_id: str) -> str:
    return f"Deleted record {record_id}"


async def main() -> None:
    # --- Section 1: Basic ConditionalTool ---
    print("--- Section 1: Basic ConditionalTool ---")

    conditional_delete = ConditionalTool(
        tool=delete_record,
        is_enabled=lambda state: state.get("approved", False),
    )

    # Schema delegates to the inner tool unchanged
    assert conditional_delete.schema.name == "delete_record"
    assert conditional_delete.schema.description == "Delete a record by ID"
    print("✓ Schema delegates to inner tool unchanged")

    # Execute delegates to the inner tool
    result = await conditional_delete.execute(record_id="rec-42")
    assert result.content == "Deleted record rec-42"
    print(f"✓ Execute delegates to inner tool: {result.content}")

    # Predicate controls visibility based on state
    assert conditional_delete.is_enabled({}) is False
    assert conditional_delete.is_enabled({"approved": True}) is True
    print("✓ Predicate evaluates against state dict")

    # --- Section 2: Registry Filtering ---
    print("\n--- Section 2: Registry Filtering ---")

    registry = ToolRegistry(tool_state={})
    registry.register(search_records)  # Always visible
    registry.register(conditional_delete)  # Visible only when approved

    schemas = registry.list_schemas()
    schema_names = [s.name for s in schemas]
    assert "search_records" in schema_names
    assert "delete_record" not in schema_names
    print(f"✓ Without approval: visible tools = {schema_names}")

    # --- Section 3: State-Driven Visibility ---
    print("\n--- Section 3: State-Driven Visibility ---")

    # Simulate approval by creating a new registry with updated state
    approved_state: dict[str, object] = {"approved": True}
    registry_approved = ToolRegistry(tool_state=approved_state)
    registry_approved.register(search_records)
    registry_approved.register(conditional_delete)

    schemas = registry_approved.list_schemas()
    schema_names = [s.name for s in schemas]
    assert "search_records" in schema_names
    assert "delete_record" in schema_names
    print(f"✓ With approval: visible tools = {schema_names}")

    print("\nAll sections passed ✓")


if __name__ == "__main__":
    asyncio.run(main())
