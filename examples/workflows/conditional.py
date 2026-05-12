"""Conditional workflow: routing input to branches based on a router function.

Demonstrates ``Conditional`` — an orchestration pattern that evaluates input through a router
function and executes the matching branch. Supports sync and async routers, default
fallback branches, and nested workflows within branches.

Related guide: docs/guides/orchestration.md
"""

import asyncio

from examples.helpers import make_emitter
from nanitics import (
    FunctionStep,
    Sequential,
    WorkflowStep,
)
from nanitics.experimental import Conditional
from nanitics.infrastructure import (
    WorkflowCompleteEvent,
    WorkflowStartEvent,
    WorkflowStepCompleteEvent,
)


async def main() -> None:
    # --- Section 1: Basic Conditional Routing ---
    print("--- Section 1: Basic Conditional Routing ---")

    # Router classifies input text by word count into one of three branches.
    # Each branch applies a different transformation.

    def route_by_length(text: str) -> str:
        word_count = len(text.split())
        if word_count <= 3:
            return "short"
        if word_count <= 8:
            return "medium"
        return "long"

    async def handle_short(text: str) -> str:
        return f"SHORT: {text.upper()}"

    async def handle_medium(text: str) -> str:
        return f"MEDIUM: {text.title()}"

    async def handle_long(text: str) -> str:
        return f"LONG: {text[:50]}..."

    emitter = make_emitter("conditional-s1")

    conditional = Conditional(
        name="text-classifier",
        router=route_by_length,
        branches={
            "short": FunctionStep("short", handle_short),
            "medium": FunctionStep("medium", handle_medium),
            "long": FunctionStep("long", handle_long),
        },
        emitter=emitter,
    )

    # Short input routes to the "short" branch
    result = await conditional.execute("hello world")

    assert result.output == "SHORT: HELLO WORLD"
    assert result.metadata["selected_branch"] == "short"
    assert result.metadata["total_steps_executed"] == 1

    print(f"  Input: 'hello world' → branch: {result.metadata['selected_branch']}")
    print(f"  Output: {result.output}")

    # Medium input routes to the "medium" branch
    # New Conditional instance required — each instance binds its emitter at construction
    emitter = make_emitter("conditional-s1b")

    conditional = Conditional(
        name="text-classifier",
        router=route_by_length,
        branches={
            "short": FunctionStep("short", handle_short),
            "medium": FunctionStep("medium", handle_medium),
            "long": FunctionStep("long", handle_long),
        },
        emitter=emitter,
    )

    result = await conditional.execute("the quick brown fox jumps")

    assert result.output == "MEDIUM: The Quick Brown Fox Jumps"
    assert result.metadata["selected_branch"] == "medium"
    assert result.metadata["total_steps_executed"] == 1

    print(f"  Input: 'the quick brown fox jumps' → branch: {result.metadata['selected_branch']}")
    print(f"  Output: {result.output}")
    print("✓ Sync router selects branches based on word count")

    # --- Section 2: Async Router ---
    print("\n--- Section 2: Async Router ---")

    # Async routers follow the same pattern — useful when routing logic
    # needs I/O (e.g., checking a cache or database).

    async def async_route(text: str) -> str:
        word_count = len(text.split())
        return "short" if word_count <= 3 else "long"

    async def format_short(text: str) -> str:
        return f"[short] {text}"

    async def format_long(text: str) -> str:
        return f"[long] {text}"

    emitter = make_emitter("conditional-s2")

    conditional = Conditional(
        name="async-router",
        router=async_route,
        branches={
            "short": FunctionStep("short", format_short),
            "long": FunctionStep("long", format_long),
        },
        emitter=emitter,
    )

    result = await conditional.execute("hi there")

    assert result.output == "[short] hi there"
    assert result.metadata["selected_branch"] == "short"

    print(f"  Async router selected: {result.metadata['selected_branch']}")
    print(f"  Output: {result.output}")
    print("✓ Async router function works identically to sync")

    # --- Section 3: Default Branch Fallback ---
    print("\n--- Section 3: Default Branch Fallback ---")

    # When the router returns a branch name not present in `branches`,
    # the `default` step handles the input. The metadata records the
    # original (unknown) branch name in parentheses.

    def always_unknown(text: str) -> str:
        return "spanish"

    async def fallback_handler(text: str) -> str:
        return f"fallback: {text}"

    emitter = make_emitter("conditional-s3")

    conditional = Conditional(
        name="with-default",
        router=always_unknown,
        branches={
            "english": FunctionStep("english", format_short),
            "french": FunctionStep("french", format_long),
        },
        default=FunctionStep("fallback", fallback_handler),
        emitter=emitter,
    )

    result = await conditional.execute("hola mundo")

    assert result.output == "fallback: hola mundo"
    assert result.metadata["selected_branch"] == "default(spanish)"

    print("  Router returned: 'spanish' (not in branches)")
    print(f"  Metadata: selected_branch = {result.metadata['selected_branch']!r}")
    print(f"  Output: {result.output}")
    print("✓ Default step catches unknown branch names")

    # --- Section 4: Missing Branch Error ---
    print("\n--- Section 4: Missing Branch Error ---")

    # Without a default, an unknown branch name raises ValueError.
    # The error message lists available branches.

    emitter = make_emitter("conditional-s4")

    conditional = Conditional(
        name="no-default",
        router=always_unknown,
        branches={
            "english": FunctionStep("english", format_short),
            "french": FunctionStep("french", format_long),
        },
        emitter=emitter,
    )

    try:
        await conditional.execute("hola mundo")
        assert False, "Should have raised ValueError"
    except ValueError as exc:
        assert "spanish" in str(exc)
        assert "english" in str(exc)
        assert "french" in str(exc)
        print(f"  Error: {exc}")

    print("✓ Missing branch without default raises ValueError with available branches")

    # --- Section 5: Nested Workflows as Branches ---
    print("\n--- Section 5: Nested Workflows as Branches ---")

    # Branches can contain arbitrary workflows via WorkflowStep.
    # Here one branch wraps a Sequential pipeline of two steps.

    async def normalize(text: str) -> str:
        return text.strip().lower()

    async def tag(text: str) -> str:
        return f"[processed] {text}"

    def route_by_type(text: str) -> str:
        return "pipeline" if len(text.split()) > 1 else "simple"

    async def handle_simple(text: str) -> str:
        return f"simple: {text}"

    emitter = make_emitter("conditional-s5")

    inner_sequential = Sequential(
        name="normalize-and-tag",
        steps=[
            FunctionStep("normalize", normalize),
            FunctionStep("tag", tag),
        ],
        emitter=emitter,
    )

    conditional = Conditional(
        name="nested-workflow",
        router=route_by_type,
        branches={
            "simple": FunctionStep("simple", handle_simple),
            "pipeline": WorkflowStep(inner_sequential),
        },
        emitter=emitter,
    )

    result = await conditional.execute("  Hello World  ")

    # Sequential ran both steps: normalize → tag
    assert result.output == "[processed] hello world"
    assert result.metadata["selected_branch"] == "pipeline"

    print("  Input: '  Hello World  '")
    print(f"  Branch: {result.metadata['selected_branch']}")
    print(f"  Output: {result.output}")
    print("✓ WorkflowStep branch executed nested Sequential pipeline")

    # --- Section 6: Workflow Events ---
    print("\n--- Section 6: Workflow Events ---")

    # Verify the event stream: one start, one step-complete, one complete.

    emitter = make_emitter("conditional-s6")

    conditional = Conditional(
        name="event-demo",
        router=route_by_length,
        branches={
            "short": FunctionStep("short", handle_short),
            "medium": FunctionStep("medium", handle_medium),
            "long": FunctionStep("long", handle_long),
        },
        emitter=emitter,
    )

    result = await conditional.execute("hello")

    # WorkflowStartEvent — emitted once with workflow metadata
    start_events = [e for e in emitter.events if isinstance(e, WorkflowStartEvent)]
    assert len(start_events) == 1
    assert start_events[0].workflow_name == "event-demo"
    assert start_events[0].workflow_type == "conditional"
    assert start_events[0].step_count == 3  # number of branches

    # WorkflowStepCompleteEvent — one event for the selected branch
    step_events = [e for e in emitter.events if isinstance(e, WorkflowStepCompleteEvent)]
    assert len(step_events) == 1
    assert step_events[0].step_name == "short"

    # WorkflowCompleteEvent — emitted once on successful completion
    complete_events = [e for e in emitter.events if isinstance(e, WorkflowCompleteEvent)]
    assert len(complete_events) == 1
    assert complete_events[0].workflow_type == "conditional"
    assert complete_events[0].total_steps_executed == 1

    print(f"  Start events: {len(start_events)} (type={start_events[0].workflow_type})")
    print(f"  Step events: {len(step_events)} (step={step_events[0].step_name})")
    print(f"  Complete events: {len(complete_events)} (total_executed={complete_events[0].total_steps_executed})")
    print("✓ Correct event stream: start → step-complete → complete")

    print("\n✅ All sections passed!")


if __name__ == "__main__":
    asyncio.run(main())
