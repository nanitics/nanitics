"""Run all SDK examples as tests to prevent bitrot."""

import importlib.util
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"

# Files at the examples root that aren't runnable examples.
_NON_EXAMPLE_NAMES = {"helpers.py", "__init__.py"}


def discover_examples() -> list[str]:
    """Find all example modules recursively (excluding helpers/infrastructure)."""
    return sorted(
        path.relative_to(EXAMPLES_DIR).with_suffix("").as_posix()
        for path in EXAMPLES_DIR.rglob("*.py")
        if path.name not in _NON_EXAMPLE_NAMES and "__pycache__" not in path.parts
    )


@pytest.mark.parametrize("example_name", discover_examples())
async def test_example(example_name):
    """Run an example's main() function."""
    path = EXAMPLES_DIR / f"{example_name}.py"
    # Keep EXAMPLES_DIR on sys.path so `import helpers` still resolves from
    # any moved example (helpers.py remains at the examples root).
    sys.path.insert(0, str(EXAMPLES_DIR))
    try:
        spec = importlib.util.spec_from_file_location(f"nanitics_example_{example_name.replace('/', '_')}", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        await module.main()
    finally:
        sys.path.pop(0)
