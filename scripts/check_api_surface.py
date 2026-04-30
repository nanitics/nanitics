"""Public-API surface drift check.

Compares `sorted(nanitics.__all__)` against the committed snapshot at
`tests/public_api_surface.txt` and prints GitHub workflow-command warning
annotations when drift is detected. Non-blocking by design: exit code is
always 0 on drift so the CI job is advisory only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nanitics

_DEFAULT_SNAPSHOT = Path("tests/public_api_surface.txt")
_SURFACE_FILE_ANNOTATION = "nanitics/__init__.py"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check that nanitics.__all__ matches the committed public-API snapshot."
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=_DEFAULT_SNAPSHOT,
        help="Path to the committed snapshot file (default: tests/public_api_surface.txt).",
    )
    return parser.parse_args(argv)


def _load_snapshot(path: Path) -> list[str] | None:
    """Return the snapshot contents as a list of names, or None if the file is missing."""
    if not path.exists():
        return None
    raw = path.read_text(encoding="utf-8")
    # Splitting on "\n" and discarding the trailing empty line (from the required trailing newline).
    lines = raw.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    return lines


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    snapshot_path: Path = args.snapshot

    snapshot = _load_snapshot(snapshot_path)
    if snapshot is None:
        print(
            f"::warning file={_SURFACE_FILE_ANNOTATION}::"
            f"Public API surface snapshot not found at {snapshot_path}. "
            f"Regenerate it with: "
            f'uv run python -c "import nanitics; '
            f"print('\\n'.join(sorted(nanitics.__all__)))\" "
            f"> {snapshot_path}"
        )
        return 0

    live = sorted(nanitics.__all__)
    snapshot_sorted = sorted(snapshot)

    added = [n for n in live if n not in set(snapshot_sorted)]
    removed = [n for n in snapshot_sorted if n not in set(live)]

    if not added and not removed:
        print(f"Public API surface matches {snapshot_path} ({len(live)} names).")
        return 0

    for name in added:
        print(
            f"::warning file={_SURFACE_FILE_ANNOTATION}::"
            f"Public API surface added: {name}. "
            f"Regenerate {snapshot_path} if intentional."
        )
    for name in removed:
        print(
            f"::warning file={_SURFACE_FILE_ANNOTATION}::"
            f"Public API surface removed: {name}. "
            f"Mark with @deprecated and update docs/deprecation-policy.md if intentional."
        )
    return 0


if __name__ == "__main__":  # pragma: no cover (module-as-script entry point; exercised via main())
    sys.exit(main())
