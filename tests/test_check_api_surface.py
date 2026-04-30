"""Unit tests for `scripts/check_api_surface.py`.

Tests import the script as a module and invoke its `main(argv)` function directly
(not via subprocess) to keep them fast and deterministic. Snapshot inputs are
constructed in `tmp_path` so tests do not depend on the repository's real snapshot.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "check_api_surface.py"


def _load_script() -> object:
    """Import `scripts/check_api_surface.py` as a module (not on sys.path by default)."""
    spec = importlib.util.spec_from_file_location("check_api_surface", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_api_surface"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def script() -> object:
    return _load_script()


def _write_snapshot(path: Path, names: list[str]) -> None:
    path.write_text("\n".join(sorted(names)) + "\n", encoding="utf-8")


def test_equal_prints_no_drift_confirmation(
    script: object,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import nanitics

    snapshot = tmp_path / "surface.txt"
    _write_snapshot(snapshot, list(nanitics.__all__))

    exit_code = script.main(["--snapshot", str(snapshot)])  # type: ignore[attr-defined]

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "::warning" not in captured.out
    assert str(snapshot) in captured.out
    assert f"({len(nanitics.__all__)} names)" in captured.out


def test_added_name_emits_added_warning(
    script: object,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import nanitics

    # Snapshot missing one known name - simulates a name that was added to __all__
    # but not yet written to the committed snapshot.
    snapshot = tmp_path / "surface.txt"
    names = [n for n in nanitics.__all__ if n != "deprecated"]
    _write_snapshot(snapshot, names)

    exit_code = script.main(["--snapshot", str(snapshot)])  # type: ignore[attr-defined]

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "::warning" in captured.out
    assert "added" in captured.out
    assert "deprecated" in captured.out
    assert "nanitics/__init__.py" in captured.out


def test_removed_name_emits_removed_warning(
    script: object,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import nanitics

    # Snapshot has an extra phantom name not in __all__ - simulates a name
    # that was removed from __all__ without regenerating the snapshot.
    snapshot = tmp_path / "surface.txt"
    names = [*nanitics.__all__, "PhantomRemovedSymbol"]
    _write_snapshot(snapshot, names)

    exit_code = script.main(["--snapshot", str(snapshot)])  # type: ignore[attr-defined]

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "::warning" in captured.out
    assert "removed" in captured.out
    assert "PhantomRemovedSymbol" in captured.out


def test_added_and_removed_both_emitted_added_first(
    script: object,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """When both categories are present, added warnings precede removed warnings."""
    import nanitics

    snapshot = tmp_path / "surface.txt"
    names = [n for n in nanitics.__all__ if n != "deprecated"] + ["PhantomRemovedSymbol"]
    _write_snapshot(snapshot, names)

    exit_code = script.main(["--snapshot", str(snapshot)])  # type: ignore[attr-defined]

    captured = capsys.readouterr()
    assert exit_code == 0
    added_idx = captured.out.index("added: deprecated")
    removed_idx = captured.out.index("removed: PhantomRemovedSymbol")
    assert added_idx < removed_idx


def test_missing_snapshot_emits_regeneration_warning(
    script: object,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing = tmp_path / "does-not-exist.txt"

    exit_code = script.main(["--snapshot", str(missing)])  # type: ignore[attr-defined]

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "::warning" in captured.out
    assert str(missing) in captured.out or "snapshot" in captured.out.lower()


def test_default_snapshot_path_used_when_no_arg(
    script: object,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Invoked without --snapshot, the script reads the committed snapshot and is clean."""
    exit_code = script.main([])  # type: ignore[attr-defined]
    captured = capsys.readouterr()
    assert exit_code == 0
    # Committed snapshot matches current __all__ at the time of this test.
    assert "::warning" not in captured.out
    assert "tests/public_api_surface.txt" in captured.out
