"""Deterministic unit tests for `scripts/validate.py`.

These tests exercise the wrapper's argument parsing, kebab-token rewriting,
target computation, and pytest-command construction without ever spawning
a real pytest subprocess. `--dry-run` is used where the CLI output is the
subject; `subprocess.run` is monkeypatched where exit-code remapping is the
subject.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WRAPPER_PATH = REPO_ROOT / "scripts" / "validate.py"


def _load_wrapper() -> ModuleType:
    """Load `scripts/validate.py` as a module without adding `scripts/` to sys.path."""
    spec = importlib.util.spec_from_file_location("validate_cli_under_test", WRAPPER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validate = _load_wrapper()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeCompleted:
    """Drop-in for `subprocess.CompletedProcess`."""

    def __init__(self, returncode: int) -> None:
        self.returncode = returncode


def _fake_run_factory(returncode: int, captured: list[list[str]]) -> Any:
    def _fake_run(cmd: list[str], check: bool = False) -> _FakeCompleted:
        captured.append(list(cmd))
        return _FakeCompleted(returncode)

    return _fake_run


@pytest.fixture
def fake_validation_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point the wrapper's glob at a temporary themed `validation/` tree."""
    validation = tmp_path / "validation"
    validation.mkdir()
    # Mirror the themed layout: one file per subdirectory.
    tree = {
        "smoke": "smoke.py",
        "tools": "tool_execution.py",
        "memory": "episodic_memory.py",
        "durability": "durable_hitl.py",
    }
    paths: list[str] = []
    for theme, name in tree.items():
        theme_dir = validation / theme
        theme_dir.mkdir()
        (theme_dir / name).write_text("# placeholder\n")
        paths.append(f"validation/{theme}/{name}")
    # An infrastructure file that must be excluded.
    helpers = validation / "helpers"
    helpers.mkdir()
    (helpers / "assertions.py").write_text("# infra\n")
    # A conftest.py that must also be excluded.
    (validation / "conftest.py").write_text("# conftest\n")

    sorted_paths = sorted(paths)

    def _fake_sorted_scripts() -> list[str]:
        return sorted_paths

    monkeypatch.setattr(validate, "_sorted_validation_scripts", _fake_sorted_scripts)
    return validation


# ---------------------------------------------------------------------------
# _rewrite_bare_kebab
# ---------------------------------------------------------------------------


def test_rewrite_bare_kebab_rewrites_known_names() -> None:
    assert validate._rewrite_bare_kebab(["fail-fast"]) == ["--fail-fast"]
    assert validate._rewrite_bare_kebab(["fail-fast=true"]) == ["--fail-fast=true"]
    assert validate._rewrite_bare_kebab(["from=validation/smoke/smoke.py"]) == ["--from=validation/smoke/smoke.py"]
    assert validate._rewrite_bare_kebab(["quick"]) == ["--quick"]


def test_rewrite_bare_kebab_passes_unknown_tokens_through() -> None:
    # A path-like positional is not a known flag — must not be rewritten.
    assert validate._rewrite_bare_kebab(["validation/smoke/smoke.py"]) == ["validation/smoke/smoke.py"]
    # A --prefixed form should also pass through.
    assert validate._rewrite_bare_kebab(["--fail-fast"]) == ["--fail-fast"]


def test_rewrite_bare_kebab_respects_separator() -> None:
    assert validate._rewrite_bare_kebab(["--", "fail-fast", "from=x"]) == [
        "--",
        "fail-fast",
        "from=x",
    ]


# ---------------------------------------------------------------------------
# _parse_truthy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["true", "TRUE", "1", "yes", "on"])
def test_parse_truthy_accepts_truthy(value: str) -> None:
    assert validate._parse_truthy(value) is True


@pytest.mark.parametrize("value", ["false", "FALSE", "0", "no", "off"])
def test_parse_truthy_accepts_falsy(value: str) -> None:
    assert validate._parse_truthy(value) is False


def test_parse_truthy_rejects_invalid() -> None:
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        validate._parse_truthy("maybe")


# ---------------------------------------------------------------------------
# _parse_parallel
# ---------------------------------------------------------------------------


def test_parse_parallel_auto() -> None:
    assert validate._parse_parallel("auto") == "auto"
    assert validate._parse_parallel("AUTO") == "auto"


@pytest.mark.parametrize("value", ["off", "OFF", "0", "1"])
def test_parse_parallel_serial_returns_none(value: str) -> None:
    assert validate._parse_parallel(value) is None


@pytest.mark.parametrize(("value", "expected"), [("2", "2"), ("4", "4"), ("16", "16")])
def test_parse_parallel_positive_integer(value: str, expected: str) -> None:
    assert validate._parse_parallel(value) == expected


def test_parse_parallel_rejects_negative() -> None:
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        validate._parse_parallel("-2")


def test_parse_parallel_rejects_non_numeric() -> None:
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        validate._parse_parallel("many")


# ---------------------------------------------------------------------------
# main() — dry-run behavior
# ---------------------------------------------------------------------------


def test_main_default_targets_validation_dir(capsys: pytest.CaptureFixture[str], fake_validation_dir: Path) -> None:
    exit_code = validate.main(["--dry-run"])
    out = capsys.readouterr().out.strip()
    assert exit_code == 0
    assert out.endswith("validation/")
    assert "-x" not in out.split()
    assert "-m" not in out.split()


def test_main_fail_fast_flag_appends_dash_x(capsys: pytest.CaptureFixture[str], fake_validation_dir: Path) -> None:
    exit_code = validate.main(["--fail-fast", "--dry-run"])
    out = capsys.readouterr().out.strip().split()
    assert exit_code == 0
    assert "-x" in out


def test_main_fail_fast_bare_kebab_enables(capsys: pytest.CaptureFixture[str], fake_validation_dir: Path) -> None:
    exit_code = validate.main(["fail-fast", "--dry-run"])
    out = capsys.readouterr().out.strip().split()
    assert exit_code == 0
    assert "-x" in out


def test_main_fail_fast_false_disables(capsys: pytest.CaptureFixture[str], fake_validation_dir: Path) -> None:
    exit_code = validate.main(["fail-fast=false", "--dry-run"])
    out = capsys.readouterr().out.strip().split()
    assert exit_code == 0
    assert "-x" not in out


def test_main_from_path_matches(capsys: pytest.CaptureFixture[str], fake_validation_dir: Path) -> None:
    exit_code = validate.main(["--from=validation/memory/episodic_memory.py", "--dry-run"])
    out = capsys.readouterr().out.strip().split()
    assert exit_code == 0
    assert "validation/memory/episodic_memory.py" in out
    assert "validation/smoke/smoke.py" in out
    assert "validation/tools/tool_execution.py" in out
    # Sorted-order scripts before memory/ must be excluded.
    assert "validation/durability/durable_hitl.py" not in out


def test_main_from_path_missing_returns_2(capsys: pytest.CaptureFixture[str], fake_validation_dir: Path) -> None:
    exit_code = validate.main(["--from=validation/does_not_exist.py"])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "no scripts at or after" in captured.err
    assert "validation/does_not_exist.py" in captured.err


def test_main_from_bare_kebab_form(capsys: pytest.CaptureFixture[str], fake_validation_dir: Path) -> None:
    exit_code = validate.main(["from=validation/memory/episodic_memory.py", "--dry-run"])
    out = capsys.readouterr().out.strip().split()
    assert exit_code == 0
    assert "validation/memory/episodic_memory.py" in out


def test_main_default_includes_parallel_auto(capsys: pytest.CaptureFixture[str], fake_validation_dir: Path) -> None:
    exit_code = validate.main(["--dry-run"])
    out = capsys.readouterr().out.strip().split()
    assert exit_code == 0
    # `-n auto` must be present and adjacent.
    assert "-n" in out
    assert out[out.index("-n") + 1] == "auto"


def test_main_parallel_off_omits_dash_n(capsys: pytest.CaptureFixture[str], fake_validation_dir: Path) -> None:
    exit_code = validate.main(["parallel=off", "--dry-run"])
    out = capsys.readouterr().out.strip().split()
    assert exit_code == 0
    assert "-n" not in out


def test_main_parallel_integer(capsys: pytest.CaptureFixture[str], fake_validation_dir: Path) -> None:
    exit_code = validate.main(["parallel=4", "--dry-run"])
    out = capsys.readouterr().out.strip().split()
    assert exit_code == 0
    assert "-n" in out
    assert out[out.index("-n") + 1] == "4"


def test_main_parallel_bare_kebab_enables_auto(capsys: pytest.CaptureFixture[str], fake_validation_dir: Path) -> None:
    # Bare `parallel` (no value) → `--parallel` with the const default `auto`.
    exit_code = validate.main(["parallel", "--dry-run"])
    out = capsys.readouterr().out.strip().split()
    assert exit_code == 0
    assert "-n" in out
    assert out[out.index("-n") + 1] == "auto"


def test_main_quick_flag(capsys: pytest.CaptureFixture[str], fake_validation_dir: Path) -> None:
    exit_code = validate.main(["--quick", "--dry-run"])
    out = capsys.readouterr().out.strip().split()
    assert exit_code == 0
    assert "-m" in out
    assert "quick" in out


def test_main_quick_with_from(capsys: pytest.CaptureFixture[str], fake_validation_dir: Path) -> None:
    exit_code = validate.main(["--quick", "--from=validation/smoke/smoke.py", "--dry-run"])
    out = capsys.readouterr().out.strip().split()
    assert exit_code == 0
    assert "-m" in out
    assert "quick" in out
    assert "validation/smoke/smoke.py" in out


def test_main_positional_pytest_args_used_as_targets(
    capsys: pytest.CaptureFixture[str], fake_validation_dir: Path
) -> None:
    # Flags precede positionals (standard CLI convention); argparse REMAINDER
    # ends flag parsing at the first non-flag token, so writing `--dry-run`
    # after the positional would forward it to pytest.
    exit_code = validate.main(["--dry-run", "validation/smoke/smoke.py"])
    out = capsys.readouterr().out.strip()
    assert exit_code == 0
    tokens = out.split()
    assert "validation/smoke/smoke.py" in tokens
    # The default `validation/` dir must NOT be prepended when explicit
    # positional targets are given.
    assert tokens[-1] == "validation/smoke/smoke.py"


def test_main_leading_separator_stripped(capsys: pytest.CaptureFixture[str], fake_validation_dir: Path) -> None:
    # `just validate -- -k smoke` → wrapper drops the leading `--`.
    # Note that because argparse REMAINDER captures from the first non-flag,
    # positional order here is: `--dry-run -- -k smoke` would put --dry-run
    # into the passthrough, which is not what we want. So place --dry-run
    # before the passthrough sentinel:
    exit_code = validate.main(["--dry-run", "--", "-k", "smoke"])
    out = capsys.readouterr().out.strip()
    assert exit_code == 0
    tokens = out.split()
    # `--` must have been stripped.
    assert "--" not in tokens
    # `-k smoke` becomes the target list because `_compute_targets`
    # treats pytest_args as targets when --from is unset and the list is
    # non-empty. That's the same behavior as the legacy recipe which
    # passed `{{args}}` directly.
    assert "-k" in tokens
    assert "smoke" in tokens


def test_main_from_wins_over_positional_args(capsys: pytest.CaptureFixture[str], fake_validation_dir: Path) -> None:
    exit_code = validate.main(
        [
            "--from=validation/memory/episodic_memory.py",
            "--dry-run",
            "--",
            "-k",
            "episodic",
        ]
    )
    out = capsys.readouterr().out.strip().split()
    assert exit_code == 0
    # Both the computed target list AND the extra pytest args appear.
    assert "validation/memory/episodic_memory.py" in out
    assert "-k" in out
    assert "episodic" in out


def test_main_rejects_unknown_flag(capsys: pytest.CaptureFixture[str], fake_validation_dir: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        validate.main(["--totally-unknown"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "unrecognized arguments" in err or "unrecognized" in err


# ---------------------------------------------------------------------------
# main() — exit-code remapping (subprocess mocked)
# ---------------------------------------------------------------------------


def test_main_pytest_exit_5_is_remapped_to_0(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    fake_validation_dir: Path,
) -> None:
    captured: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", _fake_run_factory(5, captured))

    exit_code = validate.main([])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "no scripts collected" in out
    assert captured and captured[0][:4] == ["uv", "run", "pytest", "-v"]


def test_main_pytest_nonzero_exit_without_fail_fast_no_banner(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    fake_validation_dir: Path,
) -> None:
    monkeypatch.setattr(subprocess, "run", _fake_run_factory(1, []))

    exit_code = validate.main([])
    err = capsys.readouterr().err
    assert exit_code == 1
    assert "fail-fast" not in err


def test_main_pytest_nonzero_exit_with_fail_fast_prints_banner(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    fake_validation_dir: Path,
) -> None:
    monkeypatch.setattr(subprocess, "run", _fake_run_factory(3, []))

    exit_code = validate.main(["--fail-fast"])
    err = capsys.readouterr().err
    assert exit_code == 3
    assert "fail-fast: stopped on first failure" in err


def test_main_pytest_success(
    monkeypatch: pytest.MonkeyPatch,
    fake_validation_dir: Path,
) -> None:
    captured: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", _fake_run_factory(0, captured))

    exit_code = validate.main(["--quick"])
    assert exit_code == 0
    assert captured[0][:4] == ["uv", "run", "pytest", "-v"]
    assert "-m" in captured[0]
    assert "quick" in captured[0]


# ---------------------------------------------------------------------------
# `python -m` style entry point
# ---------------------------------------------------------------------------


def test_entry_point_invokes_main(monkeypatch: pytest.MonkeyPatch, fake_validation_dir: Path) -> None:
    """Exercise the `if __name__ == '__main__'` block by re-executing the file
    under a fresh namespace with a mocked subprocess.run."""
    captured: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", _fake_run_factory(0, captured))
    monkeypatch.setattr(sys, "argv", ["validate", "--dry-run"])

    source = WRAPPER_PATH.read_text()
    namespace: dict[str, Any] = {"__name__": "__main__", "__file__": str(WRAPPER_PATH)}
    # Share the monkeypatched subprocess so the executed module sees it.
    namespace["subprocess"] = subprocess
    with pytest.raises(SystemExit) as exc:
        exec(compile(source, str(WRAPPER_PATH), "exec"), namespace)
    assert exc.value.code == 0


# ---------------------------------------------------------------------------
# _build_pytest_command
# ---------------------------------------------------------------------------


def test_build_pytest_command_order() -> None:
    cmd = validate._build_pytest_command(
        fail_fast=True,
        quick=True,
        parallel="auto",
        targets=["validation/smoke/smoke.py"],
        extra_args=["-k", "smoke"],
    )
    # Shape: uv run pytest -v -m quick -x -n auto -k smoke validation/smoke/smoke.py
    assert cmd[:4] == ["uv", "run", "pytest", "-v"]
    assert cmd[4:6] == ["-m", "quick"]
    assert cmd[6] == "-x"
    assert cmd[7:9] == ["-n", "auto"]
    assert cmd[9:11] == ["-k", "smoke"]
    assert cmd[-1] == "validation/smoke/smoke.py"


def test_build_pytest_command_minimal() -> None:
    cmd = validate._build_pytest_command(
        fail_fast=False,
        quick=False,
        parallel=None,
        targets=["validation/"],
        extra_args=[],
    )
    assert cmd == ["uv", "run", "pytest", "-v", "validation/"]


def test_build_pytest_command_parallel_integer() -> None:
    cmd = validate._build_pytest_command(
        fail_fast=False,
        quick=False,
        parallel="4",
        targets=["validation/"],
        extra_args=[],
    )
    assert cmd == ["uv", "run", "pytest", "-v", "-n", "4", "validation/"]


# Unused reference to keep pytest from complaining about unused imports when
# SimpleNamespace is used in a future expansion.
_ = SimpleNamespace
