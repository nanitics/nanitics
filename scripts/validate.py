#!/usr/bin/env python3
"""Validate CLI wrapper — builds and runs the `pytest` command for the
`validation/` suite.

This script is the single entry point for the `just validate` and
`just validate-quick` recipes. It owns argument parsing, script selection,
and pytest-command construction so the `justfile` can stay a thin delegator
and developers can write options *after* the recipe name:

    just validate fail-fast
    just validate from=validation/memory/episodic_memory.py
    just validate fail-fast from=validation/memory/episodic_memory.py
    just validate validation/smoke/smoke.py
    just validate validation/ -- -k smoke

Both conventional (`--fail-fast`, `--from=PATH`) and bare-kebab
(`fail-fast`, `from=PATH`) forms are accepted.

The wrapper uses stdlib only and is outside the `nanitics/` package so it
stays out of the published SDK and the `--cov=nanitics` gate.
"""

from __future__ import annotations

import argparse
import glob
import subprocess
import sys
from collections.abc import Sequence

# Known bare-kebab flag names that the pre-parse rewrites into `--` form.
_KEBAB_FLAG_NAMES: frozenset[str] = frozenset({"fail-fast", "from", "quick", "parallel"})


def _rewrite_bare_kebab(argv: Sequence[str]) -> list[str]:
    """Rewrite bare-kebab flag tokens into their `--`-prefixed form.

    `just validate fail-fast from=validation/memory/episodic_memory.py` should
    reach argparse as `--fail-fast --from=validation/memory/episodic_memory.py`.
    Tokens after a literal `--` are never rewritten (pytest passthrough).
    Unknown bare-kebab tokens pass through untouched — `just validate
    validation/smoke/smoke.py` is not a flag attempt.
    """
    out: list[str] = []
    after_sep = False
    for tok in argv:
        if after_sep:
            out.append(tok)
            continue
        if tok == "--":
            after_sep = True
            out.append(tok)
            continue
        # Split on the first `=` for `key=value` matching.
        name = tok.partition("=")[0]
        if not tok.startswith("-") and name in _KEBAB_FLAG_NAMES:
            # `fail-fast` alone → `--fail-fast`.
            # `fail-fast=true` → `--fail-fast=true` (argparse's BooleanOptionalAction-style
            # is avoided; we handle the truthy/falsy parse ourselves).
            out.append(f"--{tok}")
        else:
            out.append(tok)
    return out


def _parse_truthy(value: str) -> bool:
    """Parse `fail-fast=true|false` (any case) into a bool."""
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value!r} (expected 'true' or 'false')")


def _parse_parallel(value: str) -> str | None:
    """Parse `--parallel` value into a pytest-xdist `-n` argument or None.

    Returns None for serial execution (no `-n` flag added). Returns the
    string to pass after `-n` otherwise — `"auto"` or a positive integer
    rendered as a string.
    """
    normalized = value.strip().lower()
    if normalized in {"off", "0", "1"}:
        return None
    if normalized == "auto":
        return "auto"
    try:
        n = int(normalized)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"invalid --parallel value: {value!r} (expected 'auto', 'off', or positive integer)"
        ) from None
    if n < 1:
        raise argparse.ArgumentTypeError(f"invalid --parallel value: {value!r} (must be >= 1)")
    return str(n)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="validate",
        description=(
            "Run the nanitics validation suite against real LLM services. "
            "Options may be written as conventional flags (--fail-fast, "
            "--from=PATH) or bare-kebab tokens (fail-fast, from=PATH)."
        ),
        epilog=(
            "Positional args after all recognized flags are forwarded to "
            "pytest verbatim. Use `--` to pass flags that start with a dash."
        ),
    )
    # `--fail-fast`: accepts bare form and `--fail-fast=true|false`.
    parser.add_argument(
        "--fail-fast",
        "-x",
        nargs="?",
        const="true",
        default="false",
        metavar="true|false",
        help=(
            "Stop on the first failing script (appends `-x` to pytest). "
            "Bare `--fail-fast` enables; `--fail-fast=false` explicitly "
            "disables."
        ),
    )
    parser.add_argument(
        "--from",
        dest="from_path",
        metavar="PATH",
        default=None,
        help=(
            "Start the run from this validation script (sorted order) and "
            "run it plus every later `validation/**/*.py` script."
        ),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=("Add `-m quick` to the pytest invocation. Set automatically by the `validate-quick` recipe."),
    )
    parser.add_argument(
        "--parallel",
        nargs="?",
        const="auto",
        default="auto",
        metavar="N|auto|off",
        help=(
            "Run scripts in parallel via pytest-xdist. `auto` uses CPU count, "
            "`off` (or `1`) runs serially, integer N uses N workers. "
            "Default: auto. Cap with e.g. `parallel=4` if API rate limits bite."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help=("Arguments forwarded to pytest. A leading `--` separator is supported and stripped."),
    )
    return parser


def _sorted_validation_scripts() -> list[str]:
    """Return sorted `validation/**/*.py` paths (themed subdirectories).

    Excludes infrastructure that isn't a runnable script: `conftest.py`,
    `__init__.py`, and anything under `validation/helpers/` or
    `validation/traces/`.
    """
    scripts = []
    for path in glob.glob("validation/**/*.py", recursive=True):
        name = path.rsplit("/", 1)[-1]
        if name in {"conftest.py", "__init__.py"}:
            continue
        if path.startswith(("validation/helpers/", "validation/traces/")):
            continue
        scripts.append(path)
    return sorted(scripts)


def _strip_leading_separator(args: list[str]) -> list[str]:
    """Strip a single leading `--` passthrough separator if present."""
    if args and args[0] == "--":
        return args[1:]
    return args


def _compute_targets(
    from_path: str | None,
    pytest_args: list[str],
) -> tuple[list[str], list[str]] | None:
    """Compute (targets, extra_pytest_args) for the pytest command.

    Returns None if `from_path` is set but matches no scripts — caller
    handles the error + exit-2 path.
    """
    if from_path is not None:
        all_scripts = _sorted_validation_scripts()
        if from_path not in all_scripts:
            return None
        idx = all_scripts.index(from_path)
        targets = all_scripts[idx:]
        # When --from wins, positional args are extra pytest args (e.g. `-k`).
        return targets, pytest_args
    if not pytest_args:
        return ["validation/"], []
    return pytest_args, []


def _build_pytest_command(
    *,
    fail_fast: bool,
    quick: bool,
    parallel: str | None,
    targets: list[str],
    extra_args: list[str],
) -> list[str]:
    cmd = ["uv", "run", "pytest", "-v"]
    if quick:
        cmd += ["-m", "quick"]
    if fail_fast:
        cmd.append("-x")
    if parallel is not None:
        cmd += ["-n", parallel]
    cmd += extra_args
    cmd += targets
    return cmd


def _emit_fail_fast_banner() -> None:
    print("", file=sys.stderr)
    print(
        "── Validation failed (fail-fast: stopped on first failure) ──",
        file=sys.stderr,
    )
    print("See the pytest output above for traceback details.", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    rewritten = _rewrite_bare_kebab(raw)
    parser = _build_parser()
    ns = parser.parse_args(rewritten)

    fail_fast = _parse_truthy(ns.fail_fast)
    parallel = _parse_parallel(ns.parallel)
    pytest_args = _strip_leading_separator(list(ns.pytest_args))

    resolved = _compute_targets(ns.from_path, pytest_args)
    if resolved is None:
        print(
            f"validate: no scripts at or after '{ns.from_path}'.",
            file=sys.stderr,
        )
        return 2
    targets, extra_args = resolved

    cmd = _build_pytest_command(
        fail_fast=fail_fast,
        quick=ns.quick,
        parallel=parallel,
        targets=targets,
        extra_args=extra_args,
    )

    if ns.dry_run:
        print(" ".join(cmd))
        return 0

    completed = subprocess.run(cmd, check=False)
    status = completed.returncode

    # pytest exit 5 = no tests collected. The legacy recipes treat this
    # as success so the target is usable during bootstrap.
    if status == 5:
        print("validate: no scripts collected (suite is empty).")
        return 0

    if status != 0 and fail_fast:
        _emit_fail_fast_banner()

    return status


if __name__ == "__main__":
    raise SystemExit(main())
