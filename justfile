# Nanitics SDK — Task Runner

# List available recipes (default when `just` is run with no arguments)
default:
    @just --list

# Install all dependencies
setup:
    uv sync

# Auto-fix Python lint + format
fix:
    uv run ruff check --fix .
    uv run ruff format .

# Lint all Python files
lint:
    uv run ruff check .

# Type-check SDK
typecheck:
    uv run mypy nanitics

# Run all Python tests
test:
    uv run pytest -v

# Runs locally (agents after every coding batch, humans before commit) and in CI — one command, identical semantics.
# Fails fast on the first failing step. Output is mirrored to .check-output for agent consumption.
# By default, Docker-dependent tests are skipped. Use `just check docker=true` to include them.
# Single authoritative quality gate — auto-fix, format-check, lint, typecheck, tests with 100% coverage
check docker="false":
    #!/usr/bin/env bash

    outfile=".check-output"
    tmpdir=$(mktemp -d)
    trap 'rm -rf "$tmpdir"' EXIT

    # Tee helper: write to both terminal and the output file
    out() { echo "$@" | tee -a "$outfile"; }

    # Start fresh
    > "$outfile"
    rm -f .coverage

    run_step() {
        local step_num=$1 total=$2 label=$3
        shift 3
        printf "[%d/%d] %-30s" "$step_num" "$total" "$label..."
        printf "[%d/%d] %-30s" "$step_num" "$total" "$label..." >> "$outfile"
        if "$@" > "$tmpdir/out.log" 2>&1; then
            echo "OK"
            echo "OK" >> "$outfile"
        else
            echo "FAILED"
            echo "FAILED" >> "$outfile"
            out ""
            out "── $label ──"
            cat "$tmpdir/out.log" | tee -a "$outfile"
            # If coverage gaps exist, re-print them at the very end for visibility
            if grep -q "^TOTAL" "$tmpdir/out.log"; then
                local gaps
                gaps=$(grep -E "^(Name|TOTAL|nanitics/)" "$tmpdir/out.log" | grep -v "100%" | grep -v "^Name")
                if [ -n "$gaps" ]; then
                    out ""
                    out "── Coverage gaps ──"
                    grep "^Name" "$tmpdir/out.log" | head -1 | tee -a "$outfile"
                    echo "$gaps" | tee -a "$outfile"
                fi
            fi
            exit 1
        fi
    }

    # Build pytest command based on docker flag.
    # Default marker selection (excluding docker) comes from `addopts` in
    # pyproject.toml. When running with docker=true, override `-m` with a
    # tautology so docker tests are included alongside the default suite.
    cov_args="--cov=nanitics --cov=docker/full-stack --cov-report=term-missing:skip-covered --cov-fail-under=100"
    if [[ "{{docker}}" == "true" ]]; then
        test_cmd="uv run pytest -v tests -m 'docker or not docker' $cov_args"
    else
        test_cmd="uv run pytest -v tests $cov_args --cov-config=.coveragerc-no-docker"
    fi

    total=6

    run_step 1 "$total" "Auto-fix (ruff)" bash -c 'uv run ruff check --fix . && uv run ruff format .'
    run_step 2 "$total" "Format (ruff)" uv run ruff format --check .
    run_step 3 "$total" "Lint (ruff)" uv run ruff check .
    run_step 4 "$total" "Typecheck (mypy)" uv run mypy nanitics
    run_step 5 "$total" "Guide snippets" uv run python docs/_verify_guide_snippets.py
    run_step 6 "$total" "SDK tests" bash -c "$test_cmd"

    out ""
    out "All checks passed"

# Full quality gate including Docker tests
ci: (check "true")

# Includes pdoc output + llms.txt. Also used by the docs.yml workflow.
# Build the hosted API reference locally into build/docs/ (gitignored)
docs:
    #!/usr/bin/env bash
    set -euo pipefail
    rm -rf build/docs
    mkdir -p build/docs
    version=$(uv run python -c 'import tomllib; print(tomllib.loads(open("pyproject.toml").read())["project"]["version"])')
    uv run pdoc \
      -d google \
      --no-show-source \
      -e nanitics=https://github.com/nanitics/nanitics/blob/main/nanitics/ \
      --logo https://raw.githubusercontent.com/nanitics/nanitics/main/assets/nanitics-logo.png \
      --logo-link https://github.com/nanitics/nanitics \
      --footer-text "Nanitics ${version}" \
      -o build/docs/ \
      nanitics nanitics.patterns nanitics.experimental
    uv run python scripts/generate_llms_txt.py --output build/docs/llms.txt
    cp docs/CNAME build/docs/CNAME

# Quick coverage check — mirrors `check`'s non-docker coverage computation so a clean run here matches the gate
coverage:
    uv run pytest --no-header -q tests --cov=nanitics --cov=docker/full-stack --cov-report=term-missing:skip-covered --cov-fail-under=100 --cov-config=.coveragerc-no-docker

# Hard-skips every script when ANTHROPIC_API_KEY is unset. An empty suite
# (no scripts collected yet) is treated as success so the target is usable
# during framework bootstrap before scripts exist.
#
# Options follow the recipe name (both conventional and bare-kebab forms work):
#   fail-fast                       stop on the first failing script (pytest -x).
#   from=validation/<theme>/<name>.py
#                                   start the run from this script (sorted), exercising
#                                   it and every later validation/**/*.py script.
#   parallel=auto|N|off             pytest-xdist worker count. Default: auto (CPU count).
#                                   Use a fixed cap (e.g. parallel=4) if LLM rate limits
#                                   bite, or `parallel=off` to run serially.
#
# Trailing positional args pass through to pytest. Use `--` to pass pytest
# flags that start with a dash.
# Examples:
#   just validate                                               # whole suite (parallel)
#   just validate parallel=off                                  # run serially
#   just validate parallel=4                                    # cap at 4 workers
#   just validate fail-fast                                     # stop on first failure
#   just validate from=validation/memory/episodic_memory.py     # start from here onward
#   just validate fail-fast from=validation/memory/episodic_memory.py
#   just validate validation/smoke/smoke.py                     # run one script
#   just validate validation/ -- -k smoke                       # filter by keyword
#
# Run the full validation suite against real LLM services.
validate *args:
    uv run python scripts/validate.py {{args}}

# Run the quick subset (scripts tagged @pytest.mark.quick) of the validation suite. Options and positional args follow the same convention as `validate`.
validate-quick *args:
    uv run python scripts/validate.py --quick {{args}}

# Install observatory node dependencies (idempotent; fast no-op when up-to-date)
observatory-deps:
    cd observatory && npm install

# Observatory CI-equivalent gate (lint, typecheck, tests) — run before pushing observatory/ changes
observatory-check: observatory-deps
    cd observatory && npm run lint && npm run typecheck && npm run test

# Start observatory standalone dev server (port 5173)
observatory-dev: observatory-deps
    cd observatory && npm run dev

# Build observatory embed UI (served by the observatory router)
observatory-build: observatory-deps
    cd observatory && npx vite build --config vite.embed.config.ts

# Bring up the local-dev Observatory compose (builds if needed).
observatory-compose:
    cd docker/observatory-dev && docker compose up --build

# Stop and remove the local-dev Observatory compose.
observatory-compose-down:
    cd docker/observatory-dev && docker compose down

# Requires one of ANTHROPIC_API_KEY / OPENAI_API_KEY in docker/full-stack/.env
# matching NANITICS_LLM_PROVIDER. See docker/full-stack/README.md.
# Bring up the full-stack compose (Nanitics + Postgres + embedded Observatory)
full-stack-compose:
    cd docker/full-stack && docker compose up --build

# Stop and remove the full-stack compose.
full-stack-compose-down:
    cd docker/full-stack && docker compose down

# Remove caches
clean:
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

# Remove all validation trace outputs except .gitkeep and README.md
clean-traces:
    find validation/traces -mindepth 1 -not -name .gitkeep -not -name README.md -exec rm -rf {} + 2>/dev/null || true

# Remove all generated files (caches, venv, node_modules)
reset: clean
    rm -rf .venv/
    find . -type d -name "node_modules" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
