# Development

Developer setup and workflow guide for contributing to Nanitics.

## Prerequisites

- **Python 3.11, 3.12, or 3.13.** The repo pins `3.11` via `.python-version`. Required CI runs on 3.11 and 3.13; 3.12 runs advisory-only. If you use `pyenv` / `asdf`, they'll pick up the pin automatically.
- **[uv](https://docs.astral.sh/uv/)** — fast Python package + virtualenv manager (replaces `pip` / `venv`). Install via `curl -LsSf https://astral.sh/uv/install.sh | sh`.
- **[just](https://github.com/casey/just)** — task runner (think Make, friendlier syntax). Install via `brew install just` on macOS, `cargo install just` on Linux, or [see other options](https://github.com/casey/just#installation).
- **Node.js 18+** and npm (optional, for Observatory UI).
- **[Docker](https://www.docker.com/)** (optional, for database features and code execution sandbox).

## Setup

All tests pass without any API keys using `MockLLMClient`; the `.env` step below is only for running tests against real LLM providers.

```bash
git clone https://github.com/nanitics/nanitics.git
cd nanitics
just setup                   # Install Python dependencies via uv sync
cp .env.example .env         # Optional: fill in API keys for real-LLM tests
```

## Commands

| Command | Description |
|---------|-------------|
| `just setup` | Install all dependencies via `uv sync` |
| `just check` | Full quality gate: auto-fix, format, lint, typecheck, tests with 100% coverage |
| `just ci` | Full quality gate including Docker-dependent tests |
| `just test` | Run all Python tests |
| `just fix` | Auto-fix Python lint + format with ruff |
| `just lint` | Lint all Python files |
| `just typecheck` | Type-check the SDK with mypy (strict mode) |
| `just coverage` | Run the suite with coverage, reporting only files below 100% |
| `just observatory-dev` | Start Observatory standalone dev server (port 5173) |
| `just observatory-build` | Build Observatory embed UI |
| `just clean` | Remove caches (pytest, mypy, pycache) |
| `just reset` | Full environment reset (caches + venv + node_modules) |

## Quality Gate

`just check` is the single authoritative quality gate. It runs:

- ruff auto-fix, format-check, and lint
- mypy strict typecheck
- the full test suite with 100% line coverage enforced (`--cov-fail-under=100`)

CI runs the same command — local and CI are identical. Run it after every coding batch and before committing.

There are no pre-commit hooks; `just check` is the gate.

By default, `just check` skips Docker-dependent tests. Use `just check docker=true` or `just ci` to include them. Output is written to `.check-output` for easy review if the terminal truncates.

### Inner-loop workflow

`just check` is the authoritative gate, but during iterative work you rarely need the full suite on every save. Re-run only what you need:

- `uv run pytest tests/path/to/test_file.py -x` — single file, stop on the first failure.
- `uv run pytest tests/path/to/test_file.py::TestClass::test_name` — single test.
- `uv run pytest --lf` — re-run only tests that failed last time.
- `uv run pytest --ff` — run previously-failing tests first, then the rest.

These are iteration aids, not substitutes. Always run `just check` before committing.

## Project Structure

```
nanitics/          — SDK Python package
  capabilities/    — Context management, error handling, evaluation, memory, planning
  collaboration/   — Human-in-the-loop (approval, revision, durable HITL)
  composition/     — Multi-agent coordination and orchestration patterns
  core/            — Agent types, tools, system prompt builder
  infrastructure/  — LLM clients, events, tracing, persistence
  safety/          — Iteration limits, cancellation, sandboxing
observatory/       — Observatory trace viewer (React/TypeScript)
tests/             — Unit and integration tests (mirrors SDK structure)
examples/          — Runnable examples (all use MockLLMClient; see examples/README.md)
docs/              — Guides, architecture, glossary
```

## Testing

See the command table above for `just test` / `just coverage` and [Inner-loop workflow](#inner-loop-workflow) for pytest shortcuts. Coverage policy — including when `# pragma: no cover` is acceptable — lives in [CONTRIBUTING.md § Coverage exclusions](CONTRIBUTING.md#coverage-exclusions-pragma-no-cover).

### Docker-dependent tests

Tests marked `@pytest.mark.docker` require a running Docker daemon (database tests, code execution sandbox) and are skipped by default. Run them with `just ci` or `just check docker=true`. When skipped, coverage uses `.coveragerc-no-docker` to exclude Docker-dependent code paths.

## Validation suite

The validation suite under `validation/` runs SDK components against **real** LLM providers, embedding providers, and (optionally) Postgres and Docker. It complements — it does not replace — the mock-based unit test suite under `tests/`. Use it to catch problems that mocks cannot surface: system prompts that confuse real LLMs, tool descriptions that produce wrong tool selection, agent loops that take too many iterations, prompts that waste tokens.

The suite is maintainer-facing tooling. It is not shipped with the wheel, it is not part of `just check`, and it is not on CI's critical path. A PR that adds a validation script is welcomed; a PR is not blocked on validation output unless a maintainer explicitly flags it.

### Commands

```bash
just validate                                            # Full suite
just validate-quick                                      # Scripts tagged @pytest.mark.quick (smoke-weight)
just validate fail-fast                                  # Stop on the first failing script
just validate from=validation/memory/episodic_memory.py  # Start the run from this script onward (sorted)
just validate fail-fast from=validation/memory/episodic_memory.py
just validate validation/smoke/smoke.py                  # Run a single script
just validate validation/ -- -k smoke                    # Forward flags to pytest after `--`
```

Both conventional (`--fail-fast`, `--from=PATH`) and bare-kebab (`fail-fast`, `from=PATH`) option forms are accepted, and options always follow the recipe name.

### Credentials

- **Required:** `ANTHROPIC_API_KEY` — if unset, the whole suite hard-skips with a single summary line and exits 0. This is deliberate: validation is not CI-critical, so missing keys should not fail loudly.
- **Optional per-script:** `OPENAI_API_KEY`, `MISTRAL_API_KEY`, `VOYAGE_API_KEY`, `POSTGRES_URL`. Scripts that need these skip individually when their credential is missing.

Credentials are read from the environment or from a `.env` file at the repo root (loaded automatically by `validation/conftest.py`).

`POSTGRES_URL` is auto-provisioned when unset: `validation/conftest.py` starts a `pgvector/pgvector:pg16` testcontainer on session start and tears it down on session finish. Requires a reachable Docker daemon and the `testcontainers` dev dep (included in `uv sync`). Set `POSTGRES_URL` explicitly to point validation at an existing database and skip auto-provisioning.

### Trace output

Every run exports a JSON trace per script to `validation/traces/<utc-timestamp>-<script>/`. Traces are gitignored. The format — envelope + derived `summary` block + full event list — is documented on `validation/helpers/trace.py:export_trace`.

### Writing a new script

See `validation/README.md` for the authoring guide and the `validation/smoke/smoke.py` reference implementation. In short: use `make_llm_client(...)` for a real client, collect events with `InMemoryEmitter`, assert with `assert_trace_contains` and `assert_result_satisfies`, and export via `save_trace`.

## Observatory

Observatory is a React/TypeScript trace viewer for inspecting agent execution. `just observatory-dev` starts the standalone dev server on port 5173; `just observatory-build` builds the embed UI served by the Observatory API router.
