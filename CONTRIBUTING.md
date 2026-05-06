# Contributing to Nanitics

Thank you for your interest in contributing to Nanitics. This guide will help you get started.

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting Started

See [DEVELOPMENT.md](DEVELOPMENT.md) for prerequisites, setup instructions, and available commands.

## Making Changes

1. Fork the repository and create a branch from `main`
2. Make your changes
3. Run `just check` — the single authoritative quality gate (auto-fix, format-check, lint, typecheck, tests with 100% coverage)
4. Commit with `git commit --signoff` (or `-s`) — every commit must carry a `Signed-off-by` line. See [Developer Certificate of Origin](#developer-certificate-of-origin) below.
5. Open a pull request

## Developer Certificate of Origin

Nanitics requires a **Developer Certificate of Origin (DCO)** sign-off on every commit. There is no separate Contributor License Agreement to sign — the DCO is a one-line per-commit attestation that you have the right to submit your contribution under the project's licence (Apache-2.0).

### How to sign off

Add the `--signoff` flag (`-s` for short) to your `git commit`:

```sh
git commit -s -m "feat: add new tool"
```

This appends a line to the commit message:

```
Signed-off-by: Your Name <your.email@example.com>
```

The name and email must match your git `user.name` and `user.email` config. If you forgot to sign off a commit, amend it with `git commit --amend --signoff` and force-push your fork branch (force-push to `main` is blocked by branch protection).

### What you're certifying

By signing off, you confirm the [DCO v1.1](https://developercertificate.org/) — most importantly: you have the right to submit the contribution, and you intend it to be distributed under Apache-2.0 (the project's licence).

### Enforcement

The `DCO` status check on every pull request verifies that all commits carry a valid `Signed-off-by` line. PRs with unsigned commits cannot be merged.

## Governance

For how decisions land, who decides, and the release cadence, see
[GOVERNANCE.md](GOVERNANCE.md). Significant changes — new major features,
breaking-change proposals, roadmap moves — start in a Discussion or issue so
the maintainers can weigh in before implementation begins. The mechanics of
landing any individual PR are in the rest of this document; `GOVERNANCE.md`
is the project-level policy that sits above it.

## API stability

Nanitics is pre-1.0. The public API is the set of names in
`nanitics.__all__`; everything else is internal. Breaking changes follow the
[deprecation policy](docs/deprecation-policy.md) — read it before renaming,
removing, or relocating a public symbol.

## Quality gate

`just check` is the single authoritative quality gate. It runs ruff (lint + format), mypy strict, and the full test suite with 100% line coverage enforced. Local and CI run the same command with identical semantics — if it passes locally and fails in CI, that is a bug in the CI setup.

## Pull Request Process

When opening a pull request:

- Write a clear description of what the change does and why
- Include a test plan describing how the change was verified
- Keep PRs focused — one logical change per PR

Reviewers will look for: correctness, test coverage, type safety, consistency with existing patterns, and clear naming.

## Code Style

Ruff handles both formatting and linting — there is no manual style guide to follow. Run `just fix` to auto-fix formatting and lint issues.

Key conventions enforced by tooling:

- Type annotations on all public functions and methods (mypy strict)
- Line length: 120 characters
- Import sorting: handled by ruff

## Adding Examples

Examples live in the `examples/` directory, grouped by theme (e.g., `tools/tool_basics.py`, `evaluation/evaluation.py`). See `examples/README.md` for the theme list and reading order.

Requirements for new examples:

- Use `MockLLMClient` for deterministic, API-key-free execution
- Include a docstring explaining what the example demonstrates
- Must be picked up by `test_examples.py` (runs all examples as tests)
- Follow the existing numbering scheme — pick a number that groups with related examples
- Add a row to `examples/README.md` at the correct numerical slot, with Description (primary public symbol(s) plus the concrete behavior shown, one line) and Guide (link to the matching `docs/guides/*.md`) columns populated

## Adding Guides

Guides live in `docs/guides/` and are linked from the [guides README](docs/guides/README.md).

When adding a new guide:

- Add it to the guides README index
- Link to it from the main README if it covers a major feature area
- Reference relevant examples

The source docstrings under [`nanitics/`](nanitics/) are the authoritative public surface — guides should point readers at the docstrings (in their editor, in the source tree, or on the [hosted reference](https://docs.nanitics.dev/)) rather than re-documenting signatures inline. `nanitics.__all__` names the public symbols.

Docstring edits flow downstream to the hosted reference via the release workflow — treat the hosted site as a projection of the source, not a separate doc surface.

**Snippet validation.** Every fenced ```` ```python ```` block in every guide is compile-checked by `docs/_verify_guide_snippets.py` as part of `just check` — the gate catches import and API drift before it reaches `main`. If a snippet is genuinely not meant to parse in isolation (illustrative stub with `...` placeholders, top-level `await` outside an async context, caller-supplied names that cannot resolve locally), annotate it with `<!-- verify: skip — <reason> -->` on the line immediately before the opening fence, where `<reason>` is a real human-readable explanation. A bare `<!-- verify: skip -->` without a reason — or a reason of only punctuation — fails the gate.

## Adding Tests

Tests live in `tests/` and mirror the SDK package structure.

Requirements:

- Use `MockLLMClient` and `MockEmbeddingClient` for deterministic tests
- No real API calls in tests
- Test behavior, not implementation details

Coverage is enforced at 100% — see [Coverage exclusions](#coverage-exclusions-pragma-no-cover) for when `# pragma: no cover` is acceptable.

For real-service validation (separate from unit tests), see `DEVELOPMENT.md` § Validation suite.

## Coverage exclusions (`# pragma: no cover`)

100% line coverage is enforced. Every `# pragma: no cover` requires an inline comment explaining *why* the line is genuinely untestable. Every new `# pragma: no cover` is reviewed in PR by a maintainer.

**Acceptable examples:**

- **Background task runners** whose loop body pytest cannot reach — the loop is driven by an external scheduler the test harness does not invoke.

  ```python
  while not shutdown.is_set():  # pragma: no cover (driven by external scheduler; unreachable from pytest)
      await run_one_tick()
  ```

- **`typing.TYPE_CHECKING`-guarded imports** — imports that exist only to inform type checkers and are not executed at runtime.

  ```python
  if TYPE_CHECKING:  # pragma: no cover (type-check-only import; no runtime path)
      from nanitics.client import Client
  ```

- **Defensive `raise AssertionError("unreachable")` branches** guarding invariants that higher-level type constraints already enforce.

  ```python
  else:
      raise AssertionError("unreachable")  # pragma: no cover (exhausted by Literal type above)
  ```

**Not acceptable:**

- Any use that avoids writing a test for logic that could be tested. If you can write a test, write it.
- Any use without an inline explanatory comment. The comment is how a reviewer confirms the exclusion is genuine.

## Commit Conventions

[Conventional Commits](https://www.conventionalcommits.org/) style is preferred but not enforced by CI:

- `feat:` — new feature
- `fix:` — bug fix
- `docs:` — documentation changes
- `test:` — test additions or changes
- `refactor:` — code restructuring without behavior changes
- `chore:` — build, tooling, or dependency changes

## Continuous Integration

Every pull request runs two GitHub Actions jobs defined in [`.github/workflows/ci.yml`](.github/workflows/ci.yml): `python` and `observatory`.

- The `python` job runs `just check`.
- The `observatory` job runs `npm run lint`, `npm run typecheck`, and `npm run test` inside `observatory/`.
- Service-dependent tests (marked `docker`) are skipped in CI for v0.1.1. Run them locally with `just check docker=true` (or `just ci`) when Docker is available. Real-service tests live in `validation/` and are run via `just validate`.

Releases are published automatically via [`.github/workflows/release.yml`](.github/workflows/release.yml) when a GitHub release is cut — the workflow uses PyPI trusted publishing (no API tokens) and uploads a PEP 740 provenance attestation alongside each artifact, generated keylessly from the GitHub Actions OIDC identity. See [`SECURITY.md`](SECURITY.md#release-artefact-provenance) for how to verify it.

## Merge policy, branch protection, and break-glass

### Merge mechanism

Nanitics prefers a clean, linear `main` history where each commit
corresponds to a reviewed PR. The default merge mechanism is
**squash-merge**. Merge commits are permitted but reserved for cases
where preserving multi-commit history is load-bearing (release merges,
long-running feature branches where the intermediate commits carry
reviewable meaning). Rebase-merge is also permitted.

### Repository merge settings

In addition to branch protection, the repository enables three
convenience toggles at the GitHub-repo level:

- `allow_auto_merge: true` — maintainers can flip auto-merge on a PR
  (`gh pr merge --auto --squash`) and GitHub merges once required checks
  pass. Auto-merge is opt-in per PR; the repository deliberately does
  **not** apply auto-merge automatically via a workflow on externally
  authored PRs (doing so would let an attacker land code without review).
- `delete_branch_on_merge: true` — the source branch is deleted on
  merge. Recovery via `git reflog` remains possible if needed.
- `allow_update_branch: true` — adds an "Update branch" button on the PR
  page so a contributor can sync from `main` without leaving GitHub.

These toggles are not stored as code (GitHub does not expose a
single source-of-truth file for them). To audit or update:

```sh
gh api /repos/nanitics/nanitics --jq '{auto_merge: .allow_auto_merge, delete_branch: .delete_branch_on_merge, update_branch: .allow_update_branch}'
gh api -X PATCH /repos/nanitics/nanitics -F allow_auto_merge=true -F delete_branch_on_merge=true -F allow_update_branch=true
```

### Required status checks on `main`

Branch protection on `main` requires the following checks to pass
before a PR can be merged:

- `Python (3.11)` — from `ci.yml`, runs `just check` on Python 3.11.
- `Python (3.13)` — from `ci.yml`, runs `just check` on Python 3.13.
- `Build docs` — from `docs.yml`, builds the documentation site.
- `CodeQL` — from `codeql.yml`, runs CodeQL static analysis on the Python package and the TypeScript frontend.
- `DCO` — from the [DCO GitHub App](https://github.com/apps/dco), verifies every commit carries a `Signed-off-by` line.

The `Python (3.12)`, `Observatory`, and `API Surface` jobs from `ci.yml`
run on every PR but are **advisory-only and not required to merge**.
They surface drift for review without gating routine work.

### No force-push, no direct push

Branch protection forbids force-push to `main`. Every change to `main`
lands through a pull request. There is one narrow exception: the
break-glass rule below.

### Break-glass

Nanitics is pre-1.0 and has one maintainer. In genuinely urgent
situations (a critical security patch that cannot wait for CI; a
broken `main` blocking all other work) the maintainer may bypass
branch protection using GitHub's admin-bypass mechanism. Bypassing is
not a routine operation.

When a bypass is used, the maintainer commits to producing a public
audit-trail artifact — either a follow-up issue that names the bypass,
links the offending commit(s), and explains why the bypass was needed,
or a retroactive PR-against-self that carries the same audit detail.
The audit trail is the accountability seam that keeps the break-glass
from becoming a routine bypass.

As the maintainer roster grows (see `MAINTAINERS.md`), the break-glass
rule evolves into a two-maintainer bypass requirement. Until then, the
solo maintainer's personal commitment to the audit trail is the rule.

## Reporting Issues

See [docs/getting-help.md](docs/getting-help.md) for which channel (Issues, Discussions, SECURITY.md) fits which kind of question. Issue templates under [`.github/ISSUE_TEMPLATE/`](.github/ISSUE_TEMPLATE/) guide the bug, feature, question, and documentation flows. Vulnerabilities go through [SECURITY.md](SECURITY.md), never through issues.
