# Deprecation Policy (pre-1.0)

This document is the contract between Nanitics maintainers and adopters on
what the public API promises, what "deprecated" means, and how breaking
changes are handled before the 1.0 release. It sits alongside `vision.md` as
a commitment the project makes — not as a guide to building on top of the
SDK.

## 1. What counts as public

The **public surface** is the set of names exported from the top-level
`nanitics` package — concretely, `nanitics.__all__`. A name is public if and
only if it appears in that list.

Everything not in `nanitics.__all__` is **internal** and may change in any
release without notice. This includes:

- Symbols with a leading underscore (e.g. `nanitics._deprecation`, any
  `_private_helper`).
- Submodules that are not themselves re-exported (e.g. internal organization
  under `nanitics.strategies.*` or `nanitics.infrastructure.*`).
- Attributes accessed via dotted paths that are not themselves listed in
  `nanitics.__all__` — reaching into a subsystem to pull a type that is not
  re-exported is reaching into internals.

The committed snapshot `tests/public_api_surface.txt` records the public
surface at each commit. The `api-surface` CI job diffs the live `__all__`
against that snapshot and surfaces drift as workflow warning annotations on
the PR checks page. The job is advisory — it does not block merges.

## 2. What "deprecated" means here

A public symbol marked with `@nanitics.deprecated("reason")` carries the full
semantics of [PEP 702](https://peps.python.org/pep-0702/):

- Calling the symbol emits a `DeprecationWarning` with the supplied reason.
- The symbol gains a `__deprecated__` attribute that type checkers and
  documentation generators recognize.
- The symbol remains fully functional until the release in which it is
  removed. Deprecation announces intent to remove; it does not itself change
  behavior beyond emitting the warning.

`nanitics.deprecated` is a direct re-export of the stdlib
`warnings.deprecated`; the decorator is the same object and behaves
identically. This keeps authoring simple and guarantees that every tool in
the Python ecosystem already understands it.

## 3. Pre-1.0 notice period

Before the 1.0 release, Nanitics may remove deprecated public symbols in any
subsequent minor release, provided the removal is called out in
`CHANGELOG.md`. There is **no minimum calendar-time guarantee** pre-1.0 —
the promise is that removal is announced (via `@deprecated` and the
changelog) before it happens, not that authors get a fixed number of weeks
or months of parallel support.

This notice model is deliberately lightweight. Pre-1.0 the priority is
learning the right shapes quickly; a heavyweight notice period would slow
that down for little benefit while the SDK's surface is still settling.

## 4. Breaking-change process

Intentional changes to the public surface follow these steps, in order:

1. **Announce.** Add an entry to `CHANGELOG.md` describing the change and
   (for renames/moves) the migration path.
2. **Mark.** Apply `@nanitics.deprecated("reason; use <replacement>")` to the
   affected symbol. The warning text should name the replacement explicitly.
3. **Keep for at least one minor release** when removal is the eventual
   outcome. "At least one minor release" means the deprecated symbol is
   present in the minor release in which the warning is first shipped; a
   later minor release may remove it.
4. **Regenerate the snapshot.** Changes that add or remove names in
   `nanitics.__all__` must regenerate `tests/public_api_surface.txt` in the
   same pull request. The regeneration command is:

   ```bash
   uv run python -c "import nanitics; \
     print('\n'.join(sorted(nanitics.__all__)))" \
     > tests/public_api_surface.txt
   ```

The `api-surface` CI job surfaces snapshot drift as a warning when the
snapshot is not regenerated alongside a surface change. The warning is the
reviewer's signal — it does not block merging — so reviewers must confirm
that surface drift is intentional before approval.

## 5. Not promised pre-1.0

The following are **not** part of the pre-1.0 contract and will be
considered post-launch:

- **Finer stability tiers.** There is no `@stable` / `@provisional` /
  `@internal` axis. Everything in `nanitics.__all__` is "public pre-1.0"
  and subject to this policy; everything else is internal. Finer tiers
  belong to the post-launch roadmap.
- **Signature-level semver enforcement.** The CI drift check compares
  name sets only. Parameter changes, return-type changes, and kind changes
  (class vs function, sync vs async) are not detected automatically. They
  are caught by code review and by the changelog discipline above.

Revisiting either of these is a post-launch decision, made against real
adoption signal rather than speculative need.

## 6. Reporting unintended breakage

If a public symbol changes or disappears without following this process,
that is a bug, not a breaking-change event:

- **Security regressions** (including denial-of-service, RCE, or data-leak
  pathways introduced by a surface change) — see
  [`SECURITY.md`](../SECURITY.md).
- **Everything else** — open a GitHub issue at the
  [issue tracker](https://github.com/nanitics/nanitics/issues)
  with the affected symbol, the release in which the breakage appeared, and
  the reproduction.

## 7. Hosted reference versioning (pre-1.0)

The hosted API reference at https://docs.nanitics.dev/
tracks `main`. There are no per-tag or per-version subpaths, no `/latest` vs
`/stable` split, and no published archive of prior docstring states. The
current reference is always a projection of the current source — the site
moves when the code moves.

Post-1.0 versioning (per-release subpaths, archived reference for each minor,
or a snapshot pin on major releases) is a future decision driven by actual
adopter signal. Pre-1.0 the priority is keeping the reference honest about
HEAD, which a single-tracked site does by construction.
