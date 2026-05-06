# Governance

Nanitics is pre-1.0. This document describes what adopters and contributors
can rely on: how decisions land, who has final say, how often releases ship,
and where the deprecation contract lives. It is a commitment to a process,
not a full record of every maintainer practice.

## Decision process

Nanitics operates under a single-maintainer model pre-1.0. Significant
changes — new features, API shifts, roadmap moves, breaking changes — land
through pull request review by the maintainer(s) named in the roster file.
A decision lands when the maintainer approves and merges the PR.

Out-of-PR decisions (scope changes, feature removals, roadmap re-sequencing)
are announced in the *Announcements* category of GitHub Discussions with a
link to the originating issue or Discussion so the reasoning stays
discoverable.

The mechanics of preparing and landing a PR are documented in
[CONTRIBUTING.md](CONTRIBUTING.md); this document is the project-level
policy that sits above them.

In genuinely urgent situations (critical security patches, broken `main`
blocking all work) the maintainer may bypass branch protection via
GitHub's admin-bypass mechanism. Bypasses are not routine and carry a
public audit-trail commitment. See
[CONTRIBUTING.md § Break-glass](CONTRIBUTING.md#break-glass) for the full
policy and how the rule evolves as the roster grows.

## Maintainer roles

Nanitics has one role today: **Maintainer**. Maintainers are listed in
[MAINTAINERS.md](MAINTAINERS.md), which is the roster `CODEOWNERS` routes
PR review to. The roster evolves per that file's *How the roster evolves*
section — the signal is earned track record, not request.

Post-1.0 governance maturation (additional roles, committees, step-up
ladders) is driven by adopter signal, not pre-committed.

## Release cadence

Nanitics releases ship when meaningful work has landed and the quality gate
is green. There is no scheduled minor-release cadence pre-1.0.

This matches the [breaking-change policy](docs/deprecation-policy.md), which
states that pre-1.0 releases carry no minimum calendar-time guarantee
between a deprecation notice and its removal. The policy is the authoritative
contract for what "breaking change" means, what notice is given, and how
adopters are expected to track the public API surface.

## Proposing a significant change

For small fixes and documented behaviour changes, open a PR directly.

For larger changes — new APIs, new primitives, shifts in scope — the entry
point is GitHub Discussions → *Ideas*. Ideas that gain traction become
issues; issues that require design attention happen in a Discussion or
issue first, then in code.

**There is no RFC process today.** This is deliberate: an RFC process
at a solo-maintained pre-1.0 project would add ceremony without producing
review quality that PR review and Discussion threads do not already
produce. A formal RFC process is a post-launch question driven by adopter
signal, not a pre-committed path.

## Vulnerabilities

Report vulnerabilities through [SECURITY.md](SECURITY.md). Do not open
public issues for security findings.

## Code of Conduct

Participation in the Nanitics project is governed by
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## Legal scaffolding

Nanitics ships under Apache-2.0 (see [LICENSE](LICENSE)). Two additional
open-core legal documents sit alongside this governance document:

- A **Developer Certificate of Origin (DCO)** sign-off requirement on
  every commit (`git commit --signoff`), enforced by the DCO GitHub App.
  See [CONTRIBUTING.md](CONTRIBUTING.md#developer-certificate-of-origin)
  for the workflow and what each sign-off certifies. Contributions are
  accepted under Apache-2.0 on an *inbound=outbound* basis — contributions
  enter the project under the same licence the project ships under — with
  no separate Contributor License Agreement.
- A **Trademark policy** ([TRADEMARK.md](TRADEMARK.md)) describing who
  owns the Nanitics name and mark and what uses are permitted.

## How this document evolves

Governance is a commitment, not a specification — it evolves with the
project. Changes to this file flow through PR review like any other change.

Post-1.0 governance maturation (council models, step-up paths,
multi-maintainer coordination, formal RFC processes) is driven by adopter
signal — concrete need surfaced in Discussions or by contribution patterns —
not by pre-committed predictions about where the project will go.
