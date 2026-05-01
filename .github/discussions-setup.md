# GitHub Discussions setup

This file defines the Discussions configuration for `nanitics/nanitics` and
the pinned welcome post. The category list and welcome body are applied
manually; drift is prevented by PR review on this file, not by automation.

## Categories

Four categories, matching the contact-link URLs in
[`ISSUE_TEMPLATE/config.yml`](ISSUE_TEMPLATE/config.yml):

| Name          | Slug            | Format   | Answerable | Description                              |
|---------------|-----------------|----------|------------|------------------------------------------|
| Announcements | `announcements` | post     | no         | Updates from maintainers                 |
| Ideas         | `ideas`         | post     | no         | Share ideas for new features             |
| Q&A           | `q-a`           | question | yes        | Ask the community for help               |
| Show and tell | `show-and-tell` | post     | no         | Show off something you've made           |

## Pinned welcome post

**Category:** Announcements
**Title:** `Welcome to Nanitics Discussions`
**Action after posting:** Pin.

### Body (verbatim)

```markdown
Welcome — and thanks for stopping by.

Nanitics is a composable Python SDK for building single-agent and multi-agent
AI systems. Discussions is where we talk about how to use it, what to build
with it, and where it should go next. Four categories, each with a clear
purpose:

- **Q&A** — Usage questions, design trade-offs, best-practice questions. Ask
  anything; the maintainer answers what they can and the community fills in
  the rest.
- **Ideas** — Half-formed proposals and "what if…" threads. If it isn't
  concrete enough to file as an Issue yet, it belongs here.
- **Show and tell** — Share what you're building. Patterns, demos, lessons,
  war stories — all welcome.
- **Announcements** — Release notes, roadmap updates, and other posts from
  the maintainer. Read-only for everyone else.

A few pointers before you post:

- **Reproducible defects and concrete feature requests belong in
  [Issues](https://github.com/nanitics/nanitics/issues)**, not here. If a
  Discussion turns into one, a maintainer will ask you to file it and link
  back.
- **Security vulnerabilities — do not post them here or in Issues.** Use
  [GitHub Private Vulnerability Reporting](https://github.com/nanitics/nanitics/security/advisories/new)
  or email `security@nanitics.dev`. Full process in
  [`SECURITY.md`](https://github.com/nanitics/nanitics/blob/main/SECURITY.md).
- **Contributors sign the [DCO](https://developercertificate.org/) on every
  commit** (`git commit -s`). See
  [`CONTRIBUTING.md`](https://github.com/nanitics/nanitics/blob/main/CONTRIBUTING.md) for the full flow.
- **Be kind.** All interaction in Discussions and Issues is governed by the
  [Code of Conduct](https://github.com/nanitics/nanitics/blob/main/CODE_OF_CONDUCT.md).

Not sure where something belongs? Start in Q&A. We'll help you find the right
home for it.

— The Nanitics maintainers
```
