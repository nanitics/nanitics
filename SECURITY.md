# Security Policy

> For the SDK's threat model, trust-boundary split, OWASP alignment, and
> operational security guidance, see
> [`docs/guides/security.md`](docs/guides/security.md). This file is the
> vulnerability-reporting contract only.

## Reporting a Vulnerability

**Please do NOT open a public GitHub issue for security vulnerabilities.**

If you discover a security vulnerability in Nanitics, please report it privately using one of the following methods:

1. **GitHub Private Vulnerability Reporting** (preferred): Go to [Security Advisories](https://github.com/nanitics/nanitics/security/advisories/new) and create a new private advisory.
2. **Email**: Send details to security@nanitics.dev

## What to Include

When reporting a vulnerability, please include:

- A description of the vulnerability
- Steps to reproduce the issue
- The potential impact
- Any suggested fixes (if you have them)

## Response Posture

Nanitics is a pre-1.0 project maintained by a single person. We commit to a clear process, not to hard timing guarantees:

- **Acknowledgement**: best-effort within 5 business days.
- **Triage and resolution timeline**: communicated to the reporter once the report is acknowledged. Timing depends on severity and complexity.
- **Coordinated disclosure**: we will keep you informed during triage and coordinate disclosure timing with you before publishing any advisory.

## Disclosure Policy

- We follow coordinated disclosure. We ask that you do not publicly disclose the vulnerability until we have had a chance to address it.
- Once a fix is released, we will publish a security advisory on GitHub and credit the reporter (unless they prefer to remain anonymous).

## Supported Versions

Pre-1.0 Nanitics is single-tracked on the latest release. Security fixes land in the next regular release. Backports to older releases are not part of the pre-1.0 support posture.

## Scope

This security policy applies to the Nanitics SDK Python package (`nanitics`) and its first-party dependencies. Vulnerabilities in Nanitics code that *involves* third-party SDKs — incorrect usage, mishandling of SDK return values, and similar — are in scope. Bugs *inside* third-party LLM provider SDKs (e.g., `anthropic`, `mistralai`) are not; report those to the respective maintainers.

## Release artefact provenance

Every release of `nanitics` on PyPI is accompanied by a [PEP 740](https://peps.python.org/pep-0740/) provenance attestation, generated keylessly from the project's GitHub Actions trusted publisher via sigstore. The attestation binds the PyPI artefact to the exact workflow run and commit that produced it.

- **Where to see it.** On the PyPI project page under the Release history, each file shows a "Provenance" entry when an attestation is present.
- **What it proves.** The wheel or sdist you installed came from the `pypa/gh-action-pypi-publish` step in this repository's `release.yml`, running under the trusted-publisher configuration — not from a leaked token or a third party.
- **How to verify programmatically.** Use `pip install --require-hashes` with hashes pinned from the PyPI provenance, or fetch the attestation bundle from PyPI's integrity endpoint — `https://pypi.org/integrity/nanitics/<version>/<filename>/provenance` — and verify with a sigstore client. See PyPI's current documentation for attestation verification tooling.

GPG signatures are not produced — PyPI deprecated GPG signature uploads in 2023.
