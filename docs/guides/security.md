# Security

> For API details — signatures, fields, constraints — read the docstrings in your editor, in the source tree under [`nanitics/`](../../nanitics/), or browse them at [docs.nanitics.dev](https://docs.nanitics.dev/). `nanitics.__all__` is the authoritative public surface.

> Looking for the vulnerability-reporting contract? See [`SECURITY.md`](../../SECURITY.md).

The Nanitics SDK makes exactly one load-bearing security claim: the
**no-leakage invariant** — provider credentials, auth headers, and raw
HTTP context never appear in any trace event emitted by a shipped LLM
client. That claim is locked in by a release-gate test. Everything else
in an agent deployment — prompt design, tool-input validation, trace
content scrubbing, auth on the Observatory, rotation of API keys, the
posture of any host running untrusted code — is yours. This guide names
the boundary, points at the mechanisms the SDK ships for each side,
and is honest about what the SDK cannot do.

For the event model, redaction protocol, and trace-surface-hygiene
detail this guide points at, see
[observability.md § Trace Surface Hygiene](observability.md#trace-surface-hygiene).
For the vulnerability-reporting process, see
[`SECURITY.md`](../../SECURITY.md).

## Threat model

The attacker is anyone who can place adversarial content into any
input the agent reads — a user turn, a retrieved document, a tool's
output, a memory item. That content can contain instructions the
model will follow. The blast radius is whatever the agent's tool set
can do in the environments its credentials reach — not whatever the
prompt says the agent "should" do. Every mechanism in this guide
narrows one of two things: what reaches the model, or what the model
can cause once it decides to act.

## Trust boundary summary

Two surfaces carry two different sets of obligations. The split is the
same one [`observability.md`](observability.md#trace-surface-hygiene)
describes in full; the short form:

| Surface | What lives here | Who owns it |
|---|---|---|
| SDK surface | Event types, span/trace IDs, timing and usage numbers, LLM response content, tool-result shapes, the no-leakage invariant over every shipped LLM client. | SDK (invariant-enforced). |
| Adopter surface | Prompts, tool inputs and outputs, custom event fields, tool exception messages, Observatory auth and tenancy, retention policy, redaction of adopter content. | Adopter. |

The full bifurcation — including the release-gate test name, the
`RedactionHook` wire-in points, and the "what goes where" table — is in
[observability.md § Trace Surface Hygiene](observability.md#trace-surface-hygiene).
Do not copy the table here; read it there.

## OWASP Top 10 for Agentic Applications (2026) alignment

The table below maps the [OWASP Top 10 for Agentic Applications
2026](https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/)
onto the Nanitics v0.1.0 surface. "SDK" rows are addressed by
mechanisms this guide links to. "Adopter" rows are yours to address;
the SDK provides the typed boundary where your controls attach.
"Shared" rows need both. "Out of scope at v0.1.0" means the SDK ships
no mechanism for this risk and does not pretend to — bring your own.

| ID | OWASP title | Nanitics posture |
|---|---|---|
| ASI01 | Agent Goal Hijack | Adopter — typed tool boundary provided as attachment point; see [Prompt injection — honest posture](#prompt-injection--honest-posture). |
| ASI02 | Tool Misuse & Exploitation | Shared — adopter authors tool allow-lists and argument validation; SDK ships iteration and tool-call limiters (see [safety.md](safety.md)). `IterationLimiter` is default-on; `ToolCallLimiter` is opt-in via `max_tool_calls=…` on `ReactAgent`. |
| ASI03 | Agent Identity & Privilege Abuse | Adopter — scope and rotate provider credentials; see [API-key handling](#api-key-handling). |
| ASI04 | Agentic Supply Chain Compromise | SDK for `nanitics` itself (see [Supply-chain posture](#supply-chain-posture)); adopter owns the supply chain of their own tools and any MCP servers they mount. |
| ASI05 | Unexpected Code Execution | SDK — `DockerSandbox` with documented limits (see [DockerSandbox honest limits](#dockersandbox-honest-limits)). Stronger isolation is adopter-owned for high-consequence production. |
| ASI06 | Memory & Context Poisoning | Out of scope at v0.1.0 — see [Known limitations at v0.1.0](#known-limitations-at-v010). |
| ASI07 | Insecure Inter-Agent Communication | Out of scope by architecture — multi-agent primitives run in-process at v0.1.0; there is no wire protocol the SDK authors or hardens. |
| ASI08 | Cascading Agent Failures | Shared — SDK ships iteration limits, tool-call limits, and cancellation tokens (see [safety.md](safety.md)); adopter owns circuit-breaking at tool boundaries. |
| ASI09 | Human-Agent Trust Exploitation | Shared — SDK ships `ApprovalGate` and `ApprovalWrapped` HITL primitives; adopter decides what to gate and how to present it (HITL failures raise typed `ApprovalUnavailableError` / `ApprovalTimeoutError`). |
| ASI10 | Rogue Agents | Out of scope at v0.1.0 — traceable via the observability surface for post-hoc audit, not detected by the SDK; see [Known limitations at v0.1.0](#known-limitations-at-v010). |

## Prompt injection — honest posture

The SDK cannot eliminate prompt injection. An LLM that receives a tool
output, a memory item, a retrieved document, or a user turn containing
adversarial instructions may follow those instructions — this is a
property of LLMs, not of any framework built on top of them. What the
SDK does give you is a typed, inspectable tool boundary where
adopter-owned validation can attach. Use it.

The three snippets below demonstrate common validation patterns against
real Nanitics constructs. Each snippet is illustrative — adapt the
types, paths, and bounds to your own application.

**Type-enforced boundary with Pydantic.** This prevents malformed
arguments from reaching your handler; it does not prevent the LLM from
choosing semantically bad values within the allowed type.

```python
from pydantic import BaseModel, Field

from nanitics import tool


class GreetArgs(BaseModel):
    name: str = Field(min_length=1, max_length=64)


@tool("greet", "Greet the named person.")
async def greet(args: GreetArgs) -> str:
    return f"Hello, {args.name}!"
```

**Allow-list on a sensitive-action tool.** This prevents the LLM from
reaching outside the intended directory; it does not prevent it from
asking to write a pathologically large file within the allowed tree.

```python
from pathlib import Path

from nanitics import tool

ALLOWED_ROOT = Path("/srv/agent-workspace").resolve()


@tool("write_note", "Write text to a file under the agent workspace.")
async def write_note(path: str, body: str) -> str:
    target = (ALLOWED_ROOT / path).resolve()
    if not target.is_relative_to(ALLOWED_ROOT):
        raise ValueError(f"path escapes workspace: {path}")
    target.write_text(body)
    return str(target)
```

**Length and shape check on free-text input.** This caps the blast
radius of a prompt that tries to smuggle a long instruction past a
later stage; it does not prevent a short injection from succeeding.

```python
from nanitics import tool

MAX_QUERY_CHARS = 500


@tool("search", "Run a search query against the knowledge base.")
async def search(query: str) -> str:
    if len(query) > MAX_QUERY_CHARS:
        raise ValueError(f"query length {len(query)} exceeds {MAX_QUERY_CHARS}")
    return _dispatch_search(query)
```

None of these patterns, alone or together, "prevents" prompt injection.
They narrow the blast radius of an injected instruction to whatever a
well-validated argument can still cause at your tool boundary. Pair
them with the posture the rest of this guide describes — sandboxing
for code execution, redaction for traces, rotation and least-privilege
for API keys.

For the full tool-authoring contract, see [tools.md](tools.md).

## Content redaction for trace events

Trace events carry adopter content (prompts, tool inputs and outputs,
custom event fields). Redaction of that content is adopter-owned via
the `RedactionHook` protocol, wired into `TraceCollector` or
`TracedExecutor.execute`. The SDK ships no default scrubber on purpose
— a domain-neutral default would be a false promise.

For the protocol, wire-in points, call ordering, and an example hook,
see [observability.md § Trace Surface Hygiene](observability.md#trace-surface-hygiene).

## DockerSandbox honest limits

`DockerSandbox` executes LLM-generated code inside a hardened Docker
container. The class docstring in
[`nanitics/safety/sandbox/docker.py`](../../nanitics/safety/sandbox/docker.py)
is the single source of truth for what the container blocks and what it
does not; read it there. The short form:

Container isolation **blocks** host filesystem access outside the bind
mount, privilege escalation (`no-new-privileges`), and resource
exhaustion (PID, memory, CPU limits). On outbound network, the posture
needs to be read as three separate claims, not one:

- **DNS-based egress is blocked** — DNS resolution is stubbed to
  loopback, so any code that relies on resolving a hostname cannot
  reach the public internet.
- **Raw-socket / ICMP attacks are blocked** — `NET_RAW` is dropped,
  preventing crafted-packet and low-level network probes.
- **Direct-IP TCP/UDP egress is NOT blocked at the Docker level** —
  the default bridge is shared with the host for the tool-bridge TCP
  channel, and Docker does not firewall outbound traffic on it by
  itself. Full network isolation requires a host firewall or a custom
  Docker network with an egress policy you manage.

It **does not block** a determined escape exploiting a Docker daemon
CVE, side-channel or timing attacks against the host kernel, or data
exfiltration through any network destination you explicitly allow-list
or leave reachable at the host level.

`DockerSandbox` is the right tool for untrusted LLM-generated code in a
development or low-consequence context. Running untrusted code against
high-value production state requires stronger isolation the SDK does
not ship — a microVM, a dedicated host, or an external sandboxing
service. For the runtime-control mechanisms this pairs with (iteration
limits, cancellation), see [safety.md](safety.md).

## API-key handling

Nanitics never writes provider credentials, auth headers, or raw HTTP
context into trace events. That property is enforced for every shipped
LLM client (`AnthropicLLMClient`, `MistralLLMClient`,
`OpenAILLMClient`, `LiteLLMClient`) by the release-gate invariant test
at [`tests/test_no_leakage_invariant.py`](../../tests/test_no_leakage_invariant.py).
If the test ever fails, the offending event type and field path are
named and shipping is blocked.

That is the SDK's half. The adopter's half:

- **Source keys from environment variables.** Read `ANTHROPIC_API_KEY`,
  `MISTRAL_API_KEY`, `OPENAI_API_KEY` (or whatever your provider uses)
  from `os.environ` at process startup and pass them to the client
  constructor. Do not commit keys to your repository. Do not hard-code
  them in code paths the SDK serialises (system prompts, tool inputs).
- **Rotate on a cadence you own.** The SDK does not rotate keys; your
  deployment, secrets manager, or provider console does. A key compromised
  by an adopter-side leak is rotated by the adopter.
- **Scope keys to the minimum privilege your agent needs.** Most
  providers support project- or workspace-scoped keys; prefer those
  over account-root keys.
- **Never echo keys into your own events.** If you emit custom events
  from application code, treat the key as the SDK treats it — never in
  payloads, never in error messages.

> For the end-to-end secret-management pattern in the shipped compose, see [Deployment](deployment.md#secrets-and-environment).

## Supply-chain posture

Every PyPI release of `nanitics` carries a PEP 740 provenance attestation, generated keylessly from this repository's GitHub Actions trusted publisher. Verification instructions and the full policy live in [`SECURITY.md § Release artefact provenance`](../../SECURITY.md#release-artefact-provenance) — not duplicated here.

## Known limitations at v0.1.0

This section names what the SDK does not do, so you know where to
bring your own controls. Absence here is deliberate, not a roadmap.

- **No default trace redactor.** `RedactionHook` is adopter-owned;
  see [observability.md § Trace Surface Hygiene](observability.md#trace-surface-hygiene).
  A domain-neutral default would be a false promise.
- **No LLM rate limits or cost caps in the SDK.** Iteration and
  tool-call limits in [safety.md](safety.md) bound control-flow; they
  do not cap spend. `IterationLimiter` is default-on for loop-based
  agents; `ToolCallLimiter` is opt-in (set `max_tool_calls=…` on
  `ReactAgent`) — cost control is an adopter decision. Use your
  provider's quota, a proxy, or a budget wrapper.
- **No automatic prompt-injection scanning.** The SDK ships no
  classifier, filter, or pattern matcher over inputs. Adopters who
  need one wire it into their tool handlers or prompt-assembly layer.
- **No memory or retrieved-document sanitisation.** Content fed back
  to the model from memory, retrieval, or a prior tool result is not
  inspected by the SDK. Treat all such content as attacker-controllable.
- **No multi-tenant isolation on the Observatory.** The SDK and the
  Observatory at v0.1.0 assume a single scope; running a multi-tenant
  Observatory is adopter-owned, per the
  [Observatory integration guide's production framing](observatory-integration.md#for-production).
- **No inter-agent wire protocol.** Multi-agent primitives
  (delegation, broadcast, debate, consensus, bidding, blackboard, peer
  network) run in-process; there is no protocol the SDK authors or
  hardens.
- **No rogue-agent or drift detection.** The SDK produces a complete
  event trace for post-hoc audit (see [observability.md](observability.md));
  detection of emergent misbehaviour is adopter-owned.
