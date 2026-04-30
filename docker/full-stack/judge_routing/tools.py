"""Tool factories for the judge-routing runner's specialists.

Eleven tools across four specialties:

- billing: ``lookup_invoice``, ``issue_refund``
- technical: ``search_kb``, ``check_service_status``, ``escalate_bug``
- account: ``lookup_account``, ``reset_password``, ``update_profile``
- policy: ``lookup_policy``, ``cite_clause``

Every tool validates its inputs through a Pydantic model
(:class:`~nanitics.core.tools.function_tool.FunctionTool` does the
validation plumbing). Unknown ids and missing matches raise
``ValueError`` — tools surface failures rather than masking them with
empty results, so the agent sees a real error message and can adjust.

Mutation tools (``issue_refund``, ``reset_password``,
``update_profile``, ``escalate_bug``) write to the module-level
``fixtures.MUTABLE_STATE`` dict; the ``reset_state`` helper there is
called by the test fixture to keep test isolation honest.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field

from nanitics import FunctionTool, Tool, ToolResult

from . import fixtures

# ── Tool parameter models ─────────────────────────────────────


class _LookupInvoiceParams(BaseModel):
    invoice_id: str = Field(..., min_length=1)


class _IssueRefundParams(BaseModel):
    invoice_id: str = Field(..., min_length=1)
    amount: float = Field(..., gt=0)
    reason: str = Field(..., min_length=1)


class _SearchKBParams(BaseModel):
    query: str = Field(..., min_length=1)


class _CheckServiceStatusParams(BaseModel):
    service: str = Field(..., min_length=1)


class _EscalateBugParams(BaseModel):
    summary: str = Field(..., min_length=1)


class _LookupAccountParams(BaseModel):
    email: str = Field(..., min_length=1)


class _ResetPasswordParams(BaseModel):
    account_id: str = Field(..., min_length=1)


class _UpdateProfileParams(BaseModel):
    account_id: str = Field(..., min_length=1)
    changes: dict[str, str]


class _LookupPolicyParams(BaseModel):
    topic: str = Field(..., min_length=1)


class _CiteClauseParams(BaseModel):
    policy_id: str = Field(..., min_length=1)
    section: str = Field(..., min_length=1)


# ── Billing tools ─────────────────────────────────────────────


def build_lookup_invoice_tool() -> Tool:
    """``lookup_invoice(invoice_id)`` — return one invoice's details."""

    async def lookup_invoice(invoice_id: str) -> ToolResult:
        invoice = fixtures.find_invoice(invoice_id)
        return ToolResult(
            content=(
                f"Invoice {invoice.invoice_id}: ${invoice.amount:.2f} "
                f"({invoice.status}), issued {invoice.issued_at}, "
                f"account {invoice.account_id}."
            ),
            metadata={
                "invoice_id": invoice.invoice_id,
                "account_id": invoice.account_id,
                "amount": invoice.amount,
                "status": invoice.status,
                "issued_at": invoice.issued_at,
            },
        )

    return FunctionTool(
        fn=lookup_invoice,
        name="lookup_invoice",
        description="Look up one billing invoice by its invoice_id.",
        parameters_model=_LookupInvoiceParams,
    )


def build_issue_refund_tool() -> Tool:
    """``issue_refund(invoice_id, amount, reason)`` — record a refund."""

    async def issue_refund(invoice_id: str, amount: float, reason: str) -> ToolResult:
        invoice = fixtures.find_invoice(invoice_id)
        refund = {
            "refund_id": f"REF-{len(fixtures.MUTABLE_STATE['refunds']) + 1:04d}",
            "invoice_id": invoice.invoice_id,
            "amount": amount,
            "reason": reason,
            "issued_at": datetime.now(UTC).isoformat(),
        }
        fixtures.MUTABLE_STATE["refunds"].append(refund)
        return ToolResult(
            content=(f"Refund {refund['refund_id']} for ${amount:.2f} recorded against invoice {invoice.invoice_id}."),
            metadata=refund,
        )

    return FunctionTool(
        fn=issue_refund,
        name="issue_refund",
        description="Issue a refund against an invoice. Mutates the in-memory ledger.",
        parameters_model=_IssueRefundParams,
    )


# ── Technical tools ───────────────────────────────────────────


def build_search_kb_tool() -> Tool:
    """``search_kb(query)`` — substring search over title + body."""

    async def search_kb(query: str) -> ToolResult:
        needle = query.lower()
        hits = [
            article
            for article in fixtures.KB_ARTICLES
            if needle in article.title.lower() or needle in article.body.lower()
        ]
        if not hits:
            return ToolResult(
                content=f"No KB articles match '{query}'.",
                metadata={"query": query, "hits": []},
            )
        rendered = "\n".join(f"- {a.article_id} {a.title}: {a.body}" for a in hits)
        return ToolResult(
            content=rendered,
            metadata={
                "query": query,
                "hits": [{"article_id": a.article_id, "title": a.title} for a in hits],
            },
        )

    return FunctionTool(
        fn=search_kb,
        name="search_kb",
        description="Search the knowledge base by substring against title and body.",
        parameters_model=_SearchKBParams,
    )


def build_check_service_status_tool() -> Tool:
    """``check_service_status(service)`` — return one service's status."""

    async def check_service_status(service: str) -> ToolResult:
        if service not in fixtures.SERVICE_STATUS:
            raise ValueError(f"unknown service: {service}")
        status = fixtures.SERVICE_STATUS[service]
        return ToolResult(
            content=f"Service '{service}' is {status}.",
            metadata={"service": service, "status": status},
        )

    return FunctionTool(
        fn=check_service_status,
        name="check_service_status",
        description="Return the operational status of one named service.",
        parameters_model=_CheckServiceStatusParams,
    )


def build_escalate_bug_tool() -> Tool:
    """``escalate_bug(summary)`` — append to the in-memory bug tracker."""

    async def escalate_bug(summary: str) -> ToolResult:
        bug_id = f"BUG-{len(fixtures.MUTABLE_STATE['bugs']) + 1:04d}"
        record = {
            "bug_id": bug_id,
            "summary": summary,
            "filed_at": datetime.now(UTC).isoformat(),
        }
        fixtures.MUTABLE_STATE["bugs"].append(record)
        return ToolResult(
            content=f"Bug {bug_id} filed: {summary}",
            metadata=record,
        )

    return FunctionTool(
        fn=escalate_bug,
        name="escalate_bug",
        description="File a new bug in the in-memory bug tracker.",
        parameters_model=_EscalateBugParams,
    )


# ── Account tools ─────────────────────────────────────────────


def build_lookup_account_tool() -> Tool:
    """``lookup_account(email)`` — fetch one account by email."""

    async def lookup_account(email: str) -> ToolResult:
        account = fixtures.find_account_by_email(email)
        return ToolResult(
            content=(
                f"Account {account.account_id}: {account.name} <{account.email}>, tier {account.subscription_tier}."
            ),
            metadata={
                "account_id": account.account_id,
                "email": account.email,
                "name": account.name,
                "subscription_tier": account.subscription_tier,
            },
        )

    return FunctionTool(
        fn=lookup_account,
        name="lookup_account",
        description="Look up one account by its registered email address.",
        parameters_model=_LookupAccountParams,
    )


def build_reset_password_tool() -> Tool:
    """``reset_password(account_id)`` — record a password reset."""

    async def reset_password(account_id: str) -> ToolResult:
        account = fixtures.find_account_by_id(account_id)
        reset_at = datetime.now(UTC).isoformat()
        fixtures.MUTABLE_STATE["password_resets"][account.account_id] = reset_at
        return ToolResult(
            content=(f"Password reset triggered for {account.account_id} at {reset_at}."),
            metadata={"account_id": account.account_id, "reset_at": reset_at},
        )

    return FunctionTool(
        fn=reset_password,
        name="reset_password",
        description="Trigger a password reset for one account.",
        parameters_model=_ResetPasswordParams,
    )


def build_update_profile_tool() -> Tool:
    """``update_profile(account_id, changes)`` — record a profile diff."""

    async def update_profile(account_id: str, changes: dict[str, str]) -> ToolResult:
        if not changes:
            raise ValueError("changes must be a non-empty mapping")
        account = fixtures.find_account_by_id(account_id)
        existing = dict(fixtures.MUTABLE_STATE["profile_updates"].get(account.account_id, {}))
        existing.update(changes)
        fixtures.MUTABLE_STATE["profile_updates"][account.account_id] = existing
        rendered = ", ".join(f"{k}={v}" for k, v in changes.items())
        return ToolResult(
            content=f"Profile for {account.account_id} updated: {rendered}.",
            metadata={"account_id": account.account_id, "changes": dict(changes)},
        )

    return FunctionTool(
        fn=update_profile,
        name="update_profile",
        description="Apply a partial update to one account's profile.",
        parameters_model=_UpdateProfileParams,
    )


# ── Policy tools ──────────────────────────────────────────────


def build_lookup_policy_tool() -> Tool:
    """``lookup_policy(topic)`` — substring search across clause bodies."""

    async def lookup_policy(topic: str) -> ToolResult:
        needle = topic.lower()
        hits = [c for c in fixtures.POLICY_CLAUSES if needle in c.body.lower()]
        if not hits:
            return ToolResult(
                content=f"No policy clauses match '{topic}'.",
                metadata={"topic": topic, "hits": []},
            )
        rendered = "\n".join(f"- {c.policy_id} §{c.section}: {c.body}" for c in hits)
        return ToolResult(
            content=rendered,
            metadata={
                "topic": topic,
                "hits": [{"policy_id": c.policy_id, "section": c.section} for c in hits],
            },
        )

    return FunctionTool(
        fn=lookup_policy,
        name="lookup_policy",
        description="Search policy clauses by substring against the clause body.",
        parameters_model=_LookupPolicyParams,
    )


def build_cite_clause_tool() -> Tool:
    """``cite_clause(policy_id, section)`` — return one exact clause body."""

    async def cite_clause(policy_id: str, section: str) -> ToolResult:
        clause = fixtures.find_clause(policy_id, section)
        return ToolResult(
            content=f"{clause.policy_id} §{clause.section}: {clause.body}",
            metadata={"policy_id": clause.policy_id, "section": clause.section},
        )

    return FunctionTool(
        fn=cite_clause,
        name="cite_clause",
        description="Return the exact body of one named policy clause.",
        parameters_model=_CiteClauseParams,
    )


# ── Per-specialty tool bundles ────────────────────────────────


def billing_tools() -> list[Tool]:
    """Tool bundle for the billing specialist."""
    return [build_lookup_invoice_tool(), build_issue_refund_tool()]


def technical_tools() -> list[Tool]:
    """Tool bundle for the technical specialist."""
    return [
        build_search_kb_tool(),
        build_check_service_status_tool(),
        build_escalate_bug_tool(),
    ]


def account_tools() -> list[Tool]:
    """Tool bundle for the account specialist."""
    return [
        build_lookup_account_tool(),
        build_reset_password_tool(),
        build_update_profile_tool(),
    ]


def policy_tools() -> list[Tool]:
    """Tool bundle for the policy specialist."""
    return [build_lookup_policy_tool(), build_cite_clause_tool()]


__all__ = [
    "account_tools",
    "billing_tools",
    "build_check_service_status_tool",
    "build_cite_clause_tool",
    "build_escalate_bug_tool",
    "build_issue_refund_tool",
    "build_lookup_account_tool",
    "build_lookup_invoice_tool",
    "build_lookup_policy_tool",
    "build_reset_password_tool",
    "build_search_kb_tool",
    "build_update_profile_tool",
    "policy_tools",
    "technical_tools",
]
