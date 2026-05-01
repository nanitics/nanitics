"""In-memory fixtures backing the judge-routing runner's tools.

Pure data — no SDK imports. Five invoices, three accounts, a tiny
policy doc, and a small KB. The mutation tools (``issue_refund``,
``reset_password``, ``update_profile``, ``escalate_bug``) write to the
mutable ``MUTABLE_STATE`` dict in this module; tests reset that dict
via the ``reset_state`` fixture so module reuse across tests does not
leak state.

The fixtures are deliberately tiny — they exist to make trace
inspection legible (one invoice id, one bug id) rather than to model a
realistic data store. Callers that need richer data should swap this
module out at runner-construction time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# ── Frozen data records ───────────────────────────────────────


@dataclass(frozen=True)
class Invoice:
    """One billing invoice."""

    invoice_id: str
    account_id: str
    amount: float
    status: str
    issued_at: str


@dataclass(frozen=True)
class Account:
    """One customer account."""

    account_id: str
    email: str
    name: str
    subscription_tier: str


@dataclass(frozen=True)
class PolicyClause:
    """One policy-document clause."""

    policy_id: str
    section: str
    body: str


@dataclass(frozen=True)
class KBArticle:
    """One knowledge-base article."""

    article_id: str
    title: str
    body: str


# ── Static fixture data ───────────────────────────────────────


INVOICES: tuple[Invoice, ...] = (
    Invoice("INV-1001", "ACC-001", 49.0, "open", "2026-04-01"),
    Invoice("INV-1002", "ACC-001", 49.0, "paid", "2026-03-01"),
    Invoice("INV-1003", "ACC-002", 199.0, "paid", "2026-04-02"),
    Invoice("INV-1004", "ACC-003", 19.0, "open", "2026-04-15"),
    Invoice("INV-1005", "ACC-002", 199.0, "open", "2026-04-20"),
)

ACCOUNTS: tuple[Account, ...] = (
    Account("ACC-001", "ada@example.com", "Ada Lovelace", "pro"),
    Account("ACC-002", "grace@example.com", "Grace Hopper", "enterprise"),
    Account("ACC-003", "alan@example.com", "Alan Turing", "free"),
)

POLICY_CLAUSES: tuple[PolicyClause, ...] = (
    PolicyClause(
        "POL-REFUND",
        "1.2",
        "Refunds for paid invoices are issued within 14 days of the original charge.",
    ),
    PolicyClause(
        "POL-DATA",
        "3.4",
        "Customer data is retained for 30 days after account deletion.",
    ),
    PolicyClause(
        "POL-AUP",
        "2.1",
        "Acceptable use prohibits scraping and resale of API responses.",
    ),
)

KB_ARTICLES: tuple[KBArticle, ...] = (
    KBArticle(
        "KB-001",
        "Resolving 401 Unauthorized",
        "A 401 typically indicates an expired API key; rotate via the dashboard.",
    ),
    KBArticle(
        "KB-002",
        "Webhook delivery retries",
        "Webhooks retry with exponential backoff for up to 24 hours.",
    ),
    KBArticle(
        "KB-003",
        "Service status page",
        "The status page reports operational, degraded, or outage states.",
    ),
)

SERVICE_STATUS: dict[str, str] = {
    "api": "operational",
    "webhooks": "degraded",
    "dashboard": "operational",
}


# ── Mutable runner state ──────────────────────────────────────

# Mutated by the mutation tools. Tests reset via :func:`reset_state`.
MUTABLE_STATE: dict[str, Any] = {
    "refunds": [],  # list[dict]: appended by ``issue_refund``
    "password_resets": {},  # dict[str, str]: account_id → reset_at
    "profile_updates": {},  # dict[str, dict]: account_id → changes
    "bugs": [],  # list[dict]: appended by ``escalate_bug``
}


def reset_state() -> None:
    """Clear all mutable state. Tests call this between runs."""
    MUTABLE_STATE["refunds"] = []
    MUTABLE_STATE["password_resets"] = {}
    MUTABLE_STATE["profile_updates"] = {}
    MUTABLE_STATE["bugs"] = []


# ── Lookup helpers ────────────────────────────────────────────


def find_invoice(invoice_id: str) -> Invoice:
    """Return the invoice with *invoice_id* or raise ``ValueError``."""
    for invoice in INVOICES:
        if invoice.invoice_id == invoice_id:
            return invoice
    raise ValueError(f"unknown invoice_id: {invoice_id}")


def find_account_by_id(account_id: str) -> Account:
    """Return the account with *account_id* or raise ``ValueError``."""
    for account in ACCOUNTS:
        if account.account_id == account_id:
            return account
    raise ValueError(f"unknown account_id: {account_id}")


def find_account_by_email(email: str) -> Account:
    """Return the account with *email* or raise ``ValueError``."""
    for account in ACCOUNTS:
        if account.email == email:
            return account
    raise ValueError(f"unknown email: {email}")


def find_clause(policy_id: str, section: str) -> PolicyClause:
    """Return the policy clause matching *policy_id* + *section* or raise."""
    for clause in POLICY_CLAUSES:
        if clause.policy_id == policy_id and clause.section == section:
            return clause
    raise ValueError(f"unknown policy clause: {policy_id} §{section}")
