"""Pydantic v2 input/output schemas for every MCP tool."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ── start_session ────────────────────────────────────────────────────────


class StartSessionInput(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    model: str = Field(default="anthropic/claude-opus-4", max_length=128)
    system_prompt: str | None = Field(default=None, max_length=8000)


class AgentConfigOut(BaseModel):
    model_id: str
    context_limit: int
    transfer_threshold: float


class StartSessionOutput(BaseModel):
    session_id: str
    agent_config: AgentConfigOut


# ── send_prompt ─────────────────────────────────────────────────────────


class SendPromptInput(BaseModel):
    session_id: str
    message: str = Field(min_length=1, max_length=32_000)


class SendPromptOutput(BaseModel):
    response: str
    tokens_used: int
    context_pct: float
    transfer_triggered: bool
    transfer_block_pointer: str | None = None


# ── get_context_usage ───────────────────────────────────────────────────


class GetContextUsageInput(BaseModel):
    session_id: str


class GetContextUsageOutput(BaseModel):
    tokens_used: int
    context_pct: float
    limit: int
    transfer_threshold: float
    blocks_stored: int


# ── log_correction ──────────────────────────────────────────────────────


Severity = Literal["low", "medium", "high", "critical"]
Category = Literal["factual", "format", "tone", "logic"]


class LogCorrectionInput(BaseModel):
    session_id: str
    original: str = Field(min_length=1, max_length=32_000)
    correction: str = Field(min_length=1, max_length=32_000)
    severity: Severity = "low"
    category: Category | None = None


class LogCorrectionOutput(BaseModel):
    correction_id: str
    response: str
    tx_hash: str | None = None
    chain_id: int | None = None
    explorer_url: str | None = None


# ── transfer_context ────────────────────────────────────────────────────


class TransferContextInput(BaseModel):
    session_id: str


class TransferContextOutput(BaseModel):
    block_pointer: str
    tokens_transferred: int
    summary: str
    block_index: int
    tx_hash: str | None = None
    explorer_url: str | None = None


# ── get_session_state ───────────────────────────────────────────────────


class GetSessionStateInput(BaseModel):
    session_id: str


class GetSessionStateOutput(BaseModel):
    session_id: str
    user_id: str
    model_id: str
    tokens_used: int
    context_pct: float
    block_pointer: str | None
    correction_count: int
    status: str
    created_at: str
    updated_at: str
    blocks_stored: int
    outstanding_invoices_usd: float


# ── query_codebase (Greptile) ───────────────────────────────────────────


class QueryCodebaseInput(BaseModel):
    session_id: str
    question: str = Field(min_length=1, max_length=8000)
    repo_remote: str | None = Field(
        default=None,
        description="Greptile remote, e.g. 'github'. Required on first call for a session.",
    )
    repository: str | None = Field(
        default=None,
        description="`owner/name`. Required on first call for a session.",
    )
    branch: str = Field(default="main", max_length=128)


class QueryCodebaseOutput(BaseModel):
    answer: str
    sources: list[dict]
    repo_id: str
    indexed: bool


# ── create_invoice (AllScale) ───────────────────────────────────────────


class CreateInvoiceInput(BaseModel):
    session_id: str
    stablecoin: str | None = Field(
        default=None, description="USDC, USDT, etc. Defaults to env config."
    )


class CreateInvoiceOutput(BaseModel):
    invoice_id: str
    invoice_ref: str | None
    payment_url: str | None
    usd_amount: float
    stablecoin: str
    tokens_billed: int
