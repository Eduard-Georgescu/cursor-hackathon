"""SQLAlchemy 2.0 declarative models for the MCP Agent System.

Five tables:
    sessions       - one row per agent session (doc Table 7)
    corrections    - user-supplied corrections, with on-chain proofs (doc Table 8 + BGA)
    context_blocks - archived context windows from transfer_context (doc Table 9)
    billing        - AllScale stablecoin invoices for token usage
    codebase_links - Greptile-indexed repos linked to a session
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat()


class Base(DeclarativeBase):
    pass


class Session(Base):
    __tablename__ = "sessions"

    session_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    model_id: Mapped[str] = mapped_column(String(128), nullable=False)
    tokens_used: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    context_pct: Mapped[float] = mapped_column(nullable=False, default=0.0)
    block_pointer: Mapped[str | None] = mapped_column(String(36), nullable=True)
    correction_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active")
    created_at: Mapped[str] = mapped_column(String(40), nullable=False, default=_utcnow_iso)
    updated_at: Mapped[str] = mapped_column(String(40), nullable=False, default=_utcnow_iso)

    corrections: Mapped[list[Correction]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )
    blocks: Mapped[list[ContextBlock]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )
    invoices: Mapped[list[Billing]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )
    codebase_links: Mapped[list[CodebaseLink]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )


class Correction(Base):
    __tablename__ = "corrections"

    correction_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("sessions.session_id"), nullable=False, index=True
    )
    original: Mapped[str] = mapped_column(Text, nullable=False)
    correction: Mapped[str] = mapped_column(Text, nullable=False)
    response: Mapped[str | None] = mapped_column(Text, nullable=True)
    severity: Mapped[str] = mapped_column(String(16), nullable=False, default="low")
    category: Mapped[str | None] = mapped_column(String(64), nullable=True)
    timestamp: Mapped[str] = mapped_column(String(40), nullable=False, default=_utcnow_iso)
    # BGA on-chain proof
    tx_hash: Mapped[str | None] = mapped_column(String(80), nullable=True)
    chain_id: Mapped[int | None] = mapped_column(Integer, nullable=True)

    session: Mapped[Session] = relationship(back_populates="corrections")


class ContextBlock(Base):
    __tablename__ = "context_blocks"

    block_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("sessions.session_id"), nullable=False, index=True
    )
    block_index: Mapped[int] = mapped_column(Integer, nullable=False)
    tokens_transferred: Mapped[int] = mapped_column(Integer, nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    raw_context: Mapped[str] = mapped_column(Text, nullable=False)  # JSON-serialized list
    created_at: Mapped[str] = mapped_column(String(40), nullable=False, default=_utcnow_iso)
    # BGA optional proof for the context block
    tx_hash: Mapped[str | None] = mapped_column(String(80), nullable=True)

    session: Mapped[Session] = relationship(back_populates="blocks")

    __table_args__ = (
        Index("ix_context_blocks_session_index", "session_id", "block_index"),
    )


class Billing(Base):
    """AllScale invoice records for token consumption."""

    __tablename__ = "billing"

    invoice_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("sessions.session_id"), nullable=False, index=True
    )
    tokens_billed: Mapped[int] = mapped_column(Integer, nullable=False)
    usd_amount: Mapped[float] = mapped_column(nullable=False)
    stablecoin: Mapped[str] = mapped_column(String(16), nullable=False, default="USDC")
    allscale_invoice_ref: Mapped[str | None] = mapped_column(String(128), nullable=True)
    payment_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="draft")
    created_at: Mapped[str] = mapped_column(String(40), nullable=False, default=_utcnow_iso)

    session: Mapped[Session] = relationship(back_populates="invoices")


class CodebaseLink(Base):
    """Maps a session to a Greptile-indexed repository."""

    __tablename__ = "codebase_links"

    link_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("sessions.session_id"), nullable=False, index=True
    )
    repo_remote: Mapped[str] = mapped_column(String(256), nullable=False)
    repo_branch: Mapped[str] = mapped_column(String(128), nullable=False, default="main")
    greptile_repo_id: Mapped[str] = mapped_column(String(256), nullable=False)
    last_query: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    linked_at: Mapped[str] = mapped_column(String(40), nullable=False, default=_utcnow_iso)

    session: Mapped[Session] = relationship(back_populates="codebase_links")
