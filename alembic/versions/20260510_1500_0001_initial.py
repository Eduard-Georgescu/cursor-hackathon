"""initial schema (sessions, corrections, context_blocks, billing, codebase_links)

Revision ID: 0001_initial
Revises:
Create Date: 2026-05-10 15:00:00
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.create_table(
        "sessions",
        sa.Column("session_id", sa.String(length=36), primary_key=True),
        sa.Column("user_id", sa.String(length=256), nullable=False),
        sa.Column("model_id", sa.String(length=128), nullable=False),
        sa.Column("tokens_used", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("context_pct", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("block_pointer", sa.String(length=36), nullable=True),
        sa.Column("correction_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="active"),
        sa.Column("created_at", sa.String(length=40), nullable=False),
        sa.Column("updated_at", sa.String(length=40), nullable=False),
    )
    op.create_index("ix_sessions_user_id", "sessions", ["user_id"])

    op.create_table(
        "corrections",
        sa.Column("correction_id", sa.String(length=36), primary_key=True),
        sa.Column(
            "session_id",
            sa.String(length=36),
            sa.ForeignKey("sessions.session_id"),
            nullable=False,
        ),
        sa.Column("original", sa.Text(), nullable=False),
        sa.Column("correction", sa.Text(), nullable=False),
        sa.Column("response", sa.Text(), nullable=True),
        sa.Column("severity", sa.String(length=16), nullable=False, server_default="low"),
        sa.Column("category", sa.String(length=64), nullable=True),
        sa.Column("timestamp", sa.String(length=40), nullable=False),
        sa.Column("tx_hash", sa.String(length=80), nullable=True),
        sa.Column("chain_id", sa.Integer(), nullable=True),
    )
    op.create_index("ix_corrections_session_id", "corrections", ["session_id"])

    op.create_table(
        "context_blocks",
        sa.Column("block_id", sa.String(length=36), primary_key=True),
        sa.Column(
            "session_id",
            sa.String(length=36),
            sa.ForeignKey("sessions.session_id"),
            nullable=False,
        ),
        sa.Column("block_index", sa.Integer(), nullable=False),
        sa.Column("tokens_transferred", sa.Integer(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("raw_context", sa.Text(), nullable=False),
        sa.Column("created_at", sa.String(length=40), nullable=False),
        sa.Column("tx_hash", sa.String(length=80), nullable=True),
    )
    op.create_index("ix_context_blocks_session_id", "context_blocks", ["session_id"])
    op.create_index(
        "ix_context_blocks_session_index", "context_blocks", ["session_id", "block_index"]
    )

    op.create_table(
        "billing",
        sa.Column("invoice_id", sa.String(length=36), primary_key=True),
        sa.Column(
            "session_id",
            sa.String(length=36),
            sa.ForeignKey("sessions.session_id"),
            nullable=False,
        ),
        sa.Column("tokens_billed", sa.Integer(), nullable=False),
        sa.Column("usd_amount", sa.Float(), nullable=False),
        sa.Column("stablecoin", sa.String(length=16), nullable=False, server_default="USDC"),
        sa.Column("allscale_invoice_ref", sa.String(length=128), nullable=True),
        sa.Column("payment_url", sa.Text(), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="draft"),
        sa.Column("created_at", sa.String(length=40), nullable=False),
    )
    op.create_index("ix_billing_session_id", "billing", ["session_id"])

    op.create_table(
        "codebase_links",
        sa.Column("link_id", sa.String(length=36), primary_key=True),
        sa.Column(
            "session_id",
            sa.String(length=36),
            sa.ForeignKey("sessions.session_id"),
            nullable=False,
        ),
        sa.Column("repo_remote", sa.String(length=256), nullable=False),
        sa.Column(
            "repo_branch", sa.String(length=128), nullable=False, server_default="main"
        ),
        sa.Column("greptile_repo_id", sa.String(length=256), nullable=False),
        sa.Column("last_query", sa.Text(), nullable=True),
        sa.Column("last_answer", sa.Text(), nullable=True),
        sa.Column("linked_at", sa.String(length=40), nullable=False),
    )
    op.create_index("ix_codebase_links_session_id", "codebase_links", ["session_id"])


def downgrade() -> None:
    op.drop_index("ix_codebase_links_session_id", table_name="codebase_links")
    op.drop_table("codebase_links")
    op.drop_index("ix_billing_session_id", table_name="billing")
    op.drop_table("billing")
    op.drop_index("ix_context_blocks_session_index", table_name="context_blocks")
    op.drop_index("ix_context_blocks_session_id", table_name="context_blocks")
    op.drop_table("context_blocks")
    op.drop_index("ix_corrections_session_id", table_name="corrections")
    op.drop_table("corrections")
    op.drop_index("ix_sessions_user_id", table_name="sessions")
    op.drop_table("sessions")
