"""``agent://session/{id}`` resource — live snapshot of a session."""

from __future__ import annotations

from datetime import UTC, datetime

from ..db.repositories.billing_repo import BillingRepo
from ..db.repositories.context_block_repo import ContextBlockRepo
from ..db.repositories.session_repo import SessionRepo
from ..db.session import session_scope


def render_session_payload(session_id: str) -> dict:
    with session_scope() as db:
        row = SessionRepo.find(db, session_id)
        if row is None:
            return {"error": "Session not found", "session_id": session_id}
        blocks_stored = ContextBlockRepo.count_for_session(db, session_id)
        outstanding = BillingRepo.outstanding_usd(db, session_id)
        return {
            "session_id": row.session_id,
            "user_id": row.user_id,
            "model_id": row.model_id,
            "tokens_used": row.tokens_used,
            "context_pct": row.context_pct,
            "block_pointer": row.block_pointer,
            "correction_count": row.correction_count,
            "status": row.status,
            "created_at": row.created_at,
            "updated_at": row.updated_at,
            "blocks_stored": blocks_stored,
            "outstanding_invoices_usd": outstanding,
            "fetched_at": datetime.now(UTC).isoformat(),
        }
