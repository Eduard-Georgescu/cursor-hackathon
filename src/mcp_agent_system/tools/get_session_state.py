"""``get_session_state`` tool handler."""

from __future__ import annotations

from ..db.repositories.billing_repo import BillingRepo
from ..db.repositories.context_block_repo import ContextBlockRepo
from ..db.repositories.session_repo import SessionRepo
from ..db.session import session_scope
from ..schemas.tools import GetSessionStateInput, GetSessionStateOutput


class SessionNotFoundError(ValueError):
    pass


async def handle_get_session_state(
    payload: GetSessionStateInput,
) -> GetSessionStateOutput:
    with session_scope() as db:
        row = SessionRepo.find(db, payload.session_id)
        if row is None:
            raise SessionNotFoundError(f"Session not found: {payload.session_id}")
        blocks_stored = ContextBlockRepo.count_for_session(db, row.session_id)
        outstanding = BillingRepo.outstanding_usd(db, row.session_id)
        return GetSessionStateOutput(
            session_id=row.session_id,
            user_id=row.user_id,
            model_id=row.model_id,
            tokens_used=row.tokens_used,
            context_pct=row.context_pct,
            block_pointer=row.block_pointer,
            correction_count=row.correction_count,
            status=row.status,
            created_at=row.created_at,
            updated_at=row.updated_at,
            blocks_stored=blocks_stored,
            outstanding_invoices_usd=outstanding,
        )
