"""Repository for the ``context_blocks`` table."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session as OrmSession

from ..models import ContextBlock


class ContextBlockRepo:
    @staticmethod
    def create(
        db: OrmSession,
        *,
        block_id: str,
        session_id: str,
        block_index: int,
        tokens_transferred: int,
        summary: str,
        raw_context: str,
        tx_hash: str | None = None,
    ) -> ContextBlock:
        row = ContextBlock(
            block_id=block_id,
            session_id=session_id,
            block_index=block_index,
            tokens_transferred=tokens_transferred,
            summary=summary,
            raw_context=raw_context,
            created_at=datetime.now(UTC).isoformat(),
            tx_hash=tx_hash,
        )
        db.add(row)
        db.flush()
        return row

    @staticmethod
    def count_for_session(db: OrmSession, session_id: str) -> int:
        result = db.execute(
            select(func.count())
            .select_from(ContextBlock)
            .where(ContextBlock.session_id == session_id)
        ).scalar_one()
        return int(result or 0)

    @staticmethod
    def latest_for_session(db: OrmSession, session_id: str) -> ContextBlock | None:
        return db.execute(
            select(ContextBlock)
            .where(ContextBlock.session_id == session_id)
            .order_by(desc(ContextBlock.block_index))
            .limit(1)
        ).scalar_one_or_none()

    @staticmethod
    def max_block_index(db: OrmSession, session_id: str) -> int:
        result = db.execute(
            select(func.max(ContextBlock.block_index)).where(
                ContextBlock.session_id == session_id
            )
        ).scalar_one_or_none()
        return int(result or 0)
