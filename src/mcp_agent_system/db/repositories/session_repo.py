"""Repository for the ``sessions`` table."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select, update
from sqlalchemy.orm import Session as OrmSession

from ..models import Session


def _now() -> str:
    return datetime.now(UTC).isoformat()


class SessionRepo:
    @staticmethod
    def create(
        db: OrmSession,
        *,
        session_id: str,
        user_id: str,
        model_id: str,
    ) -> Session:
        now = _now()
        row = Session(
            session_id=session_id,
            user_id=user_id,
            model_id=model_id,
            tokens_used=0,
            context_pct=0.0,
            block_pointer=None,
            correction_count=0,
            status="active",
            created_at=now,
            updated_at=now,
        )
        db.add(row)
        db.flush()
        return row

    @staticmethod
    def find(db: OrmSession, session_id: str) -> Session | None:
        return db.get(Session, session_id)

    @staticmethod
    def list_active(db: OrmSession) -> list[Session]:
        return list(db.execute(select(Session).where(Session.status == "active")).scalars())

    @staticmethod
    def add_tokens(db: OrmSession, session_id: str, delta: int, context_limit: int) -> Session:
        row = db.get(Session, session_id)
        if row is None:
            raise ValueError(f"Session not found: {session_id}")
        row.tokens_used = (row.tokens_used or 0) + max(0, delta)
        row.context_pct = min(row.tokens_used / context_limit, 1.0)
        row.updated_at = _now()
        db.flush()
        return row

    @staticmethod
    def increment_correction_count(db: OrmSession, session_id: str) -> int:
        row = db.get(Session, session_id)
        if row is None:
            raise ValueError(f"Session not found: {session_id}")
        row.correction_count += 1
        row.updated_at = _now()
        db.flush()
        return row.correction_count

    @staticmethod
    def set_status(db: OrmSession, session_id: str, status: str) -> None:
        db.execute(
            update(Session)
            .where(Session.session_id == session_id)
            .values(status=status, updated_at=_now())
        )

    @staticmethod
    def update_after_transfer(
        db: OrmSession,
        session_id: str,
        *,
        block_pointer: str,
        tokens_used: int,
        context_pct: float,
    ) -> None:
        db.execute(
            update(Session)
            .where(Session.session_id == session_id)
            .values(
                block_pointer=block_pointer,
                tokens_used=tokens_used,
                context_pct=context_pct,
                status="active",
                updated_at=_now(),
            )
        )
