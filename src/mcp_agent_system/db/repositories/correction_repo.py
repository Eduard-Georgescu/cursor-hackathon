"""Repository for the ``corrections`` table."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session as OrmSession

from ..models import Correction


class CorrectionRepo:
    @staticmethod
    def create(
        db: OrmSession,
        *,
        correction_id: str,
        session_id: str,
        original: str,
        correction: str,
        response: str | None,
        severity: str,
        category: str | None,
        tx_hash: str | None = None,
        chain_id: int | None = None,
    ) -> Correction:
        row = Correction(
            correction_id=correction_id,
            session_id=session_id,
            original=original,
            correction=correction,
            response=response,
            severity=severity,
            category=category,
            timestamp=datetime.now(UTC).isoformat(),
            tx_hash=tx_hash,
            chain_id=chain_id,
        )
        db.add(row)
        db.flush()
        return row

    @staticmethod
    def attach_tx_hash(
        db: OrmSession, correction_id: str, *, tx_hash: str, chain_id: int
    ) -> None:
        row = db.get(Correction, correction_id)
        if row is not None:
            row.tx_hash = tx_hash
            row.chain_id = chain_id
            db.flush()

    @staticmethod
    def find_by_session(db: OrmSession, session_id: str) -> list[Correction]:
        return list(
            db.execute(
                select(Correction)
                .where(Correction.session_id == session_id)
                .order_by(Correction.timestamp.asc())
            ).scalars()
        )
