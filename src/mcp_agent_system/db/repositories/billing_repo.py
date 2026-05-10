"""Repository for the ``billing`` table (AllScale invoices)."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session as OrmSession

from ..models import Billing


class BillingRepo:
    @staticmethod
    def create(
        db: OrmSession,
        *,
        invoice_id: str,
        session_id: str,
        tokens_billed: int,
        usd_amount: float,
        stablecoin: str,
        allscale_invoice_ref: str | None,
        payment_url: str | None,
        status: str = "draft",
    ) -> Billing:
        row = Billing(
            invoice_id=invoice_id,
            session_id=session_id,
            tokens_billed=tokens_billed,
            usd_amount=usd_amount,
            stablecoin=stablecoin,
            allscale_invoice_ref=allscale_invoice_ref,
            payment_url=payment_url,
            status=status,
            created_at=datetime.now(UTC).isoformat(),
        )
        db.add(row)
        db.flush()
        return row

    @staticmethod
    def list_for_session(db: OrmSession, session_id: str) -> list[Billing]:
        return list(
            db.execute(
                select(Billing).where(Billing.session_id == session_id)
            ).scalars()
        )

    @staticmethod
    def outstanding_usd(db: OrmSession, session_id: str) -> float:
        result = db.execute(
            select(func.coalesce(func.sum(Billing.usd_amount), 0.0))
            .where(Billing.session_id == session_id)
            .where(Billing.status.in_(("draft", "sent")))
        ).scalar_one()
        return float(result or 0.0)
