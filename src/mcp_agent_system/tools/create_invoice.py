"""``create_invoice`` tool handler (AllScale stablecoin billing).

Reads ``tokens_used`` from the session, converts to USD via the configured
``ALLSCALE_USD_PER_1K_TOKENS`` rate, and submits the invoice to AllScale.
The returned ``payment_url`` lets the end user pay in stablecoins; the
result is persisted in the ``billing`` table.
"""

from __future__ import annotations

import uuid

from ..agents.orchestrator import AgentOrchestrator
from ..db.repositories.billing_repo import BillingRepo
from ..db.repositories.session_repo import SessionRepo
from ..db.session import session_scope
from ..integrations.allscale import AllScaleClient, stablecoin_name
from ..schemas.tools import CreateInvoiceInput, CreateInvoiceOutput


async def handle_create_invoice(
    payload: CreateInvoiceInput,
    orchestrator: AgentOrchestrator,
    allscale: AllScaleClient | None = None,
) -> CreateInvoiceOutput:
    with session_scope() as db:
        row = SessionRepo.find(db, payload.session_id)
        if row is None:
            raise ValueError(f"Session not found: {payload.session_id}")
        tokens_billed = row.tokens_used
        user_id = row.user_id

    client = allscale or AllScaleClient()
    invoice = await client.create_invoice(
        session_id=payload.session_id,
        tokens_billed=tokens_billed,
        customer_ref=user_id,
        stablecoin=payload.stablecoin,
    )
    coin_name = stablecoin_name(invoice.stablecoin)

    invoice_id = str(uuid.uuid4())
    with session_scope() as db:
        BillingRepo.create(
            db,
            invoice_id=invoice_id,
            session_id=payload.session_id,
            tokens_billed=tokens_billed,
            usd_amount=invoice.usd_amount,
            stablecoin=coin_name,
            allscale_invoice_ref=invoice.invoice_ref or None,
            payment_url=invoice.payment_url,
            status="sent" if invoice.invoice_ref else "draft",
        )

    return CreateInvoiceOutput(
        invoice_id=invoice_id,
        invoice_ref=invoice.invoice_ref or None,
        payment_url=invoice.payment_url,
        usd_amount=invoice.usd_amount,
        stablecoin=coin_name,
        tokens_billed=tokens_billed,
    )
