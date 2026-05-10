"""``log_correction`` tool handler.

Records a user-supplied correction:
    1. Active Agent acknowledges the correction (one-shot system call).
    2. Persists a corrections row + increments session.correction_count.
    3. Writes an on-chain proof to BGA's CorrectionRegistry contract
       (graceful no-op if disabled). The returned tx_hash is stored on
       the row so the public training-data commons is auditable.
"""

from __future__ import annotations

import uuid

from ..agents.orchestrator import AgentOrchestrator
from ..db.repositories.correction_repo import CorrectionRepo
from ..db.repositories.session_repo import SessionRepo
from ..db.session import session_scope
from ..integrations.chain import ChainClient
from ..schemas.tools import LogCorrectionInput, LogCorrectionOutput
from ..utils.prompts import CORRECTION_ACK_TEMPLATE


async def handle_log_correction(
    payload: LogCorrectionInput,
    orchestrator: AgentOrchestrator,
    chain: ChainClient | None = None,
) -> LogCorrectionOutput:
    active = orchestrator.get_active(payload.session_id)
    if active is None:
        raise ValueError(f"No active session: {payload.session_id}")

    ack_prompt = CORRECTION_ACK_TEMPLATE.format(
        original=payload.original, correction=payload.correction
    )
    ack = await active.send_system(ack_prompt)

    correction_id = str(uuid.uuid4())
    chain_client = chain or orchestrator.chain
    chain_tx = chain_client.record_correction(
        session_id=payload.session_id,
        original=payload.original,
        correction=payload.correction,
    )

    with session_scope() as db:
        CorrectionRepo.create(
            db,
            correction_id=correction_id,
            session_id=payload.session_id,
            original=payload.original,
            correction=payload.correction,
            response=ack.response,
            severity=payload.severity,
            category=payload.category,
            tx_hash=chain_tx.tx_hash if chain_tx else None,
            chain_id=chain_tx.chain_id if chain_tx else None,
        )
        SessionRepo.increment_correction_count(db, payload.session_id)

    return LogCorrectionOutput(
        correction_id=correction_id,
        response=ack.response,
        tx_hash=chain_tx.tx_hash if chain_tx else None,
        chain_id=chain_tx.chain_id if chain_tx else None,
        explorer_url=chain_tx.explorer_url if chain_tx else None,
    )
