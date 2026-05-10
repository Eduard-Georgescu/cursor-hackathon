"""``get_context_usage`` tool handler."""

from __future__ import annotations

from ..agents.orchestrator import AgentOrchestrator
from ..db.repositories.context_block_repo import ContextBlockRepo
from ..db.repositories.session_repo import SessionRepo
from ..db.session import session_scope
from ..schemas.tools import GetContextUsageInput, GetContextUsageOutput
from ..utils.token_counter import get_context_limit


async def handle_get_context_usage(
    payload: GetContextUsageInput, orchestrator: AgentOrchestrator
) -> GetContextUsageOutput:
    with session_scope() as db:
        row = SessionRepo.find(db, payload.session_id)
        if row is None:
            raise ValueError(f"Session not found: {payload.session_id}")
        blocks_stored = ContextBlockRepo.count_for_session(db, payload.session_id)

    active = orchestrator.get_active(payload.session_id)
    limit = active.context_limit if active else get_context_limit(row.model_id)
    threshold = active.transfer_threshold if active else 0.80

    return GetContextUsageOutput(
        tokens_used=row.tokens_used,
        context_pct=row.context_pct,
        limit=limit,
        transfer_threshold=threshold,
        blocks_stored=blocks_stored,
    )
