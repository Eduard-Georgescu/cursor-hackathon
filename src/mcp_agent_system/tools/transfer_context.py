"""``transfer_context`` tool handler.

Implements the context handoff protocol from doc §5.5:

    1. Mark the session as ``transferring`` (prevents concurrent transfers).
    2. Capture the active agent's full context.
    3. Persist a context_blocks row + LLM-generated summary via the History
       Agent. Optionally write an on-chain proof (BGA).
    4. Reset the active agent's window and inject the summary.
    5. Update the session row's block_pointer / tokens_used / status='active'.

If anything fails between steps 2-5, status is rolled back to 'active' so
the session is not permanently stuck (doc §9.3).
"""

from __future__ import annotations

from ..agents.orchestrator import AgentOrchestrator
from ..db.repositories.session_repo import SessionRepo
from ..db.session import session_scope
from ..schemas.tools import TransferContextInput, TransferContextOutput
from ..utils.token_counter import estimate_tokens_from_text


async def handle_transfer_context(
    payload: TransferContextInput, orchestrator: AgentOrchestrator
) -> TransferContextOutput:
    pair = orchestrator.get_pair(payload.session_id)
    if pair is None:
        raise ValueError(f"No active session: {payload.session_id}")
    active = pair.active
    history = pair.history

    with session_scope() as db:
        SessionRepo.set_status(db, payload.session_id, "transferring")

    try:
        full_context = active.get_full_context()
        tokens_transferred = active.token_count

        stored = await history.store_context_block(full_context, tokens_transferred)

        active.reset_context()
        active.inject_context_summary(
            summary=stored.summary,
            block_pointer=stored.block_id,
            block_index=stored.block_index,
            tokens_transferred=tokens_transferred,
        )

        summary_tokens = estimate_tokens_from_text(stored.summary)
        context_pct = min(summary_tokens / max(active.context_limit, 1), 1.0)

        with session_scope() as db:
            SessionRepo.update_after_transfer(
                db,
                payload.session_id,
                block_pointer=stored.block_id,
                tokens_used=summary_tokens,
                context_pct=context_pct,
            )

        return TransferContextOutput(
            block_pointer=stored.block_id,
            tokens_transferred=tokens_transferred,
            summary=stored.summary,
            block_index=stored.block_index,
            tx_hash=stored.tx_hash,
            explorer_url=stored.explorer_url,
        )
    except Exception:
        with session_scope() as db:
            SessionRepo.set_status(db, payload.session_id, "active")
        raise
