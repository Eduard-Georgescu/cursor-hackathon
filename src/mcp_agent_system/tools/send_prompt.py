"""``send_prompt`` tool handler.

Routes the user message through the Active Agent (CLōD-backed), updates
session token totals, and auto-fires ``transfer_context`` when the
threshold is crossed.
"""

from __future__ import annotations

from ..agents.orchestrator import AgentOrchestrator
from ..db.repositories.session_repo import SessionRepo
from ..db.session import session_scope
from ..schemas.tools import (
    SendPromptInput,
    SendPromptOutput,
    TransferContextInput,
)
from .transfer_context import handle_transfer_context


async def handle_send_prompt(
    payload: SendPromptInput, orchestrator: AgentOrchestrator
) -> SendPromptOutput:
    active = orchestrator.get_active(payload.session_id)
    if active is None:
        raise ValueError(f"No active session: {payload.session_id}")

    result = await active.send_message(payload.message)
    delta = result.usage.total

    with session_scope() as db:
        row = SessionRepo.add_tokens(
            db, payload.session_id, delta, active.context_limit
        )
        tokens_used = row.tokens_used
        context_pct = row.context_pct

    transfer_triggered = False
    transfer_block_pointer: str | None = None
    if context_pct >= active.transfer_threshold:
        transfer_out = await handle_transfer_context(
            TransferContextInput(session_id=payload.session_id), orchestrator
        )
        transfer_triggered = True
        transfer_block_pointer = transfer_out.block_pointer
        with session_scope() as db:
            row = SessionRepo.find(db, payload.session_id)
            tokens_used = row.tokens_used if row else tokens_used
            context_pct = row.context_pct if row else context_pct

    return SendPromptOutput(
        response=result.response,
        tokens_used=tokens_used,
        context_pct=context_pct,
        transfer_triggered=transfer_triggered,
        transfer_block_pointer=transfer_block_pointer,
    )
