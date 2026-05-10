"""``start_session`` tool handler."""

from __future__ import annotations

import uuid

from ..agents.orchestrator import AgentOrchestrator
from ..db.repositories.session_repo import SessionRepo
from ..db.session import session_scope
from ..schemas.tools import (
    AgentConfigOut,
    StartSessionInput,
    StartSessionOutput,
)
from ..utils.env import get_settings
from ..utils.token_counter import get_context_limit


async def handle_start_session(
    payload: StartSessionInput, orchestrator: AgentOrchestrator
) -> StartSessionOutput:
    settings = get_settings()
    session_id = str(uuid.uuid4())
    context_limit = get_context_limit(payload.model, default=settings.context_limit_tokens)
    transfer_threshold = settings.context_transfer_threshold

    with session_scope() as db:
        SessionRepo.create(
            db,
            session_id=session_id,
            user_id=payload.user_id,
            model_id=payload.model,
        )

    orchestrator.create_session(
        session_id=session_id,
        model_id=payload.model,
        system_prompt=payload.system_prompt,
        context_limit=context_limit,
        transfer_threshold=transfer_threshold,
    )

    return StartSessionOutput(
        session_id=session_id,
        agent_config=AgentConfigOut(
            model_id=payload.model,
            context_limit=context_limit,
            transfer_threshold=transfer_threshold,
        ),
    )
