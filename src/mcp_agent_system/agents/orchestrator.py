"""AgentOrchestrator: in-memory map of session_id -> (ActiveAgent, HistoryAgent)."""

from __future__ import annotations

import json
from dataclasses import dataclass

from ..db.repositories.context_block_repo import ContextBlockRepo
from ..db.repositories.session_repo import SessionRepo
from ..db.session import session_scope
from ..integrations.chain import ChainClient
from ..integrations.clod import ClodClient
from ..utils.logger import get_logger
from ..utils.prompts import DEFAULT_SYSTEM_PROMPT
from ..utils.token_counter import get_context_limit
from .active_agent import ActiveAgent, ActiveAgentConfig
from .history_agent import HistoryAgent

log = get_logger(__name__)


@dataclass
class SessionPair:
    active: ActiveAgent
    history: HistoryAgent


class AgentOrchestrator:
    """Holds the in-memory map of all active session pairs.

    Instantiated once at server startup and shared by every tool handler.
    Sponsor clients (CLōD, BGA chain) are injected so tests can swap them
    out for mocks.
    """

    def __init__(
        self,
        *,
        clod: ClodClient | None = None,
        chain: ChainClient | None = None,
    ) -> None:
        self.clod = clod or ClodClient()
        self.chain = chain or ChainClient()
        self._sessions: dict[str, SessionPair] = {}

    @property
    def active_sessions(self) -> int:
        return len(self._sessions)

    def create_session(
        self,
        *,
        session_id: str,
        model_id: str,
        system_prompt: str | None = None,
        transfer_threshold: float | None = None,
        context_limit: int | None = None,
    ) -> SessionPair:
        cfg = ActiveAgentConfig(
            session_id=session_id,
            model_id=model_id,
            system_prompt=system_prompt or DEFAULT_SYSTEM_PROMPT,
            context_limit=context_limit or get_context_limit(model_id),
            transfer_threshold=transfer_threshold or 0.80,
        )
        active = ActiveAgent(cfg=cfg, clod=self.clod)
        history = HistoryAgent(session_id=session_id, clod=self.clod, chain=self.chain)
        pair = SessionPair(active=active, history=history)
        self._sessions[session_id] = pair
        return pair

    def get_active(self, session_id: str) -> ActiveAgent | None:
        pair = self._sessions.get(session_id)
        return pair.active if pair else None

    def get_history(self, session_id: str) -> HistoryAgent | None:
        pair = self._sessions.get(session_id)
        return pair.history if pair else None

    def get_pair(self, session_id: str) -> SessionPair | None:
        return self._sessions.get(session_id)

    def destroy_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def rehydrate_active_sessions(
        self, *, transfer_threshold: float = 0.80
    ) -> list[str]:
        """Re-create in-memory pairs for sessions that were active before a restart.

        Returns the list of session_ids that were rehydrated.
        """
        recovered: list[str] = []
        with session_scope() as db:
            rows = SessionRepo.list_active(db)
            for row in rows:
                if row.session_id in self._sessions:
                    continue
                pair = self.create_session(
                    session_id=row.session_id,
                    model_id=row.model_id,
                    transfer_threshold=transfer_threshold,
                )
                latest = ContextBlockRepo.latest_for_session(db, row.session_id)
                if latest is not None:
                    try:
                        msgs = json.loads(latest.raw_context)
                    except json.JSONDecodeError:
                        msgs = []
                    if isinstance(msgs, list):
                        pair.active.messages = msgs
                        pair.active.token_count = row.tokens_used
                    pair.history.hydrate_block_index(latest.block_index)
                recovered.append(row.session_id)
        if recovered:
            log.info("orchestrator.rehydrated", count=len(recovered))
        return recovered
