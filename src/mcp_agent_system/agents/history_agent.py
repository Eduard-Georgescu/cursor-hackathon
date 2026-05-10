"""HistoryAgent: silent background recorder.

Receives full context windows from the ActiveAgent at transfer time,
generates a 3-5 sentence summary using the cheap CLōD model, and
persists a context_blocks row. Optionally records an on-chain proof
through the BGA ChainClient.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field

from ..db.repositories.context_block_repo import ContextBlockRepo
from ..db.session import session_scope
from ..integrations.chain import ChainClient
from ..integrations.clod import ClodClient


@dataclass
class StoredBlock:
    block_id: str
    block_index: int
    summary: str
    tokens_transferred: int
    tx_hash: str | None
    explorer_url: str | None


@dataclass
class HistoryAgent:
    session_id: str
    clod: ClodClient
    chain: ChainClient | None = None
    block_index: int = 0
    last_summary: str | None = field(default=None)

    async def store_context_block(
        self, messages: list[dict[str, str]], tokens_transferred: int
    ) -> StoredBlock:
        completion = await self.clod.summarize(messages)
        summary = completion.text or "(empty summary)"
        block_id = str(uuid.uuid4())
        self.block_index += 1
        block_index = self.block_index

        chain_tx = None
        if self.chain is not None:
            chain_tx = self.chain.record_context_block(
                session_id=self.session_id, block_id=block_id, summary=summary
            )

        with session_scope() as db:
            ContextBlockRepo.create(
                db,
                block_id=block_id,
                session_id=self.session_id,
                block_index=block_index,
                tokens_transferred=tokens_transferred,
                summary=summary,
                raw_context=json.dumps(messages),
                tx_hash=chain_tx.tx_hash if chain_tx else None,
            )

        self.last_summary = summary
        return StoredBlock(
            block_id=block_id,
            block_index=block_index,
            summary=summary,
            tokens_transferred=tokens_transferred,
            tx_hash=chain_tx.tx_hash if chain_tx else None,
            explorer_url=chain_tx.explorer_url if chain_tx else None,
        )

    def hydrate_block_index(self, max_existing: int) -> None:
        """Restore ``block_index`` after a server restart."""
        self.block_index = max(self.block_index, max_existing)
