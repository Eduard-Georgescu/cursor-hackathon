from __future__ import annotations

from blockchain_mcp.store.chain_store import ChainStore


def run_cancel_block(store: ChainStore, *, block_id: str, reason: str) -> dict:
    success, chain_relinked = store.cancel_block(block_id, reason)
    return {"success": success, "chain_relinked": chain_relinked}
