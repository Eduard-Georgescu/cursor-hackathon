from __future__ import annotations

from blockchain_mcp.store.chain_store import ChainStore


def run_get_chain_stats(store: ChainStore, *, chain_id: str) -> dict:
    stats = store.get_chain_stats(chain_id)
    return {
        "block_count": stats["block_count"],
        "total_tokens": stats["total_tokens"],
        "quota_remaining": stats["quota_remaining"],
        "pruning_enabled": stats["pruning_enabled"],
    }
