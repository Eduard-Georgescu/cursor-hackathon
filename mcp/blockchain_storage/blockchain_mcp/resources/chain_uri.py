from __future__ import annotations

from blockchain_mcp.store.chain_store import ChainStore


def build_chain_payload(store: ChainStore, chain_id: str) -> dict:
    chain = store.get_chain(chain_id)
    stats = store.get_chain_stats(chain_id)
    open_b = chain.open_block
    open_preview = (
        {
            "block_id": open_b.block_id,
            "token_count": open_b.token_count,
            "validation_status": open_b.validation_status,
        }
        if open_b
        else None
    )
    return {
        "chain_id": chain.chain_id,
        "committed_block_count": len(chain.committed_blocks),
        "total_committed_tokens": sum(b.token_count for b in chain.committed_blocks),
        "max_committed_blocks": chain.max_committed_blocks,
        "subscription_block_quota": chain.subscription_block_quota,
        "pruning_enabled": chain.pruning_enabled,
        "stats": stats,
        "open_block_summary": open_preview,
    }
