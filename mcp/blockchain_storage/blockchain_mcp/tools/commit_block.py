from __future__ import annotations

from blockchain_mcp.store.chain_store import ChainStore


def run_commit_block(store: ChainStore, *, block_id: str) -> dict:
    committed_block_id, next_block_id, pruned_block_id = store.commit_block(
        block_id
    )
    return {
        "committed_block_id": committed_block_id,
        "next_block_id": next_block_id,
        "pruned_block_id": pruned_block_id,
    }
