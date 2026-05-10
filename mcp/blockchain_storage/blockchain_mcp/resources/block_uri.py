from __future__ import annotations

from blockchain_mcp.store.chain_store import ChainStore


def build_block_payload(store: ChainStore, block_id: str) -> dict:
    blk = store.get_block(block_id)
    if blk is None:
        raise ValueError(f"Unknown block_id: {block_id}")
    return {
        "block_id": blk.block_id,
        "content": blk.content,
        "token_count": blk.token_count,
        "validation_status": blk.validation_status,
        "committed_at": blk.committed_at,
        "issues": blk.issues,
        "cleaned_content": blk.cleaned_content,
    }
