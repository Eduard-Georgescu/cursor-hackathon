from __future__ import annotations

from blockchain_mcp.models import MAX_BLOCK_TOKEN_CAPACITY
from blockchain_mcp.store.chain_store import ChainStore


def build_current_payload(store: ChainStore, chain_id: str) -> dict:
    blk = store.current_open_block(chain_id)
    return {
        "chain_id": chain_id,
        "block_id": blk.block_id,
        "content": blk.content,
        "token_count": blk.token_count,
        "tokens_remaining": max(0, MAX_BLOCK_TOKEN_CAPACITY - blk.token_count),
        "validation_status": blk.validation_status,
        "issues": blk.issues,
        "cleaned_content_preview": blk.cleaned_content,
    }
