from __future__ import annotations

from blockchain_mcp.store.chain_store import ChainStore


def run_append_context(
    store: ChainStore,
    *,
    chain_id: str,
    content: str,
    token_count: int | None = None,
) -> dict:
    block_id, tokens_remaining = store.append_context(
        chain_id, content, token_count
    )
    return {"block_id": block_id, "tokens_remaining": tokens_remaining}
