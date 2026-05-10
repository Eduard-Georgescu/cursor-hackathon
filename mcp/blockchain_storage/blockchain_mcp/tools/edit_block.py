from __future__ import annotations

from typing import Any

from blockchain_mcp.store.chain_store import ChainStore


def run_edit_block(store: ChainStore, *, block_id: str, edits: dict[str, Any]) -> dict:
    updated_block_id, diff = store.edit_block(block_id, edits)
    return {"updated_block_id": updated_block_id, "diff": diff}
