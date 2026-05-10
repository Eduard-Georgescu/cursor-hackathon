from __future__ import annotations

from blockchain_mcp.store.chain_store import ChainStore
from blockchain_mcp.validation_stub import run_stub_validation


def run_validate_block(store: ChainStore, *, block_id: str) -> dict:
    blk = store.get_block(block_id)
    if blk is None:
        raise ValueError(f"Unknown block_id: {block_id}")
    if blk.committed_at is not None:
        return {
            "valid": False,
            "issues": ["error:block_already_committed"],
            "cleaned_content": blk.content,
        }
    valid, issues, cleaned = run_stub_validation(blk.content)
    store.apply_validation_to_block(
        block_id,
        valid=valid,
        issues=issues,
        cleaned_content=cleaned,
    )
    return {"valid": valid, "issues": issues, "cleaned_content": cleaned}
