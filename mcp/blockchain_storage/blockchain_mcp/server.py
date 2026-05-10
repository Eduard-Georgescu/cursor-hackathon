from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ResourceError, ToolError

from blockchain_mcp.resources.block_uri import build_block_payload
from blockchain_mcp.resources.chain_uri import build_chain_payload
from blockchain_mcp.resources.current_uri import build_current_payload
from blockchain_mcp.store.chain_store import ChainStore
from blockchain_mcp.tools import (
    run_append_context,
    run_cancel_block,
    run_commit_block,
    run_edit_block,
    run_get_chain_stats,
    run_validate_block,
)

_store: ChainStore | None = None


def get_store() -> ChainStore:
    global _store
    if _store is None:
        data_dir = Path(os.environ.get("DATA_DIR", "./data")).expanduser().resolve()
        _store = ChainStore(data_dir)
    return _store


def build_mcp() -> FastMCP:
    store = get_store()
    mcp = FastMCP(
        "Blockchain Storage",
        instructions=(
            "PRMD2 blockchain-style memory: append_context, validate_block, "
            "commit_block, edit_block, cancel_block, get_chain_stats."
        ),
    )

    @mcp.tool(name="append_context")
    def append_context(
        chain_id: str,
        content: str,
        token_count: int | None = None,
    ) -> dict[str, Any]:
        """Add raw context tokens to the current open block (500K cap per block)."""
        try:
            return run_append_context(
                store,
                chain_id=chain_id,
                content=content,
                token_count=token_count,
            )
        except ValueError as e:
            raise ToolError(str(e)) from e

    @mcp.tool(name="validate_block")
    def validate_block(block_id: str) -> dict[str, Any]:
        """Stub Greptile validation on an uncommitted block."""
        try:
            return run_validate_block(store, block_id=block_id)
        except ValueError as e:
            raise ToolError(str(e)) from e

    @mcp.tool(name="commit_block")
    def commit_block(block_id: str) -> dict[str, Any]:
        """Seal the open block and open the next; prune oldest if over capacity."""
        try:
            return run_commit_block(store, block_id=block_id)
        except ValueError as e:
            raise ToolError(str(e)) from e

    @mcp.tool(name="edit_block")
    def edit_block(block_id: str, edits: dict[str, Any]) -> dict[str, Any]:
        """Edit open block content. edits: content (replace), and/or append, prepend."""
        try:
            return run_edit_block(store, block_id=block_id, edits=edits)
        except ValueError as e:
            raise ToolError(str(e)) from e

    @mcp.tool(name="cancel_block")
    def cancel_block(block_id: str, reason: str) -> dict[str, Any]:
        """Discard the current open block and relink a fresh empty block."""
        try:
            return run_cancel_block(store, block_id=block_id, reason=reason)
        except ValueError as e:
            raise ToolError(str(e)) from e

    @mcp.tool(name="get_chain_stats")
    def get_chain_stats(chain_id: str) -> dict[str, Any]:
        """Block count, token totals, quota remaining, pruning flag."""
        try:
            return run_get_chain_stats(store, chain_id=chain_id)
        except ValueError as e:
            raise ToolError(str(e)) from e

    @mcp.resource(
        "blockchain://chain/{chain_id}",
        mime_type="application/json",
    )
    def resource_chain(chain_id: str) -> dict[str, Any]:
        return build_chain_payload(store, chain_id)

    @mcp.resource(
        "blockchain://block/{block_id}",
        mime_type="application/json",
    )
    def resource_block(block_id: str) -> dict[str, Any]:
        try:
            return build_block_payload(store, block_id)
        except ValueError as e:
            raise ResourceError(str(e)) from e

    @mcp.resource(
        "blockchain://current/{chain_id}",
        mime_type="application/json",
    )
    def resource_current_explicit(chain_id: str) -> dict[str, Any]:
        return build_current_payload(store, chain_id)

    @mcp.resource(
        "blockchain://current",
        mime_type="application/json",
    )
    def resource_current_default() -> dict[str, Any]:
        cid = os.environ.get("DEFAULT_CHAIN_ID", "").strip()
        if not cid:
            raise ResourceError(
                "DEFAULT_CHAIN_ID is not set; use blockchain://current/{chain_id} "
                "or set DEFAULT_CHAIN_ID."
            )
        return build_current_payload(store, cid)

    return mcp


def run_stdio() -> None:
    asyncio.run(build_mcp().run_stdio_async())
