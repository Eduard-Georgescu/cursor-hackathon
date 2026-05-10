from __future__ import annotations

import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from blockchain_mcp.models import MAX_BLOCK_TOKEN_CAPACITY, Block, Chain, RootState
from blockchain_mcp.store import persistence
from blockchain_mcp.store.tokens import estimate_tokens


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


class ChainStore:
    """JSON-backed chain storage with open block + pruning."""

    def __init__(self, data_dir: Path | None = None) -> None:
        base = data_dir or Path(
            os.environ.get("DATA_DIR", "./data")
        ).expanduser().resolve()
        self._path = base / "chains.json"
        self._lock = threading.Lock()
        self._root = RootState()
        self._default_quota = _env_int("SUBSCRIPTION_BLOCK_QUOTA", 10_000)
        self._default_max_committed = _env_int("MAX_BLOCKS_PER_CHAIN", 256)
        self._load()

    def _load(self) -> None:
        raw = persistence.load_json(self._path)
        if not raw:
            self._root = RootState()
            return
        self._root = RootState.from_json(raw)

    def _save(self) -> None:
        persistence.atomic_write_json(self._path, self._root.to_json())

    def _ensure_chain(self, chain_id: str) -> Chain:
        if chain_id not in self._root.chains:
            bid = str(uuid.uuid4())
            self._root.chains[chain_id] = Chain(
                chain_id=chain_id,
                committed_blocks=[],
                open_block=Block(
                    block_id=bid,
                    content="",
                    token_count=0,
                    validation_status="pending",
                ),
                subscription_block_quota=self._default_quota,
                max_committed_blocks=self._default_max_committed,
                pruning_enabled=True,
            )
        chain = self._root.chains[chain_id]
        if chain.open_block is None:
            bid = str(uuid.uuid4())
            chain.open_block = Block(
                block_id=bid,
                content="",
                token_count=0,
                validation_status="pending",
            )
        return chain

    def append_context(
        self, chain_id: str, content: str, token_count: int | None = None
    ) -> tuple[str, int]:
        with self._lock:
            chain = self._ensure_chain(chain_id)
            committed_count = len(chain.committed_blocks)
            quota_remaining = chain.subscription_block_quota - committed_count
            if quota_remaining <= 0:
                raise ValueError(
                    "Subscription block quota exhausted; cannot append new context."
                )
            ob = chain.open_block
            assert ob is not None
            add_tokens = (
                token_count if token_count is not None else estimate_tokens(content)
            )
            new_total = ob.token_count + add_tokens
            if new_total > MAX_BLOCK_TOKEN_CAPACITY:
                raise ValueError(
                    f"Open block would exceed {MAX_BLOCK_TOKEN_CAPACITY} token capacity "
                    f"(current {ob.token_count}, adding {add_tokens})."
                )
            ob.content += content
            ob.token_count = new_total
            ob.validation_status = "pending"
            ob.issues = []
            ob.cleaned_content = None
            self._save()
            return ob.block_id, max(
                0, MAX_BLOCK_TOKEN_CAPACITY - ob.token_count
            )

    def apply_validation_to_block(
        self,
        block_id: str,
        *,
        valid: bool,
        issues: list[str],
        cleaned_content: str,
    ) -> Block:
        with self._lock:
            blk, _chain = self._find_block_anywhere(block_id)
            if blk is None:
                raise ValueError(f"Unknown block_id: {block_id}")
            if blk.committed_at is not None:
                raise ValueError("Cannot re-validate a committed block.")
            blk.validation_status = "valid" if valid else "invalid"
            blk.issues = issues
            blk.cleaned_content = cleaned_content
            self._save()
            return blk

    def commit_block(self, block_id: str) -> tuple[str, str, str | None]:
        with self._lock:
            chain = self._chain_for_open_block(block_id)
            if chain is None:
                raise ValueError(f"block_id is not the open block on any chain: {block_id}")
            ob = chain.open_block
            assert ob is not None
            ob.committed_at = _utc_now_iso()
            ob.validation_status = ob.validation_status or "pending"
            chain.committed_blocks.append(ob)
            pruned_id: str | None = None
            max_kept = chain.max_committed_blocks
            if chain.pruning_enabled and len(chain.committed_blocks) > max_kept:
                overflow = len(chain.committed_blocks) - max_kept
                removed = chain.committed_blocks[:overflow]
                chain.committed_blocks = chain.committed_blocks[overflow:]
                pruned_id = removed[0].block_id if removed else None
            committed_id = ob.block_id
            nbid = str(uuid.uuid4())
            chain.open_block = Block(
                block_id=nbid,
                content="",
                token_count=0,
                validation_status="pending",
            )
            self._save()
            return committed_id, nbid, pruned_id

    def edit_block(self, block_id: str, edits: dict) -> tuple[str, str]:
        import difflib

        with self._lock:
            blk, chain = self._find_block_anywhere(block_id)
            if blk is None:
                raise ValueError(f"Unknown block_id: {block_id}")
            if blk.committed_at is not None:
                raise ValueError("Cannot edit a committed block.")
            old = blk.content
            new_content = old
            if "content" in edits and edits["content"] is not None:
                new_content = str(edits["content"])
            else:
                if edits.get("prepend"):
                    new_content = str(edits["prepend"]) + new_content
                if edits.get("append"):
                    new_content = new_content + str(edits["append"])
            diff = "".join(
                difflib.unified_diff(
                    old.splitlines(True),
                    new_content.splitlines(True),
                    fromfile="before",
                    tofile="after",
                )
            )
            blk.content = new_content
            blk.token_count = estimate_tokens(new_content)
            blk.validation_status = "pending"
            blk.issues = []
            blk.cleaned_content = None
            self._save()
            return blk.block_id, diff or "(no textual changes)"

    def cancel_block(self, block_id: str, reason: str) -> tuple[bool, bool]:
        with self._lock:
            for chain in self._root.chains.values():
                ob = chain.open_block
                if ob and ob.block_id == block_id:
                    nbid = str(uuid.uuid4())
                    chain.open_block = Block(
                        block_id=nbid,
                        content="",
                        token_count=0,
                        validation_status="pending",
                        issues=[f"cancelled:{reason}"],
                    )
                    self._save()
                    return True, True
            return False, False

    def get_chain_stats(self, chain_id: str) -> dict:
        with self._lock:
            chain = self._ensure_chain(chain_id)
            committed = chain.committed_blocks
            open_b = chain.open_block
            open_tokens = open_b.token_count if open_b else 0
            committed_tokens = sum(b.token_count for b in committed)
            total_tokens = committed_tokens + open_tokens
            committed_count = len(committed)
            quota_remaining = chain.subscription_block_quota - committed_count
            return {
                "block_count": committed_count + (1 if open_b else 0),
                "total_tokens": total_tokens,
                "quota_remaining": max(0, quota_remaining),
                "pruning_enabled": chain.pruning_enabled,
            }

    def get_chain(self, chain_id: str) -> Chain:
        with self._lock:
            return self._ensure_chain(chain_id)

    def get_block(self, block_id: str) -> Block | None:
        with self._lock:
            blk, _ = self._find_block_anywhere(block_id)
            return blk

    def current_open_block(self, chain_id: str) -> Block:
        with self._lock:
            chain = self._ensure_chain(chain_id)
            assert chain.open_block is not None
            return chain.open_block

    def _find_block_anywhere(self, block_id: str) -> tuple[Block | None, Chain | None]:
        for chain in self._root.chains.values():
            if chain.open_block and chain.open_block.block_id == block_id:
                return chain.open_block, chain
            for b in chain.committed_blocks:
                if b.block_id == block_id:
                    return b, chain
        return None, None

    def _chain_for_open_block(self, block_id: str) -> Chain | None:
        for chain in self._root.chains.values():
            if chain.open_block and chain.open_block.block_id == block_id:
                return chain
        return None
