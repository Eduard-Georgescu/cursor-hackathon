from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


MAX_BLOCK_TOKEN_CAPACITY = 500_000


@dataclass
class Block:
    block_id: str
    content: str
    token_count: int
    validation_status: str | None = None  # pending | valid | invalid
    committed_at: str | None = None
    issues: list[str] = field(default_factory=list)
    cleaned_content: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Block:
        return cls(
            block_id=data["block_id"],
            content=data.get("content", ""),
            token_count=int(data.get("token_count", 0)),
            validation_status=data.get("validation_status"),
            committed_at=data.get("committed_at"),
            issues=list(data.get("issues") or []),
            cleaned_content=data.get("cleaned_content"),
        )


@dataclass
class Chain:
    chain_id: str
    committed_blocks: list[Block] = field(default_factory=list)
    open_block: Block | None = None
    subscription_block_quota: int = 10_000
    max_committed_blocks: int = 256
    pruning_enabled: bool = True

    def to_json(self) -> dict[str, Any]:
        return {
            "chain_id": self.chain_id,
            "committed_blocks": [b.to_json() for b in self.committed_blocks],
            "open_block": self.open_block.to_json() if self.open_block else None,
            "subscription_block_quota": self.subscription_block_quota,
            "max_committed_blocks": self.max_committed_blocks,
            "pruning_enabled": self.pruning_enabled,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Chain:
        open_raw = data.get("open_block")
        return cls(
            chain_id=data["chain_id"],
            committed_blocks=[
                Block.from_json(b) for b in data.get("committed_blocks") or []
            ],
            open_block=Block.from_json(open_raw) if open_raw else None,
            subscription_block_quota=int(data.get("subscription_block_quota", 10_000)),
            max_committed_blocks=int(data.get("max_committed_blocks", 256)),
            pruning_enabled=bool(data.get("pruning_enabled", True)),
        )


@dataclass
class RootState:
    chains: dict[str, Chain] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {"chains": {cid: c.to_json() for cid, c in self.chains.items()}}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> RootState:
        chains_raw = data.get("chains") or {}
        return cls(
            chains={cid: Chain.from_json(c) for cid, c in chains_raw.items()}
        )
