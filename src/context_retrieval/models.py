"""Data models for the Context Retrieval server.

These mirror the schema written by MCP Servers 1 & 2 (chain construction
and block ingestion). This server reads from `blocks` / `chains` /
`sessions` and writes to `summaries` / `checkpoints` / `snapshots`.

Any field marked `# shared` must stay byte-compatible across all three
servers. Shared models allow `extra` fields so the other servers can add
their own metadata without breaking ours.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["user", "agent", "tool", "system"]
DetailLevel = Literal["skim", "standard", "deep"]
SearchMode = Literal["semantic", "lexical", "hybrid"]
CheckpointCategory = Literal[
    "decision", "question", "result", "transition", "blocker"
]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------- shared


class Block(BaseModel):
    """An atomic unit of context inside a chain.

    A block typically corresponds to one prompt/response/tool-call.
    Blocks are immutable once written: edits become new blocks with bumped
    `sequence` numbers.
    """

    model_config = ConfigDict(extra="allow")

    block_id: str  # shared
    chain_id: str  # shared
    sequence: int  # shared, 0-based position in the chain
    role: Role  # shared
    content: str  # shared, the raw text
    metadata: dict[str, Any] = Field(default_factory=dict)  # shared
    created_at: datetime = Field(default_factory=_utcnow)  # shared


class Chain(BaseModel):
    """An ordered sequence of blocks. One chain per logical thread of work."""

    model_config = ConfigDict(extra="allow")

    chain_id: str
    session_id: str
    title: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)


class Session(BaseModel):
    """An active-agent session. Owns lookback config and the active chain."""

    model_config = ConfigDict(extra="allow")

    session_id: str
    active_chain_id: str | None = None
    lookback_limit: int = 50
    created_at: datetime = Field(default_factory=_utcnow)


# ---------------------------------------------------------------- query/match


class ScoreBreakdown(BaseModel):
    """Per-leg contribution to a Match's relevance score."""

    semantic: float = 0.0
    lexical: float = 0.0
    recency: float = 0.0
    fused: float = 0.0  # the final fused score after RRF + recency boost


class Match(BaseModel):
    """A single hit returned by `query_chain`."""

    block_id: str
    chain_id: str
    sequence: int
    role: Role
    created_at: datetime
    relevance: float = Field(ge=0.0, le=1.0)
    preview: str  # first ~200 chars of the block, for the client UI
    # Optional richer fields populated when `explain=True`.
    score_breakdown: ScoreBreakdown | None = None
    context_hint: str | None = None  # one-line "why this matched"


class TimeRange(BaseModel):
    """Inclusive time-range filter for `query_chain`."""

    from_: datetime | None = Field(default=None, alias="from")
    to: datetime | None = None

    model_config = ConfigDict(populate_by_name=True)


class QueryResult(BaseModel):
    """Return shape for `query_chain`."""

    chain_id: str
    prompt: str
    lookback_limit: int
    mode: SearchMode
    total_candidates: int
    matches: list[Match]


# ------------------------------------------------------------- adjacent scan


class AdjacentResult(BaseModel):
    """Return shape for `scan_adjacent`."""

    block_id: str
    before_window: int
    after_window: int
    blocks: list[Block]
    merged_context: str
    total_tokens: int
    total_chain_length: int
    truncated: bool = False  # true when max_tokens forced us to drop blocks


# ------------------------------------------------------------------- lookback


class LookbackResult(BaseModel):
    """Return shape for `set_lookback`."""

    session_id: str
    scope: Literal["session", "chain"]
    chain_id: str | None = None
    previous_limit: int
    updated_limit: int


# ------------------------------------------------------------------- summary


class Decision(BaseModel):
    """A meaningful decision made during a chain."""

    summary: str
    block_id: str | None = None  # anchor block where the decision happened


class Checkpoint(BaseModel):
    """A user-facing anchor inside a plain-language history.

    Each checkpoint corresponds to a block (or a small run of blocks) and
    is what the user clicks in the history tab to backtrack.
    """

    checkpoint_id: str
    summary_id: str
    block_id: str  # the anchor block
    chain_id: str
    position: int  # ordering within the summary, 0-based
    label: str  # short heading shown in the UI
    description: str  # one-sentence plain-language description
    # Optional richer fields. Always present when the summary engine is "claude".
    reason: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    category: CheckpointCategory | None = None


class Summary(BaseModel):
    """Claude-generated plain-language history for a chain."""

    summary_id: str
    chain_id: str
    session_id: str
    plain_text: str
    detail_level: DetailLevel = "standard"
    engine: str = "extractive"  # "claude-<model>" or "extractive"
    token_count: int = 0  # rough token count of plain_text + blocks summarized
    block_count: int = 0  # number of input blocks at summary time (cache key)
    themes: list[str] = Field(default_factory=list)
    decisions: list[Decision] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    checkpoints: list[Checkpoint] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)


# ------------------------------------------------------------------ snapshot


class SnapshotManifest(BaseModel):
    """Compact descriptor of what's inside a Snapshot."""

    block_count: int
    token_estimate: int
    role_counts: dict[str, int] = Field(default_factory=dict)
    chain_title: str | None = None


class Provenance(BaseModel):
    """Audit trail for a Snapshot."""

    created_by: str  # session_id of the creator
    reason: str | None = None
    dry_run: bool = False
    name: str | None = None  # optional human label


class Snapshot(BaseModel):
    """Frozen agent state at a specific chain point.

    Created by `backtrack`. The `context` blob is everything the active
    agent needs to resume: the prefix of blocks up to and including the
    anchor, plus session-level config in effect at the time.
    """

    snapshot_id: str
    session_id: str
    chain_id: str
    checkpoint_id: str | None = None
    anchor_block_id: str
    anchor_sequence: int
    lookback_limit: int = 50
    manifest: SnapshotManifest
    provenance: Provenance
    blocks: list[Block] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)


class BacktrackResult(BaseModel):
    """Return shape for `backtrack`."""

    snapshot_id: str | None  # None when dry_run=True
    anchor_block_id: str
    anchor_sequence: int
    block_count: int
    estimated_tokens: int
    role_counts: dict[str, int] = Field(default_factory=dict)
    restored_summary: str  # one-line human-readable description
    dry_run: bool = False


class ReclaimResult(BaseModel):
    """Return shape for `reclaim_context`."""

    success: bool
    snapshot_id: str
    tokens_injected: int
    blocks_injected: int
    blocks_skipped: int = 0
    compressed: bool = False
    compression_summary: str | None = None
    active_chain_id: str
