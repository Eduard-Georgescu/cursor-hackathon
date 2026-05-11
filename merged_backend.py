# =============================================================================
# MERGED BACKEND — cursor-hackathon / mcp-context-retrieval
# Files merged (in dependency order):
#   config.py · logging_utils.py · models.py · prompts.py
#   store.py · search.py · summarizer.py · server.py · web.py · __main__.py
# =============================================================================

# ── Dependencies (pyproject.toml) ────────────────────────────────────────────
# Core:    mcp[cli]>=1.27.0  pydantic>=2.6  numpy>=1.26
# Optional:
#   embeddings  → sentence-transformers>=2.7
#   summaries   → anthropic>=0.39
#   web         → fastapi>=0.110  uvicorn[standard]>=0.27
#   dev         → pytest>=8.0  pytest-asyncio>=0.23
# Install everything: pip install -e ".[all]"
# Run MCP server: python -m context_retrieval   (or: mcp-context-retrieval)
# Run web demo:   uvicorn context_retrieval.web:app --reload
# =============================================================================

from __future__ import annotations

# ── stdlib ────────────────────────────────────────────────────────────────────
import argparse
import asyncio
import hashlib
import json
import logging
import math
import os
import re
import sqlite3
import sys
import threading
import time
import uuid
from collections import Counter
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Annotated,
    Any,
    AsyncIterator,
    Iterable,
    Iterator,
    Literal,
    Protocol,
)

# ── third-party ───────────────────────────────────────────────────────────────
import numpy as np
from pydantic import AnyUrl, BaseModel, ConfigDict, Field

# ── MCP (installed via mcp[cli]) ─────────────────────────────────────────────
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.prompts.base import Message, UserMessage


# =============================================================================
# config.py
# =============================================================================

def _env(name: str, default: str | None = None) -> str | None:
    val = os.environ.get(name)
    if val is None or val == "":
        return default
    return val


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _default_db_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "context.db"


class Settings(BaseModel):
    """All runtime configuration in one place."""

    model_config = ConfigDict(extra="ignore")

    db_path: Path = Field(default_factory=_default_db_path)
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-4-5"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    log_level: str = "INFO"
    max_lookback: int = Field(default=1000, ge=1)
    max_blocks_to_summarize: int = Field(default=200, ge=1)
    max_chars_per_block: int = Field(default=1200, ge=100)
    summary_cache_ttl_sec: int = Field(default=0, ge=0)

    @classmethod
    def from_env(cls) -> "Settings":
        db_override = _env("CONTEXT_RETRIEVAL_DB")
        kwargs: dict[str, Any] = {
            "anthropic_api_key": _env("ANTHROPIC_API_KEY"),
            "anthropic_model": _env("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
            "embedding_model": _env(
                "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            "log_level": _env("LOG_LEVEL", "INFO"),
            "max_lookback": _env_int("MAX_LOOKBACK", 1000),
            "max_blocks_to_summarize": _env_int("MAX_BLOCKS_TO_SUMMARIZE", 200),
            "max_chars_per_block": _env_int("MAX_CHARS_PER_BLOCK", 1200),
            "summary_cache_ttl_sec": _env_int("SUMMARY_CACHE_TTL_SEC", 0),
        }
        if db_override:
            kwargs["db_path"] = Path(db_override).expanduser()
        return cls(**kwargs)

    @property
    def claude_enabled(self) -> bool:
        return bool(self.anthropic_api_key)


# =============================================================================
# logging_utils.py
# =============================================================================

_LOGGING_CONFIGURED = False
_LOGGER_NAME = "context_retrieval"


def configure_logging(level: str = "INFO") -> None:
    """Idempotent root config. Safe to call multiple times."""
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(level.upper())
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
    )
    logger.addHandler(handler)
    logger.propagate = False
    _LOGGING_CONFIGURED = True


def get_logger(name: str | None = None) -> logging.Logger:
    full = _LOGGER_NAME if name is None else f"{_LOGGER_NAME}.{name}"
    return logging.getLogger(full)


def _format_fields(fields: dict[str, Any]) -> str:
    parts: list[str] = []
    for k, v in fields.items():
        if v is None:
            continue
        s = str(v)
        if any(c in s for c in (" ", "=", '"')):
            s = '"' + s.replace('"', '\\"') + '"'
        parts.append(f"{k}={s}")
    return " ".join(parts)


@contextmanager
def log_call(tool: str, **fields: Any) -> Iterator[dict[str, Any]]:
    """Log the start, end, and timing of a tool call."""
    log = get_logger("tool")
    extras: dict[str, Any] = {}
    start = time.perf_counter()
    log.info(_format_fields({"event": "tool.start", "tool": tool, **fields}))
    try:
        yield extras
    except Exception as exc:
        duration_ms = round((time.perf_counter() - start) * 1000, 1)
        log.exception(
            _format_fields(
                {
                    "event": "tool.error",
                    "tool": tool,
                    "result": "error",
                    "duration_ms": duration_ms,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    **fields,
                    **extras,
                }
            )
        )
        raise
    else:
        duration_ms = round((time.perf_counter() - start) * 1000, 1)
        log.info(
            _format_fields(
                {
                    "event": "tool.end",
                    "tool": tool,
                    "result": "ok",
                    "duration_ms": duration_ms,
                    **fields,
                    **extras,
                }
            )
        )


# =============================================================================
# models.py
# =============================================================================

Role = Literal["user", "agent", "tool", "system"]
DetailLevel = Literal["skim", "standard", "deep"]
SearchMode = Literal["semantic", "lexical", "hybrid"]
CheckpointCategory = Literal[
    "decision", "question", "result", "transition", "blocker"
]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Block(BaseModel):
    model_config = ConfigDict(extra="allow")

    block_id: str
    chain_id: str
    sequence: int
    role: Role
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utcnow)


class Chain(BaseModel):
    model_config = ConfigDict(extra="allow")

    chain_id: str
    session_id: str
    title: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)


class Session(BaseModel):
    model_config = ConfigDict(extra="allow")

    session_id: str
    active_chain_id: str | None = None
    lookback_limit: int = 50
    created_at: datetime = Field(default_factory=_utcnow)


class ScoreBreakdown(BaseModel):
    semantic: float = 0.0
    lexical: float = 0.0
    recency: float = 0.0
    fused: float = 0.0


class Match(BaseModel):
    block_id: str
    chain_id: str
    sequence: int
    role: Role
    created_at: datetime
    relevance: float = Field(ge=0.0, le=1.0)
    preview: str
    score_breakdown: ScoreBreakdown | None = None
    context_hint: str | None = None


class TimeRange(BaseModel):
    from_: datetime | None = Field(default=None, alias="from")
    to: datetime | None = None

    model_config = ConfigDict(populate_by_name=True)


class QueryResult(BaseModel):
    chain_id: str
    prompt: str
    lookback_limit: int
    mode: SearchMode
    total_candidates: int
    matches: list[Match]


class AdjacentResult(BaseModel):
    block_id: str
    before_window: int
    after_window: int
    blocks: list[Block]
    merged_context: str
    total_tokens: int
    total_chain_length: int
    truncated: bool = False


class LookbackResult(BaseModel):
    session_id: str
    scope: Literal["session", "chain"]
    chain_id: str | None = None
    previous_limit: int
    updated_limit: int


class Decision(BaseModel):
    summary: str
    block_id: str | None = None


class Checkpoint(BaseModel):
    checkpoint_id: str
    summary_id: str
    block_id: str
    chain_id: str
    position: int
    label: str
    description: str
    reason: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    category: CheckpointCategory | None = None


class Summary(BaseModel):
    summary_id: str
    chain_id: str
    session_id: str
    plain_text: str
    detail_level: DetailLevel = "standard"
    engine: str = "extractive"
    token_count: int = 0
    block_count: int = 0
    themes: list[str] = Field(default_factory=list)
    decisions: list[Decision] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    checkpoints: list[Checkpoint] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)


class SnapshotManifest(BaseModel):
    block_count: int
    token_estimate: int
    role_counts: dict[str, int] = Field(default_factory=dict)
    chain_title: str | None = None


class Provenance(BaseModel):
    created_by: str
    reason: str | None = None
    dry_run: bool = False
    name: str | None = None


class Snapshot(BaseModel):
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
    snapshot_id: str | None
    anchor_block_id: str
    anchor_sequence: int
    block_count: int
    estimated_tokens: int
    role_counts: dict[str, int] = Field(default_factory=dict)
    restored_summary: str
    dry_run: bool = False


class ReclaimResult(BaseModel):
    success: bool
    snapshot_id: str
    tokens_injected: int
    blocks_injected: int
    blocks_skipped: int = 0
    compressed: bool = False
    compression_summary: str | None = None
    active_chain_id: str


# =============================================================================
# prompts.py
# =============================================================================

SHARED_PREAMBLE = """You are the History view of an agent IDE.

You will receive a chain of blocks (each one prompt / response / tool
call inside an agent session). Your job is to convert this raw chain
into a plain-language recap that the user will read in a "History" tab,
plus a small set of clickable checkpoints they can use to backtrack the
agent to earlier points.

Return ONLY valid JSON. Do not wrap the JSON in markdown fences. Never
invent block_ids — every `block_id` must reference one from the input.
"""

SKIM_SCHEMA = """Schema:
{
  "summary": string,            // 2-3 short paragraphs, second-person ("you asked...")
  "themes": [string, ...],      // 1-3 short tags
  "checkpoints": [              // 3-5 entries
    {
      "block_id": string,       // must reference an input block_id
      "label": string,          // <= 60 chars, headline-style
      "description": string,    // one sentence, plain English
      "reason": string,         // one sentence: why this is a useful anchor
      "confidence": number,     // 0..1
      "category": string        // "decision" | "question" | "result" | "transition" | "blocker"
    }
  ]
}"""

STANDARD_SCHEMA = """Schema:
{
  "summary": string,            // 3-6 short paragraphs, second-person
  "themes": [string, ...],      // 3-6 short tags
  "decisions": [
    { "summary": string, "block_id": string }
  ],
  "open_questions": [string, ...],
  "checkpoints": [              // 4-10 entries
    {
      "block_id": string,
      "label": string,
      "description": string,
      "reason": string,
      "confidence": number,
      "category": string
    }
  ]
}"""

DEEP_PASS1_SCHEMA = """Schema (Pass 1 — extraction only):
{
  "themes": [string, ...],
  "decisions": [
    { "summary": string, "block_id": string }
  ],
  "open_questions": [string, ...],
  "anchor_candidates": [
    {
      "block_id": string,
      "category": string,
      "reason": string
    }
  ]
}
Categories: "decision" | "question" | "result" | "transition" | "blocker".
Aim for 8-15 anchor candidates spread across the chain."""

DEEP_PASS2_SCHEMA = """Schema (Pass 2 — narrate around the themes/anchors from pass 1):
{
  "summary": string,
  "checkpoints": [              // 6-12 entries
    {
      "block_id": string,
      "label": string,          // <= 60 chars
      "description": string,
      "reason": string,
      "confidence": number,
      "category": string
    }
  ]
}"""


def skim_prompt() -> str:
    return SHARED_PREAMBLE + "\n" + SKIM_SCHEMA + "\n\nGuidance:\n" + _shared_guidance(short=True)


def standard_prompt() -> str:
    return SHARED_PREAMBLE + "\n" + STANDARD_SCHEMA + "\n\nGuidance:\n" + _shared_guidance(short=False)


def deep_pass1_prompt() -> str:
    return (
        SHARED_PREAMBLE
        + "\n"
        + DEEP_PASS1_SCHEMA
        + "\n\nDo not write the narrative yet — only the extraction. "
        + "Cover the entire chain, not just the most recent blocks."
    )


def deep_pass2_prompt(pass1_json: str) -> str:
    return (
        SHARED_PREAMBLE
        + "\n"
        + DEEP_PASS2_SCHEMA
        + "\n\nYou previously produced this extraction (pass 1):\n"
        + pass1_json
        + "\n\nNow write the user-facing narrative and select checkpoints. "
        + "Anchor every checkpoint to a block_id from the input. Cite "
        + "anchor_candidates where they fit; you may also pick other blocks."
    )


def _shared_guidance(short: bool) -> str:
    lines = [
        "- Write second-person, friendly, concrete. No corporate jargon.",
        "- Anchor every checkpoint to a real block_id from the input.",
        "- Pick checkpoints that are *useful entry points for backtracking*: "
        "decisions, question moments, key results, topic transitions, blockers.",
        "- Spread checkpoints across the chain. Don't cluster them all near the end.",
        "- Confidence should reflect how certain you are the user will want to backtrack to this point.",
    ]
    if not short:
        lines.append(
            "- Surface themes that span multiple blocks, not labels for single blocks."
        )
        lines.append(
            "- Surface open_questions only when there is a genuinely unresolved thread."
        )
    return "\n".join(lines)


# =============================================================================
# store.py
# =============================================================================

SCHEMA = """
CREATE TABLE IF NOT EXISTS chains (
    chain_id    TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL,
    title       TEXT,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id      TEXT PRIMARY KEY,
    active_chain_id TEXT,
    lookback_limit  INTEGER NOT NULL DEFAULT 50,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chain_lookbacks (
    session_id   TEXT NOT NULL,
    chain_id     TEXT NOT NULL,
    lookback     INTEGER NOT NULL,
    PRIMARY KEY (session_id, chain_id)
);

CREATE TABLE IF NOT EXISTS blocks (
    block_id    TEXT PRIMARY KEY,
    chain_id    TEXT NOT NULL,
    sequence    INTEGER NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    metadata    TEXT NOT NULL DEFAULT '{}',
    created_at  TEXT NOT NULL,
    UNIQUE(chain_id, sequence)
);

CREATE INDEX IF NOT EXISTS idx_blocks_chain_seq
    ON blocks(chain_id, sequence);

CREATE VIRTUAL TABLE IF NOT EXISTS blocks_fts USING fts5(
    content,
    content='blocks',
    content_rowid='rowid',
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS blocks_ai AFTER INSERT ON blocks BEGIN
    INSERT INTO blocks_fts(rowid, content) VALUES (new.rowid, new.content);
END;
CREATE TRIGGER IF NOT EXISTS blocks_ad AFTER DELETE ON blocks BEGIN
    INSERT INTO blocks_fts(blocks_fts, rowid, content)
        VALUES ('delete', old.rowid, old.content);
END;
CREATE TRIGGER IF NOT EXISTS blocks_au AFTER UPDATE ON blocks BEGIN
    INSERT INTO blocks_fts(blocks_fts, rowid, content)
        VALUES ('delete', old.rowid, old.content);
    INSERT INTO blocks_fts(rowid, content) VALUES (new.rowid, new.content);
END;

CREATE TABLE IF NOT EXISTS block_embeddings (
    block_id    TEXT PRIMARY KEY,
    model       TEXT NOT NULL,
    dim         INTEGER NOT NULL,
    vector      BLOB NOT NULL,
    FOREIGN KEY (block_id) REFERENCES blocks(block_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS summaries (
    summary_id   TEXT PRIMARY KEY,
    chain_id     TEXT NOT NULL,
    session_id   TEXT NOT NULL,
    plain_text   TEXT NOT NULL,
    detail_level TEXT NOT NULL DEFAULT 'standard',
    engine       TEXT NOT NULL DEFAULT 'extractive',
    token_count  INTEGER NOT NULL DEFAULT 0,
    block_count  INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_summaries_session
    ON summaries(session_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_summaries_chain
    ON summaries(chain_id, created_at DESC);

CREATE TABLE IF NOT EXISTS summary_themes (
    summary_id   TEXT NOT NULL,
    position     INTEGER NOT NULL,
    theme        TEXT NOT NULL,
    PRIMARY KEY (summary_id, position),
    FOREIGN KEY (summary_id) REFERENCES summaries(summary_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS summary_decisions (
    summary_id   TEXT NOT NULL,
    position     INTEGER NOT NULL,
    summary      TEXT NOT NULL,
    block_id     TEXT,
    PRIMARY KEY (summary_id, position),
    FOREIGN KEY (summary_id) REFERENCES summaries(summary_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS summary_open_questions (
    summary_id   TEXT NOT NULL,
    position     INTEGER NOT NULL,
    question     TEXT NOT NULL,
    PRIMARY KEY (summary_id, position),
    FOREIGN KEY (summary_id) REFERENCES summaries(summary_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS checkpoints (
    checkpoint_id TEXT PRIMARY KEY,
    summary_id    TEXT NOT NULL,
    block_id      TEXT NOT NULL,
    chain_id      TEXT NOT NULL,
    position      INTEGER NOT NULL,
    label         TEXT NOT NULL,
    description   TEXT NOT NULL,
    reason        TEXT,
    confidence    REAL,
    category      TEXT,
    FOREIGN KEY (summary_id) REFERENCES summaries(summary_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_checkpoints_summary
    ON checkpoints(summary_id, position);

CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id      TEXT PRIMARY KEY,
    session_id       TEXT NOT NULL,
    chain_id         TEXT NOT NULL,
    checkpoint_id    TEXT,
    anchor_block_id  TEXT NOT NULL,
    anchor_sequence  INTEGER NOT NULL,
    lookback_limit   INTEGER NOT NULL DEFAULT 50,
    manifest_json    TEXT NOT NULL,
    provenance_json  TEXT NOT NULL,
    blocks_json      TEXT NOT NULL,
    created_at       TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_snapshots_session
    ON snapshots(session_id, created_at DESC);
"""


def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def _parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s)


def _row_to_block(row: sqlite3.Row) -> Block:
    return Block(
        block_id=row["block_id"],
        chain_id=row["chain_id"],
        sequence=row["sequence"],
        role=row["role"],
        content=row["content"],
        metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        created_at=_parse_iso(row["created_at"]),
    )


class Store(Protocol):
    """Storage interface used by Server 3."""

    def upsert_chain(self, chain: Chain) -> None: ...
    def get_chain(self, chain_id: str) -> Chain | None: ...
    def upsert_session(self, session: Session) -> None: ...
    def get_session(self, session_id: str) -> Session | None: ...
    def set_session_lookback(self, session_id: str, limit: int) -> tuple[int, int]: ...
    def set_chain_lookback(self, session_id: str, chain_id: str, limit: int) -> tuple[int, int]: ...
    def get_effective_lookback(self, session_id: str, chain_id: str | None, default: int) -> int: ...
    def upsert_block(self, block: Block) -> None: ...
    def get_block(self, block_id: str) -> Block | None: ...
    def list_chain_blocks(self, chain_id: str, *, limit: int | None = None, up_to_sequence: int | None = None) -> list[Block]: ...
    def adjacent_blocks(self, block_id: str, *, before: int, after: int) -> list[Block]: ...
    def chain_block_count(self, chain_id: str) -> int: ...
    def lexical_search(self, chain_id: str, query: str, *, limit: int) -> list[tuple[Block, float]]: ...
    def get_embedding(self, block_id: str) -> tuple[str, np.ndarray] | None: ...
    def put_embedding(self, block_id: str, model: str, vector: np.ndarray) -> None: ...
    def insert_summary(self, summary: Summary) -> None: ...
    def latest_summary_for_session(self, session_id: str) -> Summary | None: ...
    def latest_summary_for_chain(self, chain_id: str, detail_level: str | None = None) -> Summary | None: ...
    def get_checkpoint(self, checkpoint_id: str) -> Checkpoint | None: ...
    def insert_snapshot(self, snapshot: Snapshot) -> None: ...
    def get_snapshot(self, snapshot_id: str) -> Snapshot | None: ...
    def list_snapshots(self, session_id: str) -> list[Snapshot]: ...


class SQLiteStore:
    """Thin SQLite wrapper. Thread-safe via a single connection + RLock."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            detect_types=sqlite3.PARSE_DECLTYPES,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        with self._conn:
            self._conn.executescript(SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            with self._conn:
                yield self._conn

    def get_chain(self, chain_id: str) -> Chain | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM chains WHERE chain_id = ?", (chain_id,)
            ).fetchone()
        if row is None:
            return None
        return Chain(
            chain_id=row["chain_id"],
            session_id=row["session_id"],
            title=row["title"],
            created_at=_parse_iso(row["created_at"]),
        )

    def upsert_chain(self, chain: Chain) -> None:
        with self._tx() as cx:
            cx.execute(
                """
                INSERT INTO chains (chain_id, session_id, title, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(chain_id) DO UPDATE SET
                    session_id = excluded.session_id,
                    title      = excluded.title
                """,
                (chain.chain_id, chain.session_id, chain.title, _iso(chain.created_at)),
            )

    def get_session(self, session_id: str) -> Session | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        if row is None:
            return None
        return Session(
            session_id=row["session_id"],
            active_chain_id=row["active_chain_id"],
            lookback_limit=row["lookback_limit"],
            created_at=_parse_iso(row["created_at"]),
        )

    def upsert_session(self, session: Session) -> None:
        with self._tx() as cx:
            cx.execute(
                """
                INSERT INTO sessions
                    (session_id, active_chain_id, lookback_limit, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    active_chain_id = excluded.active_chain_id,
                    lookback_limit  = excluded.lookback_limit
                """,
                (
                    session.session_id,
                    session.active_chain_id,
                    session.lookback_limit,
                    _iso(session.created_at),
                ),
            )

    def set_session_lookback(self, session_id: str, limit: int) -> tuple[int, int]:
        existing = self.get_session(session_id)
        if existing is None:
            self.upsert_session(Session(session_id=session_id, lookback_limit=limit))
            return (50, limit)
        previous = existing.lookback_limit
        with self._tx() as cx:
            cx.execute(
                "UPDATE sessions SET lookback_limit = ? WHERE session_id = ?",
                (limit, session_id),
            )
        return (previous, limit)

    def set_chain_lookback(self, session_id: str, chain_id: str, limit: int) -> tuple[int, int]:
        with self._lock:
            row = self._conn.execute(
                "SELECT lookback FROM chain_lookbacks WHERE session_id = ? AND chain_id = ?",
                (session_id, chain_id),
            ).fetchone()
        previous = row["lookback"] if row else self.get_effective_lookback(session_id, chain_id, 50)
        with self._tx() as cx:
            cx.execute(
                """
                INSERT INTO chain_lookbacks (session_id, chain_id, lookback)
                VALUES (?, ?, ?)
                ON CONFLICT(session_id, chain_id) DO UPDATE SET lookback = excluded.lookback
                """,
                (session_id, chain_id, limit),
            )
        return (previous, limit)

    def get_effective_lookback(self, session_id: str, chain_id: str | None, default: int) -> int:
        if chain_id is not None:
            with self._lock:
                row = self._conn.execute(
                    "SELECT lookback FROM chain_lookbacks WHERE session_id = ? AND chain_id = ?",
                    (session_id, chain_id),
                ).fetchone()
            if row is not None:
                return int(row["lookback"])
        sess = self.get_session(session_id)
        return sess.lookback_limit if sess else default

    def get_block(self, block_id: str) -> Block | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM blocks WHERE block_id = ?", (block_id,)
            ).fetchone()
        return _row_to_block(row) if row else None

    def upsert_block(self, block: Block) -> None:
        with self._tx() as cx:
            cx.execute(
                """
                INSERT INTO blocks
                    (block_id, chain_id, sequence, role, content, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(block_id) DO UPDATE SET
                    content  = excluded.content,
                    metadata = excluded.metadata
                """,
                (
                    block.block_id, block.chain_id, block.sequence, block.role,
                    block.content, json.dumps(block.metadata), _iso(block.created_at),
                ),
            )

    def list_chain_blocks(
        self,
        chain_id: str,
        *,
        limit: int | None = None,
        up_to_sequence: int | None = None,
    ) -> list[Block]:
        sql = "SELECT * FROM blocks WHERE chain_id = ?"
        params: list[Any] = [chain_id]
        if up_to_sequence is not None:
            sql += " AND sequence <= ?"
            params.append(up_to_sequence)
        sql += " ORDER BY sequence ASC"
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        blocks = [_row_to_block(r) for r in rows]
        if limit is not None and limit > 0 and len(blocks) > limit:
            blocks = blocks[-limit:]
        return blocks

    def chain_block_count(self, chain_id: str) -> int:
        with self._lock:
            (count,) = self._conn.execute(
                "SELECT COUNT(*) FROM blocks WHERE chain_id = ?", (chain_id,)
            ).fetchone()
        return int(count)

    def adjacent_blocks(self, block_id: str, *, before: int, after: int) -> list[Block]:
        anchor = self.get_block(block_id)
        if anchor is None:
            return []
        lo = max(0, anchor.sequence - max(0, before))
        hi = anchor.sequence + max(0, after)
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM blocks WHERE chain_id = ? AND sequence BETWEEN ? AND ? ORDER BY sequence ASC",
                (anchor.chain_id, lo, hi),
            ).fetchall()
        return [_row_to_block(r) for r in rows]

    def lexical_search(self, chain_id: str, query: str, *, limit: int) -> list[tuple[Block, float]]:
        if not query.strip():
            return []
        sanitized = _sanitize_fts_query(query)
        if not sanitized:
            return []
        with self._lock:
            try:
                rows = self._conn.execute(
                    """
                    SELECT b.*, bm25(blocks_fts) AS score
                    FROM blocks_fts
                    JOIN blocks b ON b.rowid = blocks_fts.rowid
                    WHERE blocks_fts MATCH ? AND b.chain_id = ?
                    ORDER BY score ASC
                    LIMIT ?
                    """,
                    (sanitized, chain_id, limit),
                ).fetchall()
            except sqlite3.OperationalError:
                return []
        return [(_row_to_block(r), float(r["score"])) for r in rows]

    def get_embedding(self, block_id: str) -> tuple[str, np.ndarray] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT model, dim, vector FROM block_embeddings WHERE block_id = ?",
                (block_id,),
            ).fetchone()
        if row is None:
            return None
        vec = np.frombuffer(row["vector"], dtype=np.float32).reshape(row["dim"])
        return row["model"], np.array(vec, copy=True)

    def put_embedding(self, block_id: str, model: str, vector: np.ndarray) -> None:
        vec = np.ascontiguousarray(vector, dtype=np.float32)
        with self._tx() as cx:
            cx.execute(
                """
                INSERT INTO block_embeddings (block_id, model, dim, vector)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(block_id) DO UPDATE SET
                    model = excluded.model, dim = excluded.dim, vector = excluded.vector
                """,
                (block_id, model, vec.shape[0], vec.tobytes()),
            )

    def insert_summary(self, summary: Summary) -> None:
        with self._tx() as cx:
            cx.execute(
                """
                INSERT INTO summaries
                    (summary_id, chain_id, session_id, plain_text,
                     detail_level, engine, token_count, block_count, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    summary.summary_id, summary.chain_id, summary.session_id,
                    summary.plain_text, summary.detail_level, summary.engine,
                    summary.token_count, summary.block_count, _iso(summary.created_at),
                ),
            )
            for i, theme in enumerate(summary.themes):
                cx.execute(
                    "INSERT INTO summary_themes (summary_id, position, theme) VALUES (?, ?, ?)",
                    (summary.summary_id, i, theme),
                )
            for i, dec in enumerate(summary.decisions):
                cx.execute(
                    "INSERT INTO summary_decisions (summary_id, position, summary, block_id) VALUES (?, ?, ?, ?)",
                    (summary.summary_id, i, dec.summary, dec.block_id),
                )
            for i, q in enumerate(summary.open_questions):
                cx.execute(
                    "INSERT INTO summary_open_questions (summary_id, position, question) VALUES (?, ?, ?)",
                    (summary.summary_id, i, q),
                )
            for cp in summary.checkpoints:
                cx.execute(
                    """
                    INSERT INTO checkpoints
                        (checkpoint_id, summary_id, block_id, chain_id,
                         position, label, description, reason, confidence, category)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        cp.checkpoint_id, cp.summary_id, cp.block_id, cp.chain_id,
                        cp.position, cp.label, cp.description, cp.reason,
                        cp.confidence, cp.category,
                    ),
                )

    def _hydrate_summary(self, row: sqlite3.Row) -> Summary:
        sid = row["summary_id"]
        with self._lock:
            themes = [r["theme"] for r in self._conn.execute(
                "SELECT theme FROM summary_themes WHERE summary_id = ? ORDER BY position", (sid,)
            ).fetchall()]
            decisions = [
                Decision(summary=r["summary"], block_id=r["block_id"])
                for r in self._conn.execute(
                    "SELECT summary, block_id FROM summary_decisions WHERE summary_id = ? ORDER BY position", (sid,)
                ).fetchall()
            ]
            open_questions = [r["question"] for r in self._conn.execute(
                "SELECT question FROM summary_open_questions WHERE summary_id = ? ORDER BY position", (sid,)
            ).fetchall()]
            cp_rows = self._conn.execute(
                "SELECT * FROM checkpoints WHERE summary_id = ? ORDER BY position ASC", (sid,)
            ).fetchall()
        checkpoints = [
            Checkpoint(
                checkpoint_id=r["checkpoint_id"], summary_id=r["summary_id"],
                block_id=r["block_id"], chain_id=r["chain_id"], position=r["position"],
                label=r["label"], description=r["description"], reason=r["reason"],
                confidence=r["confidence"], category=r["category"],
            )
            for r in cp_rows
        ]
        return Summary(
            summary_id=row["summary_id"], chain_id=row["chain_id"],
            session_id=row["session_id"], plain_text=row["plain_text"],
            detail_level=row["detail_level"], engine=row["engine"],
            token_count=row["token_count"], block_count=row["block_count"],
            themes=themes, decisions=decisions, open_questions=open_questions,
            checkpoints=checkpoints, created_at=_parse_iso(row["created_at"]),
        )

    def latest_summary_for_session(self, session_id: str) -> Summary | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM summaries WHERE session_id = ? ORDER BY created_at DESC LIMIT 1",
                (session_id,),
            ).fetchone()
        return self._hydrate_summary(row) if row else None

    def latest_summary_for_chain(self, chain_id: str, detail_level: str | None = None) -> Summary | None:
        sql = "SELECT * FROM summaries WHERE chain_id = ?"
        params: list[Any] = [chain_id]
        if detail_level is not None:
            sql += " AND detail_level = ?"
            params.append(detail_level)
        sql += " ORDER BY created_at DESC LIMIT 1"
        with self._lock:
            row = self._conn.execute(sql, params).fetchone()
        return self._hydrate_summary(row) if row else None

    def get_checkpoint(self, checkpoint_id: str) -> Checkpoint | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM checkpoints WHERE checkpoint_id = ?", (checkpoint_id,)
            ).fetchone()
        if row is None:
            return None
        return Checkpoint(
            checkpoint_id=row["checkpoint_id"], summary_id=row["summary_id"],
            block_id=row["block_id"], chain_id=row["chain_id"], position=row["position"],
            label=row["label"], description=row["description"], reason=row["reason"],
            confidence=row["confidence"], category=row["category"],
        )

    def insert_snapshot(self, snapshot: Snapshot) -> None:
        with self._tx() as cx:
            cx.execute(
                """
                INSERT INTO snapshots
                    (snapshot_id, session_id, chain_id, checkpoint_id,
                     anchor_block_id, anchor_sequence, lookback_limit,
                     manifest_json, provenance_json, blocks_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot.snapshot_id, snapshot.session_id, snapshot.chain_id,
                    snapshot.checkpoint_id, snapshot.anchor_block_id, snapshot.anchor_sequence,
                    snapshot.lookback_limit, snapshot.manifest.model_dump_json(),
                    snapshot.provenance.model_dump_json(),
                    json.dumps([b.model_dump(mode="json") for b in snapshot.blocks]),
                    _iso(snapshot.created_at),
                ),
            )

    def get_snapshot(self, snapshot_id: str) -> Snapshot | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM snapshots WHERE snapshot_id = ?", (snapshot_id,)
            ).fetchone()
        return _row_to_snapshot(row) if row else None

    def list_snapshots(self, session_id: str) -> list[Snapshot]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM snapshots WHERE session_id = ? ORDER BY created_at DESC",
                (session_id,),
            ).fetchall()
        return [_row_to_snapshot(r) for r in rows]


def _row_to_snapshot(row: sqlite3.Row) -> Snapshot:
    return Snapshot(
        snapshot_id=row["snapshot_id"], session_id=row["session_id"],
        chain_id=row["chain_id"], checkpoint_id=row["checkpoint_id"],
        anchor_block_id=row["anchor_block_id"], anchor_sequence=row["anchor_sequence"],
        lookback_limit=row["lookback_limit"],
        manifest=SnapshotManifest.model_validate_json(row["manifest_json"]),
        provenance=Provenance.model_validate_json(row["provenance_json"]),
        blocks=[Block.model_validate(b) for b in json.loads(row["blocks_json"])],
        created_at=_parse_iso(row["created_at"]),
    )


_FTS_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _sanitize_fts_query(query: str) -> str:
    tokens = [t for t in _FTS_TOKEN_RE.findall(query) if len(t) > 1]
    if not tokens:
        return ""
    return " OR ".join(f'"{t}"' for t in tokens)


# =============================================================================
# search.py
# =============================================================================

_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
_FALLBACK_DIM = 256
_FALLBACK_MODEL = f"hashed-tfidf-{_FALLBACK_DIM}"
_RRF_K = 60
_RECENCY_TAU_DAYS = 30.0


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _WORD_RE.findall(text)]


def _hash_idx(token: str, dim: int) -> int:
    h = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(h, "big") % dim


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v if n == 0.0 else v / n


class Embedder:
    """Embedding interface with lazy model loading + graceful fallback."""

    def __init__(self, prefer_model: str | None = None):
        self._st_model = None
        self._st_name: str | None = None
        if prefer_model is None:
            prefer_model = "sentence-transformers/all-MiniLM-L6-v2"
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self._st_model = SentenceTransformer(prefer_model)
            self._st_name = prefer_model
        except Exception:
            self._st_model = None
            self._st_name = None

    @property
    def model_name(self) -> str:
        return self._st_name or _FALLBACK_MODEL

    @property
    def dim(self) -> int:
        if self._st_model is not None:
            return int(self._st_model.get_sentence_embedding_dimension())
        return _FALLBACK_DIM

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if self._st_model is not None:
            vecs = self._st_model.encode(
                texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False,
            )
            return vecs.astype(np.float32, copy=False)
        return np.stack([self._hashed_tfidf(t) for t in texts], axis=0)

    def _hashed_tfidf(self, text: str) -> np.ndarray:
        toks = _tokens(text)
        if not toks:
            return np.zeros(_FALLBACK_DIM, dtype=np.float32)
        counts = Counter(toks)
        vec = np.zeros(_FALLBACK_DIM, dtype=np.float32)
        for tok, c in counts.items():
            idx = _hash_idx(tok, _FALLBACK_DIM)
            sign = 1.0 if (hash(tok) & 1) == 0 else -1.0
            vec[idx] += sign * (1.0 + math.log(c))
        return _normalize(vec)


class SemanticIndex:
    """Hybrid retrieval over chain blocks."""

    def __init__(self, store: Store, embedder: Embedder | None = None):
        self.store = store
        self.embedder = embedder or Embedder()

    def _cached_embedding(self, block: Block) -> np.ndarray:
        cached = self.store.get_embedding(block.block_id)
        if cached is not None:
            model, vec = cached
            if model == self.embedder.model_name and vec.shape[0] == self.embedder.dim:
                return vec
        vec = self.embedder.encode([block.content])[0]
        self.store.put_embedding(block.block_id, self.embedder.model_name, vec)
        return vec

    def ensure_embedded(self, blocks: Iterable[Block]) -> None:
        missing: list[Block] = []
        for b in blocks:
            cached = self.store.get_embedding(b.block_id)
            if (
                cached is None
                or cached[0] != self.embedder.model_name
                or cached[1].shape[0] != self.embedder.dim
            ):
                missing.append(b)
        if not missing:
            return
        vecs = self.embedder.encode([b.content for b in missing])
        for b, v in zip(missing, vecs):
            self.store.put_embedding(b.block_id, self.embedder.model_name, v)

    def search(
        self,
        chain_id: str,
        prompt: str,
        *,
        lookback_limit: int,
        top_k: int = 10,
        mode: SearchMode = "hybrid",
        diversify: bool = False,
        roles: list[Role] | None = None,
        time_range: TimeRange | None = None,
        min_relevance: float = 0.0,
        explain: bool = False,
    ) -> tuple[list[Match], int]:
        prompt = prompt.strip()
        if not prompt:
            return ([], 0)

        candidates = self.store.list_chain_blocks(chain_id, limit=lookback_limit)
        filtered = _apply_filters(candidates, roles=roles, time_range=time_range)
        if not filtered:
            return ([], len(candidates))

        block_by_id = {b.block_id: b for b in filtered}
        valid_ids = set(block_by_id.keys())

        semantic_scores: dict[str, float] = {}
        if mode in ("semantic", "hybrid"):
            self.ensure_embedded(filtered)
            q = self.embedder.encode([prompt])[0]
            vectors = np.stack([self._cached_embedding(b) for b in filtered], axis=0)
            sims = vectors @ q
            sims_clamped = (sims + 1.0) / 2.0
            for b, s in zip(filtered, sims_clamped):
                semantic_scores[b.block_id] = float(s)

        lexical_scores: dict[str, float] = {}
        if mode in ("lexical", "hybrid"):
            raw = self.store.lexical_search(chain_id, prompt, limit=max(top_k * 4, 40))
            raw = [(b, s) for (b, s) in raw if b.block_id in valid_ids]
            if raw:
                worst = max(s for _, s in raw)
                for b, s in raw:
                    lexical_scores[b.block_id] = float(worst - s + 1e-3)

        sem_rank = _rank_dict(semantic_scores)
        lex_rank = _rank_dict(lexical_scores)
        rrf: dict[str, float] = {}
        for bid in valid_ids:
            score = 0.0
            if bid in sem_rank:
                score += 1.0 / (_RRF_K + sem_rank[bid])
            if bid in lex_rank:
                score += 1.0 / (_RRF_K + lex_rank[bid])
            if score > 0:
                rrf[bid] = score

        now = datetime.now(timezone.utc)
        recency_scores: dict[str, float] = {}
        for bid, b in block_by_id.items():
            age_days = max(0.0, (now - _ensure_aware(b.created_at)).total_seconds() / 86400.0)
            recency_scores[bid] = math.exp(-age_days / _RECENCY_TAU_DAYS)

        if not rrf:
            return ([], len(candidates))
        max_rrf = max(rrf.values())
        max_rec = max(recency_scores.values()) if recency_scores else 1.0
        fused: dict[str, float] = {}
        for bid in rrf:
            base = rrf[bid] / max_rrf if max_rrf else 0.0
            rec = (recency_scores.get(bid, 0.0) / max_rec) if max_rec else 0.0
            fused[bid] = max(0.0, min(1.0, 0.85 * base + 0.15 * rec))

        if diversify and mode in ("semantic", "hybrid") and semantic_scores:
            ordered_ids = _mmr(
                fused=fused,
                vectors={bid: self._cached_embedding(block_by_id[bid]) for bid in fused},
                k=top_k,
                lambda_=0.7,
            )
        else:
            ordered_ids = sorted(fused, key=lambda b: -fused[b])[:top_k]

        ordered_ids = [bid for bid in ordered_ids if fused[bid] >= min_relevance]

        results: list[Match] = []
        for bid in ordered_ids:
            b = block_by_id[bid]
            preview = b.content.strip().replace("\n", " ")
            if len(preview) > 220:
                preview = preview[:217] + "..."
            breakdown: ScoreBreakdown | None = None
            hint: str | None = None
            if explain:
                breakdown = ScoreBreakdown(
                    semantic=round(semantic_scores.get(bid, 0.0), 4),
                    lexical=round(lexical_scores.get(bid, 0.0), 4),
                    recency=round(recency_scores.get(bid, 0.0), 4),
                    fused=round(fused[bid], 4),
                )
                hint = _build_context_hint(b, prompt)
            results.append(
                Match(
                    block_id=b.block_id, chain_id=b.chain_id, sequence=b.sequence,
                    role=b.role, created_at=b.created_at,
                    relevance=round(fused[bid], 4), preview=preview,
                    score_breakdown=breakdown, context_hint=hint,
                )
            )
        return (results, len(candidates))


def _apply_filters(blocks: list[Block], *, roles: list[Role] | None, time_range: TimeRange | None) -> list[Block]:
    out = blocks
    if roles:
        role_set = set(roles)
        out = [b for b in out if b.role in role_set]
    if time_range is not None:
        if time_range.from_ is not None:
            lower = _ensure_aware(time_range.from_)
            out = [b for b in out if _ensure_aware(b.created_at) >= lower]
        if time_range.to is not None:
            upper = _ensure_aware(time_range.to)
            out = [b for b in out if _ensure_aware(b.created_at) <= upper]
    return out


def _rank_dict(scores: dict[str, float]) -> dict[str, int]:
    if not scores:
        return {}
    ordered = sorted(scores.items(), key=lambda kv: -kv[1])
    return {bid: i + 1 for i, (bid, _) in enumerate(ordered)}


def _ensure_aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def _mmr(*, fused: dict[str, float], vectors: dict[str, np.ndarray], k: int, lambda_: float) -> list[str]:
    remaining = list(fused.keys())
    chosen: list[str] = []
    while remaining and len(chosen) < k:
        best_id: str | None = None
        best_score = -math.inf
        for bid in remaining:
            rel = fused[bid]
            penalty = 0.0 if not chosen else max(float(np.dot(vectors[bid], vectors[o])) for o in chosen)
            score = lambda_ * rel - (1.0 - lambda_) * penalty
            if score > best_score:
                best_score = score
                best_id = bid
        if best_id is None:
            break
        chosen.append(best_id)
        remaining.remove(best_id)
    return chosen


def _build_context_hint(block: Block, prompt: str) -> str:
    p_terms = {t for t in _tokens(prompt) if len(t) > 2}
    b_terms = _tokens(block.content)
    hits = [t for t in dict.fromkeys(b_terms) if t in p_terms]
    lead = block.content.strip().splitlines()[0] if block.content.strip() else ""
    if len(lead) > 120:
        lead = lead[:117] + "..."
    if hits:
        return f"matches on: {', '.join(hits[:5])} — {lead}"
    return lead or "(no preview)"


# =============================================================================
# summarizer.py
# =============================================================================

VALID_CATEGORIES = {"decision", "question", "result", "transition", "blocker"}


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\n", "", text)
        text = re.sub(r"\n```$", "", text)
    return text.strip()


def _truncate_blocks(blocks: list[Any], *, max_blocks: int, max_chars: int) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for b in blocks[-max_blocks:]:
        content = b.content
        if len(content) > max_chars:
            content = content[:max_chars] + " ... [truncated]"
        payload.append({"block_id": b.block_id, "sequence": b.sequence, "role": b.role, "content": content})
    return payload


_SENT_RE = re.compile(r"[^.!?\n]+[.!?]?")


def _first_sentence(text: str, max_len: int = 200) -> str:
    text = text.strip().replace("\n", " ")
    if not text:
        return ""
    m = _SENT_RE.search(text)
    sentence = (m.group(0) if m else text).strip()
    if len(sentence) > max_len:
        sentence = sentence[: max_len - 1].rstrip() + "…"
    return sentence


def _estimate_tokens(text: str) -> int:
    return max(1, len(text.split()))


class Summarizer:
    """Wraps Claude (when available) with an extractive fallback."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings.from_env()
        self._client = None
        if self.settings.anthropic_api_key:
            try:
                import anthropic  # type: ignore
                self._client = anthropic.Anthropic(api_key=self.settings.anthropic_api_key)
            except Exception as exc:
                get_logger("summarizer").warning(f"event=anthropic.unavailable error={type(exc).__name__}")

    @property
    def using_claude(self) -> bool:
        return self._client is not None

    @property
    def engine_id(self) -> str:
        return f"claude-{self.settings.anthropic_model}" if self._client is not None else "extractive"

    def summarize(
        self,
        chain_id: str,
        session_id: str,
        blocks: list[Any],
        *,
        detail_level: DetailLevel = "standard",
        model_override: str | None = None,
    ) -> Summary:
        summary_id = str(uuid.uuid4())
        if not blocks:
            return Summary(
                summary_id=summary_id, chain_id=chain_id, session_id=session_id,
                plain_text="This chain has no activity yet.", detail_level=detail_level,
                engine=self.engine_id, token_count=0, block_count=0,
            )

        plain_text = ""
        themes: list[str] = []
        decisions: list[Decision] = []
        open_questions: list[str] = []
        cp_dicts: list[dict[str, Any]] = []
        engine = self.engine_id

        if self._client is not None:
            try:
                plain_text, themes, decisions, open_questions, cp_dicts = self._summarize_with_claude(
                    blocks, detail_level=detail_level, model_override=model_override,
                )
            except Exception as exc:
                get_logger("summarizer").warning(
                    f"event=claude.summarize.failed error={type(exc).__name__} "
                    f'message="{exc}" detail_level={detail_level}'
                )
                engine = "extractive"
                plain_text, themes, decisions, open_questions, cp_dicts = self._summarize_extractive(
                    blocks, detail_level=detail_level
                )
        else:
            plain_text, themes, decisions, open_questions, cp_dicts = self._summarize_extractive(
                blocks, detail_level=detail_level
            )

        checkpoints: list[Checkpoint] = []
        for i, cp in enumerate(cp_dicts):
            checkpoints.append(
                Checkpoint(
                    checkpoint_id=str(uuid.uuid4()), summary_id=summary_id,
                    block_id=cp["block_id"], chain_id=chain_id, position=i,
                    label=cp["label"], description=cp.get("description", ""),
                    reason=cp.get("reason"), confidence=cp.get("confidence"),
                    category=cp.get("category"),
                )
            )

        token_count = _estimate_tokens(plain_text) + sum(_estimate_tokens(b.content) for b in blocks)
        return Summary(
            summary_id=summary_id, chain_id=chain_id, session_id=session_id,
            plain_text=plain_text, detail_level=detail_level, engine=engine,
            token_count=token_count, block_count=len(blocks), themes=themes,
            decisions=decisions, open_questions=open_questions, checkpoints=checkpoints,
        )

    def _call_claude(self, *, system_prompt: str, user_payload: str, model_override: str | None, max_tokens: int = 2048) -> str:
        assert self._client is not None
        model = model_override or self.settings.anthropic_model
        start = time.perf_counter()
        resp = self._client.messages.create(
            model=model, max_tokens=max_tokens, system=system_prompt,
            messages=[{"role": "user", "content": user_payload}],
        )
        elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
        get_logger("summarizer").info(f"event=claude.call model={model} max_tokens={max_tokens} duration_ms={elapsed_ms}")
        return "".join(block.text for block in resp.content if getattr(block, "type", None) == "text")

    def _summarize_with_claude(
        self, blocks: list[Any], *, detail_level: DetailLevel, model_override: str | None,
    ) -> tuple[str, list[str], list[Decision], list[str], list[dict[str, Any]]]:
        payload = _truncate_blocks(blocks, max_blocks=self.settings.max_blocks_to_summarize, max_chars=self.settings.max_chars_per_block)
        valid_ids = {b["block_id"] for b in payload}

        if detail_level == "skim":
            text = self._call_claude(
                system_prompt=skim_prompt(),
                user_payload="Here is the chain in JSON. Produce the history JSON now.\n\n" + json.dumps(payload, ensure_ascii=False),
                model_override=model_override, max_tokens=1500,
            )
            data = self._parse_json(text)
            return (str(data.get("summary", "")).strip(), _coerce_str_list(data.get("themes")), [], [], _validate_checkpoints(data.get("checkpoints"), valid_ids))

        if detail_level == "standard":
            text = self._call_claude(
                system_prompt=standard_prompt(),
                user_payload="Here is the chain in JSON. Produce the history JSON now.\n\n" + json.dumps(payload, ensure_ascii=False),
                model_override=model_override, max_tokens=2500,
            )
            data = self._parse_json(text)
            return (
                str(data.get("summary", "")).strip(),
                _coerce_str_list(data.get("themes")),
                _validate_decisions(data.get("decisions"), valid_ids),
                _coerce_str_list(data.get("open_questions")),
                _validate_checkpoints(data.get("checkpoints"), valid_ids),
            )

        pass1_text = self._call_claude(
            system_prompt=deep_pass1_prompt(),
            user_payload="Here is the chain in JSON.\n\n" + json.dumps(payload, ensure_ascii=False),
            model_override=model_override, max_tokens=2500,
        )
        pass1 = self._parse_json(pass1_text)
        themes = _coerce_str_list(pass1.get("themes"))
        decisions = _validate_decisions(pass1.get("decisions"), valid_ids)
        open_q = _coerce_str_list(pass1.get("open_questions"))
        pass2_text = self._call_claude(
            system_prompt=deep_pass2_prompt(_strip_code_fence(pass1_text)),
            user_payload="Here is the chain in JSON. Produce the pass-2 JSON now.\n\n" + json.dumps(payload, ensure_ascii=False),
            model_override=model_override, max_tokens=3000,
        )
        pass2 = self._parse_json(pass2_text)
        return (str(pass2.get("summary", "")).strip(), themes, decisions, open_q, _validate_checkpoints(pass2.get("checkpoints"), valid_ids))

    def _parse_json(self, text: str) -> dict[str, Any]:
        cleaned = _strip_code_fence(text)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Claude returned non-JSON: {exc.msg}") from exc
        if not isinstance(data, dict):
            raise ValueError("Claude returned non-object JSON")
        return data

    def _summarize_extractive(
        self, blocks: list[Any], *, detail_level: DetailLevel,
    ) -> tuple[str, list[str], list[Decision], list[str], list[dict[str, Any]]]:
        turns: list[list[Any]] = []
        for b in blocks:
            if turns and turns[-1][-1].role == b.role:
                turns[-1].append(b)
            else:
                turns.append([b])

        first_user = next((b for b in blocks if b.role == "user"), blocks[0])
        opener = _first_sentence(first_user.content)
        paragraphs = [f"This session opened with you asking: “{opener}”."]

        notable = [t for t in turns if t[0].role in ("user", "agent")]
        mid_count = {"skim": 1, "standard": 4, "deep": 6}.get(detail_level, 4)
        mid = notable[1:-1][:mid_count] if len(notable) > 2 else []
        for turn in mid:
            head = turn[0]
            actor = "You" if head.role == "user" else "The agent"
            paragraphs.append(f"{actor} then: {_first_sentence(head.content)}")

        last = blocks[-1]
        tail_actor = "you" if last.role == "user" else "the agent"
        paragraphs.append(f"Most recently, {tail_actor}: {_first_sentence(last.content)}")
        summary_text = "\n\n".join(paragraphs)

        theme_cap = {"skim": 3, "standard": 6, "deep": 8}.get(detail_level, 6)
        themes = _extract_themes(blocks, cap=theme_cap)

        decisions: list[Decision] = []
        for b in blocks:
            if b.role != "agent":
                continue
            sent = _first_sentence(b.content).lower()
            if any(sent.startswith(v) for v in ("set ", "use ", "switch ", "move ", "add ", "install ", "drop ")):
                decisions.append(Decision(summary=_first_sentence(b.content), block_id=b.block_id))
            if len(decisions) >= 3:
                break

        open_questions: list[str] = []
        if last.role == "user" and last.content.strip().endswith("?"):
            open_questions.append(_first_sentence(last.content))

        cp_cap = {"skim": 5, "standard": 8, "deep": 12}.get(detail_level, 8)
        candidates = [turn[0] for turn in turns]
        if blocks[-1] is not candidates[-1]:
            candidates.append(blocks[-1])
        if len(candidates) > cp_cap:
            step = len(candidates) / cp_cap
            candidates = [candidates[int(i * step)] for i in range(cp_cap)]

        cp_dicts: list[dict[str, Any]] = []
        seen: set[str] = set()
        for b in candidates:
            if b.block_id in seen:
                continue
            seen.add(b.block_id)
            label = _first_sentence(b.content)
            if len(label) > 60:
                label = label[:57] + "..."
            actor = {"user": "You", "agent": "Agent", "tool": "Tool", "system": "System"}.get(b.role, b.role.capitalize())
            cp_dicts.append({
                "block_id": b.block_id,
                "label": label or f"{actor} turn",
                "description": f"{actor}: {_first_sentence(b.content) or '(empty block)'}",
                "reason": "turn boundary in the chain",
                "confidence": 0.5,
                "category": _guess_category(b),
            })
        return (summary_text, themes, decisions, open_questions, cp_dicts)


def _coerce_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [v.strip() for v in value if isinstance(v, str) and v.strip()]


def _validate_decisions(value: Any, valid_ids: set[str]) -> list[Decision]:
    if not isinstance(value, list):
        return []
    out: list[Decision] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        summary = str(item.get("summary", "")).strip()
        if not summary:
            continue
        block_id = item.get("block_id")
        if block_id is not None and block_id not in valid_ids:
            block_id = None
        out.append(Decision(summary=summary, block_id=block_id))
    return out


def _validate_checkpoints(value: Any, valid_ids: set[str]) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        block_id = item.get("block_id")
        if not isinstance(block_id, str) or block_id not in valid_ids:
            continue
        label = str(item.get("label", "Checkpoint"))[:60].strip() or "Checkpoint"
        description = str(item.get("description", "")).strip()
        reason = item.get("reason")
        reason = reason.strip() or None if isinstance(reason, str) else None
        try:
            confidence = float(item.get("confidence")) if item.get("confidence") is not None else None
            if confidence is not None:
                confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            confidence = None
        category = item.get("category")
        if not isinstance(category, str) or category not in VALID_CATEGORIES:
            category = None
        out.append({"block_id": block_id, "label": label, "description": description, "reason": reason, "confidence": confidence, "category": category})
    return out


_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "you", "your", "from", "into",
    "have", "are", "was", "were", "but", "not", "any", "all", "can", "out",
    "use", "via", "per", "let", "got", "now", "one", "two", "ill", "its", "ive",
    "they", "them", "their", "there", "then", "than", "what", "when", "which",
    "who", "how", "why", "where", "would", "could", "should", "about", "after",
    "before", "back", "here", "just", "like", "make", "more", "most", "much",
    "some", "such", "want", "well", "will", "yes", "your", "yourself",
    "im", "ive", "id", "ll", "didnt", "doesnt", "isnt",
}

_WORD_RE2 = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


def _extract_themes(blocks: list[Any], cap: int) -> list[str]:
    counts: Counter[str] = Counter()
    for b in blocks:
        for token in [t.lower() for t in _WORD_RE2.findall(b.content)]:
            if token not in _STOPWORDS and len(token) >= 4:
                counts[token] += 1
    return [w for w, _ in counts.most_common(cap)]


def _guess_category(block: Any) -> str:
    role = getattr(block, "role", "")
    text = (getattr(block, "content", "") or "").lower()
    if role == "user" and text.strip().endswith("?"):
        return "question"
    if role == "tool":
        return "result"
    if any(kw in text for kw in ("blocked", "stuck", "error", "fail")):
        return "blocker"
    if any(text.startswith(v) for v in ("set ", "use ", "switch ", "move ")):
        return "decision"
    return "transition"


# =============================================================================
# server.py  (FastMCP)
# =============================================================================

_log = get_logger("server")


@dataclass
class AppState:
    settings: Settings
    store: Store
    index: SemanticIndex
    summarizer: Summarizer


_APP_STATE: AppState | None = None


@asynccontextmanager
async def lifespan(_: FastMCP) -> AsyncIterator[AppState]:
    global _APP_STATE
    settings = Settings.from_env()
    configure_logging(settings.log_level)
    _log.info(
        f"event=server.start db={settings.db_path} "
        f"claude={settings.claude_enabled} embedding={settings.embedding_model}"
    )
    store = SQLiteStore(settings.db_path)
    state = AppState(
        settings=settings,
        store=store,
        index=SemanticIndex(store, embedder=Embedder(settings.embedding_model)),
        summarizer=Summarizer(settings),
    )
    _APP_STATE = state
    try:
        yield state
    finally:
        _APP_STATE = None
        store.close()
        _log.info("event=server.stop")


mcp = FastMCP("context-retrieval", lifespan=lifespan)


def _state(ctx: Context) -> AppState:
    return ctx.request_context.lifespan_context  # type: ignore[return-value]


def _state_for_resource() -> AppState:
    if _APP_STATE is None:
        raise RuntimeError("server lifespan has not initialized state yet")
    return _APP_STATE


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_session(state: AppState, session_id: str) -> Session:
    sess = state.store.get_session(session_id)
    if sess is None:
        sess = Session(session_id=session_id)
        state.store.upsert_session(sess)
    return sess


def _block_role_counts(blocks: list[Block]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for b in blocks:
        counts[b.role] = counts.get(b.role, 0) + 1
    return counts


def _estimate_block_tokens(content: str) -> int:
    return max(1, len(content.split()))


def _format_blocks_as_text(blocks: list[Block]) -> str:
    parts = []
    for b in blocks:
        header = f"[{b.sequence:>4} | {b.role}]"
        parts.append(f"{header} {b.content.strip()}")
    return "\n\n".join(parts)


async def _notify_resource_updated(ctx: Context, uri: str) -> None:
    try:
        await ctx.session.send_resource_updated(AnyUrl(uri))
    except Exception as exc:
        _log.debug(f"event=resource.notify.skipped uri={uri} error={exc!r}")


@mcp.tool(title="Query Chain", description="Hybrid semantic + lexical search across the blocks of a chain.")
def query_chain(
    chain_id: str, prompt: str, ctx: Context,
    lookback_limit: int | None = None, session_id: str | None = None,
    top_k: Annotated[int, Field(ge=1, le=100)] = 10,
    mode: SearchMode = "hybrid", diversify: bool = False,
    roles: list[Role] | None = None, time_range: TimeRange | None = None,
    min_relevance: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0,
    explain: bool = False,
) -> QueryResult:
    state = _state(ctx)
    limit = (
        lookback_limit if lookback_limit and lookback_limit > 0
        else state.store.get_effective_lookback(session_id or "", chain_id, default=state.settings.max_lookback)
    )
    limit = min(limit, state.settings.max_lookback)
    with log_call("query_chain", session=session_id, chain=chain_id, mode=mode, top_k=top_k) as fields:
        matches, total = state.index.search(
            chain_id=chain_id, prompt=prompt, lookback_limit=limit, top_k=top_k,
            mode=mode, diversify=diversify, roles=roles, time_range=time_range,
            min_relevance=min_relevance, explain=explain,
        )
        fields["matches"] = len(matches)
        fields["candidates"] = total
    return QueryResult(chain_id=chain_id, prompt=prompt, lookback_limit=limit, mode=mode, total_candidates=total, matches=matches)


@mcp.tool(title="Scan Adjacent Blocks", description="Fetch neighboring blocks around a match.")
def scan_adjacent(
    block_id: str, ctx: Context,
    window: Annotated[int, Field(ge=0, le=200)] = 1,
    before_window: int | None = None, after_window: int | None = None,
    roles: list[Role] | None = None,
    max_tokens: Annotated[int | None, Field(ge=1)] = None,
) -> AdjacentResult:
    state = _state(ctx)
    before = max(0, before_window if before_window is not None else window)
    after = max(0, after_window if after_window is not None else window)
    with log_call("scan_adjacent", block=block_id, before=before, after=after) as fields:
        anchor = state.store.get_block(block_id)
        if anchor is None:
            raise ValueError(f"unknown block_id: {block_id}")
        blocks = state.store.adjacent_blocks(block_id, before=before, after=after)
        if roles:
            role_set = set(roles)
            blocks = [b for b in blocks if b.role in role_set or b.block_id == block_id]
        total_chain_length = state.store.chain_block_count(anchor.chain_id)
        truncated = False
        if max_tokens is not None:
            blocks, dropped = _trim_to_token_budget(blocks, anchor_block_id=block_id, max_tokens=max_tokens)
            truncated = dropped > 0
            fields["dropped"] = dropped
        merged = _format_blocks_as_text(blocks)
        total_tokens = sum(_estimate_block_tokens(b.content) for b in blocks)
        fields["blocks"] = len(blocks)
    return AdjacentResult(block_id=block_id, before_window=before, after_window=after, blocks=blocks, merged_context=merged, total_tokens=total_tokens, total_chain_length=total_chain_length, truncated=truncated)


@mcp.tool(title="Set Lookback", description="Configure the maximum number of blocks to search back.")
def set_lookback(
    session_id: str, limit: int, ctx: Context,
    scope: Literal["session", "chain"] = "session", chain_id: str | None = None,
) -> LookbackResult:
    state = _state(ctx)
    if limit < 1 or limit > state.settings.max_lookback:
        raise ValueError(f"limit must be in [1, {state.settings.max_lookback}]; got {limit}")
    if scope == "chain" and not chain_id:
        raise ValueError("chain_id is required when scope='chain'")
    with log_call("set_lookback", session=session_id, scope=scope, chain=chain_id, limit=limit) as fields:
        if scope == "session":
            prev, updated = state.store.set_session_lookback(session_id, limit)
        else:
            assert chain_id is not None
            prev, updated = state.store.set_chain_lookback(session_id, chain_id, limit)
        fields["previous"] = prev
    return LookbackResult(session_id=session_id, scope=scope, chain_id=chain_id, previous_limit=prev, updated_limit=updated)


@mcp.tool(title="Summarize History", description="Convert the raw chain into a plain-language history with checkpoints.")
async def summarize_history(
    chain_id: str, session_id: str, ctx: Context,
    detail_level: DetailLevel = "standard", force_refresh: bool = False, model: str | None = None,
) -> dict[str, Any]:
    state = _state(ctx)
    with log_call("summarize_history", session=session_id, chain=chain_id, detail_level=detail_level, force_refresh=force_refresh) as fields:
        blocks = state.store.list_chain_blocks(chain_id)
        block_count = len(blocks)
        cached = state.store.latest_summary_for_chain(chain_id, detail_level=detail_level)
        if cached is not None and not force_refresh and cached.block_count == block_count:
            fields["cached"] = True
            return {
                "summary_id": cached.summary_id, "plain_text": cached.plain_text,
                "checkpoints": [cp.model_dump(mode="json") for cp in cached.checkpoints],
                "themes": cached.themes, "decisions": [d.model_dump(mode="json") for d in cached.decisions],
                "open_questions": cached.open_questions, "engine": cached.engine,
                "detail_level": cached.detail_level, "generated_at": cached.created_at.isoformat(),
                "cached": True, "block_count": cached.block_count,
            }
        summary = await asyncio.to_thread(state.summarizer.summarize, chain_id, session_id, blocks, detail_level=detail_level, model_override=model)
        state.store.insert_summary(summary)
        fields["cached"] = False
        fields["checkpoints"] = len(summary.checkpoints)
        fields["engine"] = summary.engine
        await _notify_resource_updated(ctx, f"retrieval://history/{session_id}")
        return {
            "summary_id": summary.summary_id, "plain_text": summary.plain_text,
            "checkpoints": [cp.model_dump(mode="json") for cp in summary.checkpoints],
            "themes": summary.themes, "decisions": [d.model_dump(mode="json") for d in summary.decisions],
            "open_questions": summary.open_questions, "engine": summary.engine,
            "detail_level": summary.detail_level, "generated_at": summary.created_at.isoformat(),
            "cached": False, "block_count": summary.block_count,
        }


@mcp.tool(title="Backtrack", description="Restore agent context to a specific historical point.")
def backtrack(
    session_id: str, ctx: Context,
    checkpoint_id: str | None = None, block_id: str | None = None,
    name: str | None = None, reason: str | None = None, dry_run: bool = False,
) -> BacktrackResult:
    state = _state(ctx)
    if not checkpoint_id and not block_id:
        raise ValueError("provide either checkpoint_id or block_id")
    with log_call("backtrack", session=session_id, checkpoint=checkpoint_id, block=block_id, dry_run=dry_run) as fields:
        if checkpoint_id is not None:
            cp = state.store.get_checkpoint(checkpoint_id)
            if cp is None:
                raise ValueError(f"unknown checkpoint_id: {checkpoint_id}")
            anchor = state.store.get_block(cp.block_id)
            chain_id = cp.chain_id
            anchor_label = cp.label
        else:
            assert block_id is not None
            anchor = state.store.get_block(block_id)
            if anchor is None:
                raise ValueError(f"unknown block_id: {block_id}")
            cp = None
            chain_id = anchor.chain_id
            anchor_label = f"block {anchor.sequence}"
        if anchor is None:
            raise ValueError("anchor block not found")
        prefix = state.store.list_chain_blocks(chain_id, up_to_sequence=anchor.sequence)
        sess = _ensure_session(state, session_id)
        role_counts = _block_role_counts(prefix)
        token_estimate = sum(_estimate_block_tokens(b.content) for b in prefix)
        chain = state.store.get_chain(chain_id)
        manifest = SnapshotManifest(block_count=len(prefix), token_estimate=token_estimate, role_counts=role_counts, chain_title=chain.title if chain else None)
        provenance = Provenance(created_by=session_id, reason=reason, dry_run=dry_run, name=name)
        snapshot_id: str | None = None
        if not dry_run:
            snapshot_id = str(uuid.uuid4())
            state.store.insert_snapshot(Snapshot(
                snapshot_id=snapshot_id, session_id=session_id, chain_id=chain_id,
                checkpoint_id=cp.checkpoint_id if cp else None,
                anchor_block_id=anchor.block_id, anchor_sequence=anchor.sequence,
                lookback_limit=sess.lookback_limit, manifest=manifest, provenance=provenance, blocks=prefix,
            ))
        fields["blocks"] = len(prefix)
        fields["tokens"] = token_estimate
        fields["snapshot"] = snapshot_id
        restored_summary = f"{'Would restore' if dry_run else 'Restored'} {len(prefix)} block(s) up to '{anchor_label}' (sequence {anchor.sequence})."
        return BacktrackResult(snapshot_id=snapshot_id, anchor_block_id=anchor.block_id, anchor_sequence=anchor.sequence, block_count=len(prefix), estimated_tokens=token_estimate, role_counts=role_counts, restored_summary=restored_summary, dry_run=dry_run)


@mcp.tool(title="Reclaim Context", description="Re-inject a snapshot's historical context into the Active Agent.")
async def reclaim_context(
    session_id: str, snapshot_id: str, ctx: Context,
    token_budget: Annotated[int | None, Field(ge=1)] = None,
    roles: list[Role] | None = None, compress_if_over: bool = True,
) -> ReclaimResult:
    state = _state(ctx)
    with log_call("reclaim_context", session=session_id, snapshot=snapshot_id, budget=token_budget) as fields:
        snap = state.store.get_snapshot(snapshot_id)
        if snap is None:
            raise ValueError(f"unknown snapshot_id: {snapshot_id}")
        if snap.session_id != session_id:
            raise ValueError(f"snapshot {snapshot_id} belongs to session {snap.session_id}, not {session_id}")
        blocks = list(snap.blocks)
        if roles:
            role_set = set(roles)
            blocks = [b for b in blocks if b.role in role_set]
        skipped: list[Block] = []
        if token_budget is not None:
            blocks, skipped = _trim_to_token_budget_pair(blocks, max_tokens=token_budget)
        compressed = False
        compression_summary: str | None = None
        if skipped and compress_if_over:
            compress_sum = await asyncio.to_thread(state.summarizer.summarize, snap.chain_id, session_id, skipped, detail_level="skim")
            compression_summary = compress_sum.plain_text
            compressed = True
        tokens_injected = sum(_estimate_block_tokens(b.content) for b in blocks)
        if compression_summary:
            tokens_injected += _estimate_block_tokens(compression_summary)
        sess = _ensure_session(state, session_id)
        sess.active_chain_id = snap.chain_id
        state.store.upsert_session(sess)
        fields["blocks_injected"] = len(blocks)
        fields["blocks_skipped"] = len(skipped)
        fields["compressed"] = compressed
        return ReclaimResult(success=True, snapshot_id=snapshot_id, tokens_injected=tokens_injected, blocks_injected=len(blocks), blocks_skipped=len(skipped), compressed=compressed, compression_summary=compression_summary, active_chain_id=snap.chain_id)


@mcp.resource("retrieval://history/{session_id}", title="Plain-language History", mime_type="application/json")
def history_resource(session_id: str) -> str:
    state = _state_for_resource()
    summary = state.store.latest_summary_for_session(session_id)
    if summary is None:
        return json.dumps({"session_id": session_id, "summary": None}, ensure_ascii=False)
    payload = summary.model_dump(mode="json")
    payload["session_id"] = session_id
    return json.dumps(payload, ensure_ascii=False, indent=2)


@mcp.resource("retrieval://snapshot/{point_id}", title="Frozen Agent Snapshot", mime_type="application/json")
def snapshot_resource(point_id: str) -> str:
    state = _state_for_resource()
    snap = state.store.get_snapshot(point_id)
    if snap is None:
        return json.dumps({"snapshot_id": point_id, "error": "not_found"}, ensure_ascii=False)
    return json.dumps(snap.model_dump(mode="json"), ensure_ascii=False, indent=2)


@mcp.prompt(title="Recap History", description="Generate a plain-language recap of the session.")
def recap_history(session_id: str, detail_level: DetailLevel = "standard") -> list[Message]:
    return [UserMessage(
        "Recap what's happened in this session for me.\n\n"
        f"1. Call `summarize_history(chain_id=<active chain for session '{session_id}'>, session_id='{session_id}', detail_level='{detail_level}')`.\n"
        "2. Tell me the story in 4-6 short paragraphs.\n"
        "3. End with a numbered list of checkpoints I can click to backtrack."
    )]


@mcp.prompt(title="Find Related Context", description="Search the chain for context related to a topic.")
def find_related_context(chain_id: str, topic: str) -> list[Message]:
    return [UserMessage(
        f"I want context about \"{topic}\" from chain {chain_id}.\n\n"
        f"1. Call `query_chain(chain_id='{chain_id}', prompt='{topic}', mode='hybrid', explain=True, diversify=True, top_k=5)`.\n"
        "2. For each match, summarize in 1-2 sentences and cite the sequence and context_hint.\n"
        "3. Suggest whether I should backtrack to one of them."
    )]


@mcp.prompt(title="Restore to Decision", description="Find a key decision moment and backtrack there.")
def restore_to_decision(session_id: str, decision_query: str) -> list[Message]:
    return [UserMessage(
        f"I want to backtrack to when we decided about \"{decision_query}\".\n\n"
        f"1. Call `summarize_history(session_id='{session_id}', detail_level='deep')`.\n"
        f"2. Find the checkpoint matching \"{decision_query}\".\n"
        "3. Call `backtrack(..., dry_run=True)` and show me the manifest. Wait for confirmation.\n"
        "4. On confirmation, call `backtrack` without dry_run, then `reclaim_context(..., compress_if_over=True)`."
    )]


def _trim_to_token_budget(blocks: list[Block], *, anchor_block_id: str, max_tokens: int) -> tuple[list[Block], int]:
    kept = list(blocks)
    total = sum(_estimate_block_tokens(b.content) for b in kept)
    dropped = 0
    while kept and total > max_tokens:
        if len(kept) == 1:
            break
        if kept[0].block_id == anchor_block_id:
            gone = kept.pop()
        else:
            gone = kept.pop(0)
        total -= _estimate_block_tokens(gone.content)
        dropped += 1
    return kept, dropped


def _trim_to_token_budget_pair(blocks: list[Block], *, max_tokens: int) -> tuple[list[Block], list[Block]]:
    kept = list(blocks)
    total = sum(_estimate_block_tokens(b.content) for b in kept)
    dropped: list[Block] = []
    while kept and total > max_tokens:
        gone = kept.pop(0)
        dropped.append(gone)
        total -= _estimate_block_tokens(gone.content)
    return kept, dropped


# =============================================================================
# web.py  (FastAPI HTTP bridge — optional, requires fastapi + uvicorn)
# =============================================================================

def create_app(settings: Settings | None = None):
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import FileResponse, JSONResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError:
        raise ImportError("Install web extras: pip install 'mcp-context-retrieval[web]'")

    settings = settings or Settings.from_env()
    configure_logging(settings.log_level)
    get_logger("web").info(f"event=web.start db={settings.db_path}")

    store = SQLiteStore(settings.db_path)
    search_index = SemanticIndex(store, embedder=Embedder(settings.embedding_model))
    summarizer = Summarizer(settings)

    app = FastAPI(title="Context Retrieval Demo", version="0.1.0")

    class QueryBody(BaseModel):
        chain_id: str; prompt: str; session_id: str | None = None
        lookback_limit: int | None = None
        top_k: int = Field(default=10, ge=1, le=100)
        mode: SearchMode = "hybrid"; diversify: bool = False
        roles: list[Role] | None = None; time_range: TimeRange | None = None
        min_relevance: float = Field(default=0.0, ge=0.0, le=1.0); explain: bool = True

    class AdjacentBody(BaseModel):
        block_id: str; window: int = Field(default=1, ge=0, le=200)
        before_window: int | None = None; after_window: int | None = None
        roles: list[Role] | None = None; max_tokens: int | None = Field(default=None, ge=1)

    class LookbackBody(BaseModel):
        session_id: str; limit: int = Field(ge=1, le=10_000)
        scope: Literal["session", "chain"] = "session"; chain_id: str | None = None

    class SummarizeBody(BaseModel):
        chain_id: str; session_id: str; detail_level: DetailLevel = "standard"
        force_refresh: bool = False; model: str | None = None

    class BacktrackBody(BaseModel):
        session_id: str; checkpoint_id: str | None = None; block_id: str | None = None
        name: str | None = None; reason: str | None = None; dry_run: bool = False

    class ReclaimBody(BaseModel):
        session_id: str; snapshot_id: str
        token_budget: int | None = Field(default=None, ge=1)
        roles: list[Role] | None = None; compress_if_over: bool = True

    @app.post("/api/query", response_model=QueryResult)
    def query(body: QueryBody) -> QueryResult:
        limit = body.lookback_limit if body.lookback_limit and body.lookback_limit > 0 else store.get_effective_lookback(body.session_id or "", body.chain_id, default=settings.max_lookback)
        limit = min(limit, settings.max_lookback)
        matches, total = search_index.search(chain_id=body.chain_id, prompt=body.prompt, lookback_limit=limit, top_k=body.top_k, mode=body.mode, diversify=body.diversify, roles=body.roles, time_range=body.time_range, min_relevance=body.min_relevance, explain=body.explain)
        return QueryResult(chain_id=body.chain_id, prompt=body.prompt, lookback_limit=limit, mode=body.mode, total_candidates=total, matches=matches)

    @app.post("/api/adjacent", response_model=AdjacentResult)
    def adjacent(body: AdjacentBody) -> AdjacentResult:
        anchor = store.get_block(body.block_id)
        if anchor is None:
            raise HTTPException(404, f"unknown block_id: {body.block_id}")
        before = body.before_window if body.before_window is not None else body.window
        after = body.after_window if body.after_window is not None else body.window
        blocks = store.adjacent_blocks(body.block_id, before=before, after=after)
        if body.roles:
            role_set = set(body.roles)
            blocks = [b for b in blocks if b.role in role_set or b.block_id == body.block_id]
        total_chain_length = store.chain_block_count(anchor.chain_id)
        truncated = False
        if body.max_tokens is not None:
            kept = list(blocks)
            total = sum(_estimate_block_tokens(b.content) for b in kept)
            while kept and total > body.max_tokens and len(kept) > 1:
                gone = kept.pop() if kept[0].block_id == body.block_id else kept.pop(0)
                total -= _estimate_block_tokens(gone.content)
                truncated = True
            blocks = kept
        merged = _format_blocks_as_text(blocks)
        return AdjacentResult(block_id=body.block_id, before_window=before, after_window=after, blocks=blocks, merged_context=merged, total_tokens=sum(_estimate_block_tokens(b.content) for b in blocks), total_chain_length=total_chain_length, truncated=truncated)

    @app.post("/api/lookback", response_model=LookbackResult)
    def lookback(body: LookbackBody) -> LookbackResult:
        if body.limit > settings.max_lookback:
            raise HTTPException(400, f"limit must be in [1, {settings.max_lookback}]; got {body.limit}")
        if body.scope == "chain" and not body.chain_id:
            raise HTTPException(400, "chain_id is required when scope='chain'")
        if body.scope == "session":
            prev, upd = store.set_session_lookback(body.session_id, body.limit)
        else:
            assert body.chain_id is not None
            prev, upd = store.set_chain_lookback(body.session_id, body.chain_id, body.limit)
        return LookbackResult(session_id=body.session_id, scope=body.scope, chain_id=body.chain_id, previous_limit=prev, updated_limit=upd)

    @app.post("/api/summarize")
    def summarize(body: SummarizeBody) -> dict[str, Any]:
        blocks = store.list_chain_blocks(body.chain_id)
        cached = store.latest_summary_for_chain(body.chain_id, detail_level=body.detail_level)
        if cached is not None and not body.force_refresh and cached.block_count == len(blocks):
            return {"summary_id": cached.summary_id, "plain_text": cached.plain_text, "checkpoints": [cp.model_dump(mode="json") for cp in cached.checkpoints], "themes": cached.themes, "decisions": [d.model_dump(mode="json") for d in cached.decisions], "open_questions": cached.open_questions, "engine": cached.engine, "detail_level": cached.detail_level, "generated_at": cached.created_at.isoformat(), "cached": True, "block_count": cached.block_count}
        summary = summarizer.summarize(body.chain_id, body.session_id, blocks, detail_level=body.detail_level, model_override=body.model)
        store.insert_summary(summary)
        return {"summary_id": summary.summary_id, "plain_text": summary.plain_text, "checkpoints": [cp.model_dump(mode="json") for cp in summary.checkpoints], "themes": summary.themes, "decisions": [d.model_dump(mode="json") for d in summary.decisions], "open_questions": summary.open_questions, "engine": summary.engine, "detail_level": summary.detail_level, "generated_at": summary.created_at.isoformat(), "cached": False, "block_count": summary.block_count}

    @app.post("/api/backtrack", response_model=BacktrackResult)
    def backtrack_web(body: BacktrackBody) -> BacktrackResult:
        if not body.checkpoint_id and not body.block_id:
            raise HTTPException(400, "provide either checkpoint_id or block_id")
        if body.checkpoint_id is not None:
            cp = store.get_checkpoint(body.checkpoint_id)
            if cp is None:
                raise HTTPException(404, f"unknown checkpoint_id: {body.checkpoint_id}")
            anchor = store.get_block(cp.block_id); chain_id = cp.chain_id; anchor_label = cp.label
        else:
            assert body.block_id is not None
            anchor = store.get_block(body.block_id)
            if anchor is None:
                raise HTTPException(404, f"unknown block_id: {body.block_id}")
            cp = None; chain_id = anchor.chain_id; anchor_label = f"block {anchor.sequence}"
        if anchor is None:
            raise HTTPException(404, "anchor block not found")
        prefix = store.list_chain_blocks(chain_id, up_to_sequence=anchor.sequence)
        sess = store.get_session(body.session_id) or Session(session_id=body.session_id)
        if store.get_session(body.session_id) is None:
            store.upsert_session(sess)
        role_counts = _block_role_counts(prefix)
        token_estimate = sum(_estimate_block_tokens(b.content) for b in prefix)
        chain = store.get_chain(chain_id)
        manifest = SnapshotManifest(block_count=len(prefix), token_estimate=token_estimate, role_counts=role_counts, chain_title=chain.title if chain else None)
        prov = Provenance(created_by=body.session_id, reason=body.reason, dry_run=body.dry_run, name=body.name)
        snapshot_id: str | None = None
        if not body.dry_run:
            snapshot_id = str(uuid.uuid4())
            store.insert_snapshot(Snapshot(snapshot_id=snapshot_id, session_id=body.session_id, chain_id=chain_id, checkpoint_id=cp.checkpoint_id if cp else None, anchor_block_id=anchor.block_id, anchor_sequence=anchor.sequence, lookback_limit=sess.lookback_limit, manifest=manifest, provenance=prov, blocks=prefix))
        verb = "Would restore" if body.dry_run else "Restored"
        return BacktrackResult(snapshot_id=snapshot_id, anchor_block_id=anchor.block_id, anchor_sequence=anchor.sequence, block_count=len(prefix), estimated_tokens=token_estimate, role_counts=role_counts, restored_summary=f"{verb} {len(prefix)} block(s) up to '{anchor_label}' (sequence {anchor.sequence}).", dry_run=body.dry_run)

    @app.post("/api/reclaim", response_model=ReclaimResult)
    def reclaim(body: ReclaimBody) -> ReclaimResult:
        snap = store.get_snapshot(body.snapshot_id)
        if snap is None:
            raise HTTPException(404, f"unknown snapshot_id: {body.snapshot_id}")
        if snap.session_id != body.session_id:
            raise HTTPException(400, f"snapshot {body.snapshot_id} belongs to session {snap.session_id}, not {body.session_id}")
        blocks = list(snap.blocks)
        if body.roles:
            role_set = set(body.roles); blocks = [b for b in blocks if b.role in role_set]
        skipped: list[Block] = []
        if body.token_budget is not None:
            total = sum(_estimate_block_tokens(b.content) for b in blocks)
            while blocks and total > body.token_budget:
                gone = blocks.pop(0); skipped.append(gone); total -= _estimate_block_tokens(gone.content)
        compression_summary: str | None = None; compressed = False
        if skipped and body.compress_if_over:
            comp = summarizer.summarize(snap.chain_id, body.session_id, skipped, detail_level="skim")
            compression_summary = comp.plain_text; compressed = True
        tokens_injected = sum(_estimate_block_tokens(b.content) for b in blocks)
        if compression_summary:
            tokens_injected += _estimate_block_tokens(compression_summary)
        sess = store.get_session(body.session_id) or Session(session_id=body.session_id)
        sess.active_chain_id = snap.chain_id; store.upsert_session(sess)
        return ReclaimResult(success=True, snapshot_id=body.snapshot_id, tokens_injected=tokens_injected, blocks_injected=len(blocks), blocks_skipped=len(skipped), compressed=compressed, compression_summary=compression_summary, active_chain_id=snap.chain_id)

    @app.get("/api/sessions")
    def list_sessions() -> list[dict[str, Any]]:
        with store._lock:  # type: ignore[attr-defined]
            rows = store._conn.execute("SELECT session_id, active_chain_id, lookback_limit, created_at FROM sessions").fetchall()  # type: ignore[attr-defined]
        return [dict(r) for r in rows]

    @app.get("/api/chains")
    def list_chains() -> list[dict[str, Any]]:
        with store._lock:  # type: ignore[attr-defined]
            rows = store._conn.execute("SELECT chain_id, session_id, title, created_at FROM chains ORDER BY created_at DESC").fetchall()  # type: ignore[attr-defined]
        return [dict(r) for r in rows]

    @app.get("/api/chain/{chain_id}/blocks", response_model=list[Block])
    def list_blocks(chain_id: str) -> list[Block]:
        return store.list_chain_blocks(chain_id)

    @app.get("/api/history/{session_id}")
    def history_res(session_id: str) -> dict[str, Any]:
        summary = store.latest_summary_for_session(session_id)
        if summary is None:
            return {"session_id": session_id, "summary": None}
        payload = summary.model_dump(mode="json"); payload["session_id"] = session_id
        return payload

    @app.get("/api/snapshot/{point_id}")
    def snapshot_res(point_id: str) -> dict[str, Any]:
        snap = store.get_snapshot(point_id)
        if snap is None:
            raise HTTPException(404, "snapshot not found")
        return snap.model_dump(mode="json")

    @app.get("/api/snapshots/{session_id}")
    def snapshots_for_session(session_id: str) -> list[dict[str, Any]]:
        out = []
        for s in store.list_snapshots(session_id):
            d = s.model_dump(mode="json"); d.pop("blocks", None); out.append(d)
        return out

    @app.get("/api/prompts")
    def prompts_metadata() -> list[dict[str, Any]]:
        return [
            {"name": "recap_history", "title": "Recap History", "description": "One-click recap of the session.", "params": [{"name": "session_id", "kind": "session"}, {"name": "detail_level", "kind": "enum", "options": ["skim", "standard", "deep"]}]},
            {"name": "find_related_context", "title": "Find Related Context", "description": "Hybrid search a topic and explain the top matches.", "params": [{"name": "chain_id", "kind": "chain"}, {"name": "topic", "kind": "text"}]},
            {"name": "restore_to_decision", "title": "Restore to Decision", "description": "Find a decision moment and walk through backtrack -> reclaim_context.", "params": [{"name": "session_id", "kind": "session"}, {"name": "decision_query", "kind": "text"}]},
        ]

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "claude_enabled": settings.claude_enabled, "engine": "claude" if settings.claude_enabled else "extractive", "embedding_model": settings.embedding_model, "db_path": str(settings.db_path), "timestamp": _utcnow_iso()}

    web_dir = Path(__file__).resolve().parent.parent / "web"
    if web_dir.exists():
        app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(web_dir / "index.html")
    else:
        @app.get("/")
        def index_missing() -> JSONResponse:
            return JSONResponse({"error": f"web/ not found at {web_dir}"}, status_code=500)

    return app


# =============================================================================
# __main__.py  (CLI entrypoint)
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="MCP Server 3: Context Retrieval")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="MCP transport (default: stdio)",
    )
    args = parser.parse_args()
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
