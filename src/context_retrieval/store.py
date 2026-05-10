"""Storage layer.

`Store` is a `typing.Protocol` so we can swap SQLite for Postgres/Redis
later without touching tool code. `SQLiteStore` is the concrete
implementation used by the hackathon build.

The schema is auto-created on init. Tables fall into two groups:

    Shared (written by Servers 1 & 2, read by us):
        chains, sessions, blocks, blocks_fts (FTS5 mirror of blocks.content)

    Owned by Server 3 (we write & read):
        block_embeddings, summaries, summary_themes, summary_decisions,
        summary_open_questions, checkpoints, snapshots

We use WAL mode (concurrent reads while we write), foreign_keys=ON, and
a single connection guarded by an RLock since FastMCP can fire tools in
parallel from a single client.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Protocol

import numpy as np

from .models import (
    Block,
    Chain,
    Checkpoint,
    Decision,
    Provenance,
    Session,
    Snapshot,
    SnapshotManifest,
    Summary,
)


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

-- Per-chain lookback override. Optional companion to the session-level
-- default; consulted first when present.
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

-- FTS5 mirror of blocks.content. Triggers keep it in sync. Used by the
-- lexical leg of hybrid search.
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


# ----------------------------------------------------------------- helpers


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


# ----------------------------------------------------------- Store Protocol


class Store(Protocol):
    """Storage interface used by Server 3. SQLite is the default impl."""

    # ---- chains
    def upsert_chain(self, chain: Chain) -> None: ...
    def get_chain(self, chain_id: str) -> Chain | None: ...

    # ---- sessions
    def upsert_session(self, session: Session) -> None: ...
    def get_session(self, session_id: str) -> Session | None: ...
    def set_session_lookback(self, session_id: str, limit: int) -> tuple[int, int]: ...
    def set_chain_lookback(
        self, session_id: str, chain_id: str, limit: int
    ) -> tuple[int, int]: ...
    def get_effective_lookback(
        self, session_id: str, chain_id: str | None, default: int
    ) -> int: ...

    # ---- blocks
    def upsert_block(self, block: Block) -> None: ...
    def get_block(self, block_id: str) -> Block | None: ...
    def list_chain_blocks(
        self,
        chain_id: str,
        *,
        limit: int | None = None,
        up_to_sequence: int | None = None,
    ) -> list[Block]: ...
    def adjacent_blocks(
        self, block_id: str, *, before: int, after: int
    ) -> list[Block]: ...
    def chain_block_count(self, chain_id: str) -> int: ...
    def lexical_search(
        self, chain_id: str, query: str, *, limit: int
    ) -> list[tuple[Block, float]]: ...

    # ---- embeddings
    def get_embedding(self, block_id: str) -> tuple[str, np.ndarray] | None: ...
    def put_embedding(
        self, block_id: str, model: str, vector: np.ndarray
    ) -> None: ...

    # ---- summaries / checkpoints
    def insert_summary(self, summary: Summary) -> None: ...
    def latest_summary_for_session(self, session_id: str) -> Summary | None: ...
    def latest_summary_for_chain(
        self, chain_id: str, detail_level: str | None = None
    ) -> Summary | None: ...
    def get_checkpoint(self, checkpoint_id: str) -> Checkpoint | None: ...

    # ---- snapshots
    def insert_snapshot(self, snapshot: Snapshot) -> None: ...
    def get_snapshot(self, snapshot_id: str) -> Snapshot | None: ...
    def list_snapshots(self, session_id: str) -> list[Snapshot]: ...


# ----------------------------------------------------------- SQLite impl


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

    # ---------------------------------------------------------------- chains

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

    # -------------------------------------------------------------- sessions

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
        """Return (previous_limit, updated_limit). Creates the session if missing."""
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

    def set_chain_lookback(
        self, session_id: str, chain_id: str, limit: int
    ) -> tuple[int, int]:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT lookback FROM chain_lookbacks
                WHERE session_id = ? AND chain_id = ?
                """,
                (session_id, chain_id),
            ).fetchone()
        previous = row["lookback"] if row else self.get_effective_lookback(
            session_id, chain_id, 50
        )
        with self._tx() as cx:
            cx.execute(
                """
                INSERT INTO chain_lookbacks (session_id, chain_id, lookback)
                VALUES (?, ?, ?)
                ON CONFLICT(session_id, chain_id) DO UPDATE SET
                    lookback = excluded.lookback
                """,
                (session_id, chain_id, limit),
            )
        return (previous, limit)

    def get_effective_lookback(
        self, session_id: str, chain_id: str | None, default: int
    ) -> int:
        if chain_id is not None:
            with self._lock:
                row = self._conn.execute(
                    """
                    SELECT lookback FROM chain_lookbacks
                    WHERE session_id = ? AND chain_id = ?
                    """,
                    (session_id, chain_id),
                ).fetchone()
            if row is not None:
                return int(row["lookback"])
        sess = self.get_session(session_id)
        return sess.lookback_limit if sess else default

    # ----------------------------------------------------------------- blocks

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
                    block.block_id,
                    block.chain_id,
                    block.sequence,
                    block.role,
                    block.content,
                    json.dumps(block.metadata),
                    _iso(block.created_at),
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
            # Lookback caps the *tail* of the chain — the most recent N blocks.
            blocks = blocks[-limit:]
        return blocks

    def chain_block_count(self, chain_id: str) -> int:
        with self._lock:
            (count,) = self._conn.execute(
                "SELECT COUNT(*) FROM blocks WHERE chain_id = ?", (chain_id,)
            ).fetchone()
        return int(count)

    def adjacent_blocks(
        self, block_id: str, *, before: int, after: int
    ) -> list[Block]:
        anchor = self.get_block(block_id)
        if anchor is None:
            return []
        before = max(0, before)
        after = max(0, after)
        lo = max(0, anchor.sequence - before)
        hi = anchor.sequence + after
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM blocks
                WHERE chain_id = ? AND sequence BETWEEN ? AND ?
                ORDER BY sequence ASC
                """,
                (anchor.chain_id, lo, hi),
            ).fetchall()
        return [_row_to_block(r) for r in rows]

    def lexical_search(
        self, chain_id: str, query: str, *, limit: int
    ) -> list[tuple[Block, float]]:
        """FTS5 BM25 search restricted to one chain. Lower bm25() = better match."""
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
                # Bad FTS5 query syntax (e.g. user typed something weird).
                # Degrade gracefully to "no matches" instead of failing search.
                return []
        return [(_row_to_block(r), float(r["score"])) for r in rows]

    # -------------------------------------------------------------- embeddings

    def get_embedding(self, block_id: str) -> tuple[str, np.ndarray] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT model, dim, vector FROM block_embeddings WHERE block_id = ?",
                (block_id,),
            ).fetchone()
        if row is None:
            return None
        vec = np.frombuffer(row["vector"], dtype=np.float32).reshape(row["dim"])
        # frombuffer returns a read-only view; return a writable copy so
        # callers can do in-place math without surprises.
        return row["model"], np.array(vec, copy=True)

    def put_embedding(
        self, block_id: str, model: str, vector: np.ndarray
    ) -> None:
        vec = np.ascontiguousarray(vector, dtype=np.float32)
        with self._tx() as cx:
            cx.execute(
                """
                INSERT INTO block_embeddings (block_id, model, dim, vector)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(block_id) DO UPDATE SET
                    model  = excluded.model,
                    dim    = excluded.dim,
                    vector = excluded.vector
                """,
                (block_id, model, vec.shape[0], vec.tobytes()),
            )

    # --------------------------------------------------------------- summaries

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
                    summary.summary_id,
                    summary.chain_id,
                    summary.session_id,
                    summary.plain_text,
                    summary.detail_level,
                    summary.engine,
                    summary.token_count,
                    summary.block_count,
                    _iso(summary.created_at),
                ),
            )
            for i, theme in enumerate(summary.themes):
                cx.execute(
                    "INSERT INTO summary_themes (summary_id, position, theme) VALUES (?, ?, ?)",
                    (summary.summary_id, i, theme),
                )
            for i, dec in enumerate(summary.decisions):
                cx.execute(
                    """
                    INSERT INTO summary_decisions
                        (summary_id, position, summary, block_id)
                    VALUES (?, ?, ?, ?)
                    """,
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
                        cp.checkpoint_id,
                        cp.summary_id,
                        cp.block_id,
                        cp.chain_id,
                        cp.position,
                        cp.label,
                        cp.description,
                        cp.reason,
                        cp.confidence,
                        cp.category,
                    ),
                )

    def _hydrate_summary(self, row: sqlite3.Row) -> Summary:
        summary_id = row["summary_id"]
        with self._lock:
            themes = [
                r["theme"]
                for r in self._conn.execute(
                    "SELECT theme FROM summary_themes WHERE summary_id = ? ORDER BY position",
                    (summary_id,),
                ).fetchall()
            ]
            decisions = [
                Decision(summary=r["summary"], block_id=r["block_id"])
                for r in self._conn.execute(
                    "SELECT summary, block_id FROM summary_decisions WHERE summary_id = ? ORDER BY position",
                    (summary_id,),
                ).fetchall()
            ]
            open_questions = [
                r["question"]
                for r in self._conn.execute(
                    "SELECT question FROM summary_open_questions WHERE summary_id = ? ORDER BY position",
                    (summary_id,),
                ).fetchall()
            ]
            cp_rows = self._conn.execute(
                """
                SELECT * FROM checkpoints
                WHERE summary_id = ?
                ORDER BY position ASC
                """,
                (summary_id,),
            ).fetchall()
        checkpoints = [
            Checkpoint(
                checkpoint_id=r["checkpoint_id"],
                summary_id=r["summary_id"],
                block_id=r["block_id"],
                chain_id=r["chain_id"],
                position=r["position"],
                label=r["label"],
                description=r["description"],
                reason=r["reason"],
                confidence=r["confidence"],
                category=r["category"],
            )
            for r in cp_rows
        ]
        return Summary(
            summary_id=row["summary_id"],
            chain_id=row["chain_id"],
            session_id=row["session_id"],
            plain_text=row["plain_text"],
            detail_level=row["detail_level"],
            engine=row["engine"],
            token_count=row["token_count"],
            block_count=row["block_count"],
            themes=themes,
            decisions=decisions,
            open_questions=open_questions,
            checkpoints=checkpoints,
            created_at=_parse_iso(row["created_at"]),
        )

    def latest_summary_for_session(self, session_id: str) -> Summary | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT * FROM summaries
                WHERE session_id = ?
                ORDER BY created_at DESC LIMIT 1
                """,
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return self._hydrate_summary(row)

    def latest_summary_for_chain(
        self, chain_id: str, detail_level: str | None = None
    ) -> Summary | None:
        sql = "SELECT * FROM summaries WHERE chain_id = ?"
        params: list[Any] = [chain_id]
        if detail_level is not None:
            sql += " AND detail_level = ?"
            params.append(detail_level)
        sql += " ORDER BY created_at DESC LIMIT 1"
        with self._lock:
            row = self._conn.execute(sql, params).fetchone()
        if row is None:
            return None
        return self._hydrate_summary(row)

    def get_checkpoint(self, checkpoint_id: str) -> Checkpoint | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM checkpoints WHERE checkpoint_id = ?",
                (checkpoint_id,),
            ).fetchone()
        if row is None:
            return None
        return Checkpoint(
            checkpoint_id=row["checkpoint_id"],
            summary_id=row["summary_id"],
            block_id=row["block_id"],
            chain_id=row["chain_id"],
            position=row["position"],
            label=row["label"],
            description=row["description"],
            reason=row["reason"],
            confidence=row["confidence"],
            category=row["category"],
        )

    # ---------------------------------------------------------------- snapshots

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
                    snapshot.snapshot_id,
                    snapshot.session_id,
                    snapshot.chain_id,
                    snapshot.checkpoint_id,
                    snapshot.anchor_block_id,
                    snapshot.anchor_sequence,
                    snapshot.lookback_limit,
                    snapshot.manifest.model_dump_json(),
                    snapshot.provenance.model_dump_json(),
                    json.dumps(
                        [b.model_dump(mode="json") for b in snapshot.blocks]
                    ),
                    _iso(snapshot.created_at),
                ),
            )

    def get_snapshot(self, snapshot_id: str) -> Snapshot | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM snapshots WHERE snapshot_id = ?", (snapshot_id,)
            ).fetchone()
        if row is None:
            return None
        return _row_to_snapshot(row)

    def list_snapshots(self, session_id: str) -> list[Snapshot]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM snapshots
                WHERE session_id = ?
                ORDER BY created_at DESC
                """,
                (session_id,),
            ).fetchall()
        return [_row_to_snapshot(r) for r in rows]


# ---------------------------------------------------------------- helpers


def _row_to_snapshot(row: sqlite3.Row) -> Snapshot:
    manifest = SnapshotManifest.model_validate_json(row["manifest_json"])
    provenance = Provenance.model_validate_json(row["provenance_json"])
    block_dicts = json.loads(row["blocks_json"])
    blocks = [Block.model_validate(b) for b in block_dicts]
    return Snapshot(
        snapshot_id=row["snapshot_id"],
        session_id=row["session_id"],
        chain_id=row["chain_id"],
        checkpoint_id=row["checkpoint_id"],
        anchor_block_id=row["anchor_block_id"],
        anchor_sequence=row["anchor_sequence"],
        lookback_limit=row["lookback_limit"],
        manifest=manifest,
        provenance=provenance,
        blocks=blocks,
        created_at=_parse_iso(row["created_at"]),
    )


_FTS_TOKEN_RE = __import__("re").compile(r"[A-Za-z0-9_]+")


def _sanitize_fts_query(query: str) -> str:
    """Turn an arbitrary user prompt into a safe FTS5 MATCH expression.

    FTS5's query syntax has many reserved characters (`-`, `:`, `*`, `(`, `)`,
    quotes). We tokenize the prompt and OR the tokens together as prefix
    matches so partial words still hit. Empty if there are no tokens.
    """
    tokens = _FTS_TOKEN_RE.findall(query)
    # Skip extremely short tokens (FTS5 won't index single chars well).
    tokens = [t for t in tokens if len(t) > 1]
    if not tokens:
        return ""
    # Quote each token to avoid being parsed as syntax; `"foo" *` allows prefix.
    return " OR ".join(f'"{t}"' for t in tokens)


__all__ = ["Store", "SQLiteStore"]
