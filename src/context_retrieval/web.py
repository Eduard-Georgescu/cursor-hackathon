"""HTTP bridge for the demo website.

Wraps the same `Store` / `SemanticIndex` / `Summarizer` code the MCP
tools call. The browser hits this HTTP API; the API does the same work
the MCP server would do over stdio.

Routes mirror the 6 tools (plus a few read helpers the UI needs):

    POST  /api/query              query_chain
    POST  /api/adjacent           scan_adjacent
    POST  /api/lookback           set_lookback
    POST  /api/summarize          summarize_history
    POST  /api/backtrack          backtrack
    POST  /api/reclaim            reclaim_context

    GET   /api/sessions           list sessions (UI bootstrap)
    GET   /api/chains             list chains (UI bootstrap)
    GET   /api/chain/{id}/blocks  list raw blocks for the timeline
    GET   /api/history/{sid}      retrieval://history/{session_id}
    GET   /api/snapshot/{id}      retrieval://snapshot/{point_id}
    GET   /api/snapshots/{sid}    history of snapshots for a session
    GET   /api/prompts            metadata for the 3 MCP prompts

The static frontend is served at GET / from ./web/ (relative to repo root).
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import Settings
from .logging_utils import configure_logging, get_logger
from .models import (
    AdjacentResult,
    BacktrackResult,
    Block,
    DetailLevel,
    LookbackResult,
    Provenance,
    QueryResult,
    ReclaimResult,
    Role,
    SearchMode,
    Session,
    Snapshot,
    SnapshotManifest,
    TimeRange,
)
from .search import Embedder, SemanticIndex
from .store import SQLiteStore
from .summarizer import Summarizer


_log = get_logger("web")


# ---------------------------------------------------------------- request schemas


class QueryBody(BaseModel):
    chain_id: str
    prompt: str
    session_id: str | None = None
    lookback_limit: int | None = None
    top_k: int = Field(default=10, ge=1, le=100)
    mode: SearchMode = "hybrid"
    diversify: bool = False
    roles: list[Role] | None = None
    time_range: TimeRange | None = None
    min_relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    explain: bool = True


class AdjacentBody(BaseModel):
    block_id: str
    window: int = Field(default=1, ge=0, le=200)
    before_window: int | None = None
    after_window: int | None = None
    roles: list[Role] | None = None
    max_tokens: int | None = Field(default=None, ge=1)


class LookbackBody(BaseModel):
    session_id: str
    limit: int = Field(ge=1, le=10_000)
    scope: Literal["session", "chain"] = "session"
    chain_id: str | None = None


class SummarizeBody(BaseModel):
    chain_id: str
    session_id: str
    detail_level: DetailLevel = "standard"
    force_refresh: bool = False
    model: str | None = None


class BacktrackBody(BaseModel):
    session_id: str
    checkpoint_id: str | None = None
    block_id: str | None = None
    name: str | None = None
    reason: str | None = None
    dry_run: bool = False


class ReclaimBody(BaseModel):
    session_id: str
    snapshot_id: str
    token_budget: int | None = Field(default=None, ge=1)
    roles: list[Role] | None = None
    compress_if_over: bool = True


# ---------------------------------------------------------------- app factory


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    configure_logging(settings.log_level)
    _log.info(f"event=web.start db={settings.db_path}")

    store = SQLiteStore(settings.db_path)
    search_index = SemanticIndex(store, embedder=Embedder(settings.embedding_model))
    summarizer = Summarizer(settings)

    app = FastAPI(title="Context Retrieval Demo", version="0.1.0")

    # ------------------------------------------------------------------ tools

    @app.post("/api/query", response_model=QueryResult)
    def query(body: QueryBody) -> QueryResult:
        limit = (
            body.lookback_limit
            if body.lookback_limit and body.lookback_limit > 0
            else store.get_effective_lookback(
                body.session_id or "", body.chain_id, default=settings.max_lookback
            )
        )
        limit = min(limit, settings.max_lookback)
        matches, total = search_index.search(
            chain_id=body.chain_id,
            prompt=body.prompt,
            lookback_limit=limit,
            top_k=body.top_k,
            mode=body.mode,
            diversify=body.diversify,
            roles=body.roles,
            time_range=body.time_range,
            min_relevance=body.min_relevance,
            explain=body.explain,
        )
        return QueryResult(
            chain_id=body.chain_id,
            prompt=body.prompt,
            lookback_limit=limit,
            mode=body.mode,
            total_candidates=total,
            matches=matches,
        )

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
            blocks = [
                b for b in blocks if b.role in role_set or b.block_id == body.block_id
            ]
        total_chain_length = store.chain_block_count(anchor.chain_id)
        truncated = False
        if body.max_tokens is not None:
            kept = list(blocks)
            total = sum(_estimate_block_tokens(b.content) for b in kept)
            while kept and total > body.max_tokens and len(kept) > 1:
                if kept[0].block_id == body.block_id:
                    gone = kept.pop()
                else:
                    gone = kept.pop(0)
                total -= _estimate_block_tokens(gone.content)
                truncated = True
            blocks = kept
        merged = _format_blocks_as_text(blocks)
        return AdjacentResult(
            block_id=body.block_id,
            before_window=before,
            after_window=after,
            blocks=blocks,
            merged_context=merged,
            total_tokens=sum(_estimate_block_tokens(b.content) for b in blocks),
            total_chain_length=total_chain_length,
            truncated=truncated,
        )

    @app.post("/api/lookback", response_model=LookbackResult)
    def lookback(body: LookbackBody) -> LookbackResult:
        if body.limit > settings.max_lookback:
            raise HTTPException(
                400,
                f"limit must be in [1, {settings.max_lookback}]; got {body.limit}",
            )
        if body.scope == "chain" and not body.chain_id:
            raise HTTPException(400, "chain_id is required when scope='chain'")
        if body.scope == "session":
            prev, upd = store.set_session_lookback(body.session_id, body.limit)
        else:
            assert body.chain_id is not None
            prev, upd = store.set_chain_lookback(
                body.session_id, body.chain_id, body.limit
            )
        return LookbackResult(
            session_id=body.session_id,
            scope=body.scope,
            chain_id=body.chain_id,
            previous_limit=prev,
            updated_limit=upd,
        )

    @app.post("/api/summarize")
    def summarize(body: SummarizeBody) -> dict[str, Any]:
        blocks = store.list_chain_blocks(body.chain_id)
        block_count = len(blocks)
        cached = store.latest_summary_for_chain(
            body.chain_id, detail_level=body.detail_level
        )
        if (
            cached is not None
            and not body.force_refresh
            and cached.block_count == block_count
        ):
            return {
                "summary_id": cached.summary_id,
                "plain_text": cached.plain_text,
                "checkpoints": [cp.model_dump(mode="json") for cp in cached.checkpoints],
                "themes": cached.themes,
                "decisions": [d.model_dump(mode="json") for d in cached.decisions],
                "open_questions": cached.open_questions,
                "engine": cached.engine,
                "detail_level": cached.detail_level,
                "generated_at": cached.created_at.isoformat(),
                "cached": True,
                "block_count": cached.block_count,
            }
        summary = summarizer.summarize(
            body.chain_id,
            body.session_id,
            blocks,
            detail_level=body.detail_level,
            model_override=body.model,
        )
        store.insert_summary(summary)
        return {
            "summary_id": summary.summary_id,
            "plain_text": summary.plain_text,
            "checkpoints": [cp.model_dump(mode="json") for cp in summary.checkpoints],
            "themes": summary.themes,
            "decisions": [d.model_dump(mode="json") for d in summary.decisions],
            "open_questions": summary.open_questions,
            "engine": summary.engine,
            "detail_level": summary.detail_level,
            "generated_at": summary.created_at.isoformat(),
            "cached": False,
            "block_count": summary.block_count,
        }

    @app.post("/api/backtrack", response_model=BacktrackResult)
    def backtrack(body: BacktrackBody) -> BacktrackResult:
        if not body.checkpoint_id and not body.block_id:
            raise HTTPException(400, "provide either checkpoint_id or block_id")
        if body.checkpoint_id is not None:
            cp = store.get_checkpoint(body.checkpoint_id)
            if cp is None:
                raise HTTPException(404, f"unknown checkpoint_id: {body.checkpoint_id}")
            anchor = store.get_block(cp.block_id)
            chain_id = cp.chain_id
            anchor_label = cp.label
        else:
            assert body.block_id is not None
            anchor = store.get_block(body.block_id)
            if anchor is None:
                raise HTTPException(404, f"unknown block_id: {body.block_id}")
            cp = None
            chain_id = anchor.chain_id
            anchor_label = f"block {anchor.sequence}"
        if anchor is None:
            raise HTTPException(404, "anchor block not found")
        prefix = store.list_chain_blocks(chain_id, up_to_sequence=anchor.sequence)
        sess = store.get_session(body.session_id) or Session(session_id=body.session_id)
        if store.get_session(body.session_id) is None:
            store.upsert_session(sess)
        role_counts = _block_role_counts(prefix)
        token_estimate = sum(_estimate_block_tokens(b.content) for b in prefix)
        chain = store.get_chain(chain_id)
        manifest = SnapshotManifest(
            block_count=len(prefix),
            token_estimate=token_estimate,
            role_counts=role_counts,
            chain_title=chain.title if chain else None,
        )
        prov = Provenance(
            created_by=body.session_id,
            reason=body.reason,
            dry_run=body.dry_run,
            name=body.name,
        )
        snapshot_id: str | None = None
        if not body.dry_run:
            snapshot_id = str(uuid.uuid4())
            store.insert_snapshot(
                Snapshot(
                    snapshot_id=snapshot_id,
                    session_id=body.session_id,
                    chain_id=chain_id,
                    checkpoint_id=cp.checkpoint_id if cp else None,
                    anchor_block_id=anchor.block_id,
                    anchor_sequence=anchor.sequence,
                    lookback_limit=sess.lookback_limit,
                    manifest=manifest,
                    provenance=prov,
                    blocks=prefix,
                )
            )
        verb = "Would restore" if body.dry_run else "Restored"
        return BacktrackResult(
            snapshot_id=snapshot_id,
            anchor_block_id=anchor.block_id,
            anchor_sequence=anchor.sequence,
            block_count=len(prefix),
            estimated_tokens=token_estimate,
            role_counts=role_counts,
            restored_summary=(
                f"{verb} {len(prefix)} block(s) up to '{anchor_label}' "
                f"(sequence {anchor.sequence})."
            ),
            dry_run=body.dry_run,
        )

    @app.post("/api/reclaim", response_model=ReclaimResult)
    def reclaim(body: ReclaimBody) -> ReclaimResult:
        snap = store.get_snapshot(body.snapshot_id)
        if snap is None:
            raise HTTPException(404, f"unknown snapshot_id: {body.snapshot_id}")
        if snap.session_id != body.session_id:
            raise HTTPException(
                400,
                f"snapshot {body.snapshot_id} belongs to session {snap.session_id}, "
                f"not {body.session_id}",
            )
        blocks = list(snap.blocks)
        if body.roles:
            role_set = set(body.roles)
            blocks = [b for b in blocks if b.role in role_set]
        skipped: list[Block] = []
        if body.token_budget is not None:
            total = sum(_estimate_block_tokens(b.content) for b in blocks)
            while blocks and total > body.token_budget:
                gone = blocks.pop(0)
                skipped.append(gone)
                total -= _estimate_block_tokens(gone.content)
        compression_summary: str | None = None
        compressed = False
        if skipped and body.compress_if_over:
            comp = summarizer.summarize(
                snap.chain_id,
                body.session_id,
                skipped,
                detail_level="skim",
            )
            compression_summary = comp.plain_text
            compressed = True
        tokens_injected = sum(_estimate_block_tokens(b.content) for b in blocks)
        if compression_summary:
            tokens_injected += _estimate_block_tokens(compression_summary)
        sess = store.get_session(body.session_id) or Session(session_id=body.session_id)
        sess.active_chain_id = snap.chain_id
        store.upsert_session(sess)
        return ReclaimResult(
            success=True,
            snapshot_id=body.snapshot_id,
            tokens_injected=tokens_injected,
            blocks_injected=len(blocks),
            blocks_skipped=len(skipped),
            compressed=compressed,
            compression_summary=compression_summary,
            active_chain_id=snap.chain_id,
        )

    # ------------------------------------------------------------ read helpers

    @app.get("/api/sessions")
    def list_sessions() -> list[dict[str, Any]]:
        # The store doesn't expose `list sessions` (Servers 1 & 2 own writes);
        # fall back to a SELECT against the underlying connection.
        with store._lock:  # type: ignore[attr-defined]
            rows = store._conn.execute(  # type: ignore[attr-defined]
                "SELECT session_id, active_chain_id, lookback_limit, created_at FROM sessions"
            ).fetchall()
        return [dict(r) for r in rows]

    @app.get("/api/chains")
    def list_chains() -> list[dict[str, Any]]:
        with store._lock:  # type: ignore[attr-defined]
            rows = store._conn.execute(  # type: ignore[attr-defined]
                "SELECT chain_id, session_id, title, created_at FROM chains ORDER BY created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    @app.get("/api/chain/{chain_id}/blocks", response_model=list[Block])
    def list_blocks(chain_id: str) -> list[Block]:
        return store.list_chain_blocks(chain_id)

    @app.get("/api/history/{session_id}")
    def history_resource(session_id: str) -> dict[str, Any]:
        summary = store.latest_summary_for_session(session_id)
        if summary is None:
            return {"session_id": session_id, "summary": None}
        payload = summary.model_dump(mode="json")
        payload["session_id"] = session_id
        return payload

    @app.get("/api/snapshot/{point_id}")
    def snapshot_resource(point_id: str) -> dict[str, Any]:
        snap = store.get_snapshot(point_id)
        if snap is None:
            raise HTTPException(404, "snapshot not found")
        return snap.model_dump(mode="json")

    @app.get("/api/snapshots/{session_id}")
    def snapshots_for_session(session_id: str) -> list[dict[str, Any]]:
        snaps = store.list_snapshots(session_id)
        # Trim block payloads — UI only needs the manifest in the list view.
        out: list[dict[str, Any]] = []
        for s in snaps:
            d = s.model_dump(mode="json")
            d.pop("blocks", None)
            out.append(d)
        return out

    @app.get("/api/prompts")
    def prompts_metadata() -> list[dict[str, Any]]:
        # Static metadata describing the 3 MCP prompts. The UI surfaces them
        # as one-click recipes that fire the underlying tool sequence.
        return [
            {
                "name": "recap_history",
                "title": "Recap History",
                "description": "One-click recap of the session, narrated to the user.",
                "params": [
                    {"name": "session_id", "kind": "session"},
                    {
                        "name": "detail_level",
                        "kind": "enum",
                        "options": ["skim", "standard", "deep"],
                    },
                ],
            },
            {
                "name": "find_related_context",
                "title": "Find Related Context",
                "description": "Hybrid search a topic and explain the top matches.",
                "params": [
                    {"name": "chain_id", "kind": "chain"},
                    {"name": "topic", "kind": "text"},
                ],
            },
            {
                "name": "restore_to_decision",
                "title": "Restore to Decision",
                "description": "Find a decision moment and walk through backtrack -> reclaim_context.",
                "params": [
                    {"name": "session_id", "kind": "session"},
                    {"name": "decision_query", "kind": "text"},
                ],
            },
        ]

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "claude_enabled": settings.claude_enabled,
            "engine": "claude" if settings.claude_enabled else "extractive",
            "embedding_model": settings.embedding_model,
            "db_path": str(settings.db_path),
            "timestamp": _utcnow_iso(),
        }

    # -------------------------------------------------------------- frontend

    web_dir = Path(__file__).resolve().parents[2] / "web"
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


# Module-level app for `uvicorn context_retrieval.web:app`.
app = create_app()
