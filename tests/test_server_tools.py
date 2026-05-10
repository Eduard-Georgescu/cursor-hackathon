"""In-process tests for the FastMCP tools.

We bypass the MCP transport by constructing a minimal Context shim and
calling the tool callables directly. This exercises the same validation
and code paths the protocol invokes, just without stdio/SSE in the way.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from context_retrieval.config import Settings
from context_retrieval.models import Block, Chain, Session
from context_retrieval.search import Embedder, SemanticIndex
from context_retrieval.store import SQLiteStore
from context_retrieval.summarizer import Summarizer

# Importing the server module is enough to register tools/resources on `mcp`.
import context_retrieval.server as server_mod
from context_retrieval.server import (
    AppState,
    backtrack,
    history_resource,
    query_chain,
    reclaim_context,
    scan_adjacent,
    set_lookback,
    snapshot_resource,
    summarize_history,
)


# ----------------------------------------------------------------- shims


class _NoopSession:
    async def send_resource_updated(self, _uri):  # noqa: D401
        return None


@dataclass
class _RequestContext:
    lifespan_context: Any


@dataclass
class FakeContext:
    """Just enough of mcp.server.fastmcp.Context for our tools."""

    request_context: _RequestContext
    session: _NoopSession


# --------------------------------------------------------------- fixtures


@pytest.fixture
def app_state(tmp_path: Path) -> AppState:
    settings = Settings(db_path=tmp_path / "tools.db", anthropic_api_key=None)
    store = SQLiteStore(settings.db_path)

    # Seed a small chain.
    store.upsert_session(
        Session(session_id="s1", active_chain_id="c1", lookback_limit=50)
    )
    store.upsert_chain(Chain(chain_id="c1", session_id="s1", title="t"))
    for i, (role, content) in enumerate(
        [
            ("user", "Auth is broken: sign-in redirects to /login."),
            ("agent", "Set sameSite='lax' on the auth cookie."),
            ("user", "Done. Now add a dark mode toggle."),
            ("agent", "Installed next-themes and added a toggle component."),
            ("user", "Chart on /analytics doesn't respect dark mode."),
            ("agent", "Updating Recharts to read CSS variables."),
        ]
    ):
        store.upsert_block(
            Block(
                block_id=f"b{i:02d}",
                chain_id="c1",
                sequence=i,
                role=role,  # type: ignore[arg-type]
                content=content,
            )
        )

    state = AppState(
        settings=settings,
        store=store,
        index=SemanticIndex(
            store, embedder=Embedder(prefer_model="this/model/does/not/exist")
        ),
        summarizer=Summarizer(settings),
    )
    # Mirror the module global the resources read from.
    server_mod._APP_STATE = state
    yield state
    server_mod._APP_STATE = None
    store.close()


@pytest.fixture
def ctx(app_state: AppState) -> FakeContext:
    return FakeContext(
        request_context=_RequestContext(lifespan_context=app_state),
        session=_NoopSession(),
    )


# --------------------------------------------------------------- query_chain


def test_query_chain_hybrid(ctx: FakeContext):
    result = query_chain("c1", "cookie sameSite issue", ctx, top_k=3, mode="hybrid")
    assert result.chain_id == "c1"
    assert result.mode == "hybrid"
    assert result.matches, "hybrid search must return at least one match"
    assert all(0.0 <= m.relevance <= 1.0 for m in result.matches)


def test_query_chain_explain_populates_breakdown(ctx: FakeContext):
    result = query_chain("c1", "cookie", ctx, top_k=2, explain=True)
    assert result.matches
    for m in result.matches:
        assert m.score_breakdown is not None
        assert m.context_hint is not None


def test_query_chain_uses_session_lookback_when_omitted(ctx: FakeContext):
    ctx.request_context.lifespan_context.store.set_session_lookback("s1", 2)
    result = query_chain("c1", "cookie", ctx, session_id="s1", top_k=5)
    # With lookback=2 only the two most recent blocks are candidates.
    assert result.lookback_limit == 2
    assert result.total_candidates == 2


# --------------------------------------------------------------- scan_adjacent


def test_scan_adjacent_returns_neighbors(ctx: FakeContext):
    result = scan_adjacent("b02", ctx, window=1)
    assert result.before_window == 1
    assert result.after_window == 1
    assert [b.sequence for b in result.blocks] == [1, 2, 3]
    assert "b02" in result.merged_context or "sequence" not in result.merged_context


def test_scan_adjacent_unknown_block_raises(ctx: FakeContext):
    with pytest.raises(ValueError):
        scan_adjacent("nope", ctx, window=1)


def test_scan_adjacent_max_tokens_truncates(ctx: FakeContext):
    result = scan_adjacent("b02", ctx, window=2, max_tokens=5)
    # The anchor block must survive trimming even if it eats the budget.
    assert any(b.block_id == "b02" for b in result.blocks)
    assert result.truncated is True


# ---------------------------------------------------------------- set_lookback


def test_set_lookback_session_scope(ctx: FakeContext):
    result = set_lookback("s1", 12, ctx)
    assert result.scope == "session"
    assert result.previous_limit == 50
    assert result.updated_limit == 12


def test_set_lookback_chain_scope_requires_chain_id(ctx: FakeContext):
    with pytest.raises(ValueError):
        set_lookback("s1", 12, ctx, scope="chain")


def test_set_lookback_chain_scope(ctx: FakeContext):
    result = set_lookback("s1", 5, ctx, scope="chain", chain_id="c1")
    assert result.scope == "chain"
    assert result.updated_limit == 5
    eff = ctx.request_context.lifespan_context.store.get_effective_lookback(
        "s1", "c1", default=50
    )
    assert eff == 5


def test_set_lookback_rejects_too_large(ctx: FakeContext):
    state = ctx.request_context.lifespan_context
    with pytest.raises(ValueError):
        set_lookback("s1", state.settings.max_lookback + 1, ctx)


# ---------------------------------------------------------- summarize_history


def test_summarize_history_extractive(ctx: FakeContext):
    result = asyncio.run(summarize_history("c1", "s1", ctx, detail_level="standard"))
    assert result["engine"] == "extractive"
    assert result["checkpoints"], "extractive path must include checkpoints"
    assert result["block_count"] == 6
    # Resource is now reachable.
    body = history_resource("s1")
    payload = json.loads(body)
    assert payload["summary_id"] == result["summary_id"]


def test_summarize_history_caches(ctx: FakeContext):
    first = asyncio.run(summarize_history("c1", "s1", ctx, detail_level="standard"))
    second = asyncio.run(summarize_history("c1", "s1", ctx, detail_level="standard"))
    assert first["summary_id"] == second["summary_id"]
    assert second["cached"] is True


def test_summarize_history_force_refresh_skips_cache(ctx: FakeContext):
    first = asyncio.run(summarize_history("c1", "s1", ctx, detail_level="standard"))
    refreshed = asyncio.run(
        summarize_history("c1", "s1", ctx, detail_level="standard", force_refresh=True)
    )
    assert first["summary_id"] != refreshed["summary_id"]


# ------------------------------------------------------------------ backtrack


def test_backtrack_dry_run_does_not_persist(ctx: FakeContext):
    result = backtrack("s1", ctx, block_id="b03", dry_run=True)
    assert result.dry_run is True
    assert result.snapshot_id is None
    assert result.block_count == 4
    snaps = ctx.request_context.lifespan_context.store.list_snapshots("s1")
    assert snaps == []


def test_backtrack_real_persists_snapshot(ctx: FakeContext):
    result = backtrack(
        "s1", ctx, block_id="b03", name="checkpoint-test", reason="testing"
    )
    assert result.snapshot_id is not None
    snap = ctx.request_context.lifespan_context.store.get_snapshot(result.snapshot_id)
    assert snap is not None
    assert snap.provenance.name == "checkpoint-test"
    assert snap.anchor_sequence == 3


def test_backtrack_requires_anchor(ctx: FakeContext):
    with pytest.raises(ValueError):
        backtrack("s1", ctx)


# ----------------------------------------------------------- reclaim_context


def test_reclaim_context_respects_token_budget_and_compresses(ctx: FakeContext):
    bt = backtrack("s1", ctx, block_id="b05")
    assert bt.snapshot_id is not None
    result = asyncio.run(
        reclaim_context("s1", bt.snapshot_id, ctx, token_budget=8, compress_if_over=True)
    )
    assert result.success is True
    assert result.blocks_skipped >= 1
    assert result.compressed is True
    assert result.compression_summary
    assert result.active_chain_id == "c1"

    # snapshot resource still readable
    body = snapshot_resource(bt.snapshot_id)
    assert "anchor_sequence" in body


def test_reclaim_context_wrong_session_raises(ctx: FakeContext):
    bt = backtrack("s1", ctx, block_id="b03")
    assert bt.snapshot_id is not None
    with pytest.raises(ValueError):
        asyncio.run(reclaim_context("other-session", bt.snapshot_id, ctx))


def test_snapshot_resource_returns_not_found(ctx: FakeContext):
    body = snapshot_resource("nope")
    assert json.loads(body)["error"] == "not_found"
