"""Store CRUD, FTS sync, FK behavior, and snapshot lifecycle."""

from __future__ import annotations

import numpy as np
import pytest

from context_retrieval.models import (
    Block,
    Checkpoint,
    Decision,
    Provenance,
    Session,
    Snapshot,
    SnapshotManifest,
    Summary,
)
from context_retrieval.store import SQLiteStore


def test_chain_session_block_roundtrip(seeded_store: SQLiteStore, chain_id: str, session_id: str):
    chain = seeded_store.get_chain(chain_id)
    assert chain is not None and chain.title == "Test chain"

    sess = seeded_store.get_session(session_id)
    assert sess is not None
    assert sess.active_chain_id == chain_id

    blocks = seeded_store.list_chain_blocks(chain_id)
    assert len(blocks) == 10
    assert [b.sequence for b in blocks] == list(range(10))


def test_list_chain_blocks_lookback_keeps_tail(
    seeded_store: SQLiteStore, chain_id: str
):
    blocks = seeded_store.list_chain_blocks(chain_id, limit=3)
    assert [b.sequence for b in blocks] == [7, 8, 9]


def test_list_chain_blocks_up_to_sequence(
    seeded_store: SQLiteStore, chain_id: str
):
    blocks = seeded_store.list_chain_blocks(chain_id, up_to_sequence=4)
    assert [b.sequence for b in blocks] == [0, 1, 2, 3, 4]


def test_adjacent_blocks(seeded_store: SQLiteStore):
    anchor = seeded_store.get_block("b05")
    assert anchor is not None
    neighbors = seeded_store.adjacent_blocks("b05", before=2, after=1)
    assert [b.sequence for b in neighbors] == [3, 4, 5, 6]


def test_fts_index_finds_seeded_blocks(
    seeded_store: SQLiteStore, chain_id: str
):
    hits = seeded_store.lexical_search(chain_id, "cookie sameSite", limit=5)
    assert hits, "FTS5 trigger should have indexed block content"
    block_ids = {b.block_id for b, _ in hits}
    # The 'sameSite' agent block (b03) should be in there somewhere.
    assert "b03" in block_ids


def test_fts_handles_garbage_query(seeded_store: SQLiteStore, chain_id: str):
    # Bad FTS5 syntax must degrade to "no results", not raise.
    hits = seeded_store.lexical_search(chain_id, ":-)*", limit=5)
    assert hits == []


def test_embedding_roundtrip(store: SQLiteStore):
    block = Block(
        block_id="x", chain_id="c", sequence=0, role="user", content="hi"
    )
    store.upsert_block(block)
    vec = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    store.put_embedding("x", "model-a", vec)
    cached = store.get_embedding("x")
    assert cached is not None
    model, got = cached
    assert model == "model-a"
    np.testing.assert_allclose(got, vec)


def test_session_lookback_setter(seeded_store: SQLiteStore, session_id: str):
    prev, upd = seeded_store.set_session_lookback(session_id, 25)
    assert prev == 50
    assert upd == 25
    sess = seeded_store.get_session(session_id)
    assert sess is not None and sess.lookback_limit == 25


def test_chain_lookback_overrides_session(
    seeded_store: SQLiteStore, session_id: str, chain_id: str
):
    seeded_store.set_session_lookback(session_id, 25)
    seeded_store.set_chain_lookback(session_id, chain_id, 7)
    assert seeded_store.get_effective_lookback(session_id, chain_id, default=50) == 7
    # A different chain falls back to the session default.
    assert (
        seeded_store.get_effective_lookback(session_id, "other-chain", default=50)
        == 25
    )


def test_set_lookback_autocreates_missing_session(store: SQLiteStore):
    prev, upd = store.set_session_lookback("brand-new", 17)
    assert (prev, upd) == (50, 17)
    sess = store.get_session("brand-new")
    assert sess is not None and sess.lookback_limit == 17


def test_summary_roundtrip(store: SQLiteStore, seeded_store: SQLiteStore):
    summary = Summary(
        summary_id="sum-1",
        chain_id="test-chain",
        session_id="test-session",
        plain_text="hello",
        detail_level="standard",
        engine="extractive",
        token_count=2,
        block_count=3,
        themes=["auth", "themes"],
        decisions=[Decision(summary="set sameSite=lax", block_id="b03")],
        open_questions=["what's next?"],
        checkpoints=[
            Checkpoint(
                checkpoint_id="cp-1",
                summary_id="sum-1",
                block_id="b03",
                chain_id="test-chain",
                position=0,
                label="cookie fix",
                description="set sameSite=lax",
                reason="auth unblocks here",
                confidence=0.9,
                category="decision",
            )
        ],
    )
    seeded_store.insert_summary(summary)
    got = seeded_store.latest_summary_for_session("test-session")
    assert got is not None
    assert got.summary_id == "sum-1"
    assert got.themes == ["auth", "themes"]
    assert len(got.checkpoints) == 1
    assert got.checkpoints[0].category == "decision"

    cp = seeded_store.get_checkpoint("cp-1")
    assert cp is not None and cp.label == "cookie fix"


def test_snapshot_roundtrip(seeded_store: SQLiteStore, session_id: str, chain_id: str):
    blocks = seeded_store.list_chain_blocks(chain_id, up_to_sequence=4)
    snap = Snapshot(
        snapshot_id="snap-1",
        session_id=session_id,
        chain_id=chain_id,
        anchor_block_id="b04",
        anchor_sequence=4,
        lookback_limit=20,
        manifest=SnapshotManifest(
            block_count=len(blocks),
            token_estimate=42,
            role_counts={"user": 3, "agent": 1, "tool": 1},
            chain_title="Test chain",
        ),
        provenance=Provenance(created_by=session_id, reason="testing", dry_run=False),
        blocks=blocks,
    )
    seeded_store.insert_snapshot(snap)
    got = seeded_store.get_snapshot("snap-1")
    assert got is not None
    assert got.anchor_sequence == 4
    assert len(got.blocks) == 5
    listed = seeded_store.list_snapshots(session_id)
    assert any(s.snapshot_id == "snap-1" for s in listed)


def test_upsert_session_overwrites_active_chain(store: SQLiteStore):
    s = Session(session_id="s1", active_chain_id="c1", lookback_limit=10)
    store.upsert_session(s)
    s2 = Session(session_id="s1", active_chain_id="c2", lookback_limit=15)
    store.upsert_session(s2)
    got = store.get_session("s1")
    assert got is not None
    assert got.active_chain_id == "c2"
    assert got.lookback_limit == 15
