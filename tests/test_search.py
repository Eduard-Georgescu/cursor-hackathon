"""Hybrid retrieval: semantic vs lexical vs hybrid, MMR, filters, fallback."""

from __future__ import annotations

from datetime import datetime, timezone

from context_retrieval.models import TimeRange
from context_retrieval.search import Embedder, SemanticIndex
from context_retrieval.store import SQLiteStore


def test_fallback_embedder_works_without_sentence_transformers():
    # Force the hashed-TF-IDF path by pointing at a missing model. We
    # rely on the fact that Embedder swallows load errors and falls back.
    emb = Embedder(prefer_model="this/model/definitely-does-not-exist")
    vecs = emb.encode(["hello world", "another piece of text"])
    assert vecs.shape == (2, emb.dim)
    # Vectors should be roughly unit-norm.
    norms = (vecs * vecs).sum(axis=1) ** 0.5
    assert (norms > 0).all()


def _index(store: SQLiteStore) -> SemanticIndex:
    return SemanticIndex(
        store, embedder=Embedder(prefer_model="this/model/does/not/exist")
    )


def test_hybrid_returns_relevant_blocks(seeded_store, chain_id):
    idx = _index(seeded_store)
    matches, total = idx.search(
        chain_id, "cookie sameSite issue", lookback_limit=50, top_k=5, mode="hybrid"
    )
    assert total == 10
    assert matches
    # Block 03 mentions sameSite explicitly; expect it in the top results.
    top_ids = {m.block_id for m in matches}
    assert "b03" in top_ids


def test_semantic_only_mode_returns_results(seeded_store, chain_id):
    idx = _index(seeded_store)
    matches, _ = idx.search(
        chain_id, "dark mode", lookback_limit=50, top_k=3, mode="semantic"
    )
    assert matches
    assert all(0.0 <= m.relevance <= 1.0 for m in matches)


def test_lexical_only_mode_returns_results(seeded_store, chain_id):
    idx = _index(seeded_store)
    matches, _ = idx.search(
        chain_id, "Recharts CSS variables", lookback_limit=50, top_k=3, mode="lexical"
    )
    assert matches
    # b08 is the Recharts/CSS variables block.
    assert any(m.block_id == "b08" for m in matches)


def test_explain_populates_breakdown_and_hint(seeded_store, chain_id):
    idx = _index(seeded_store)
    matches, _ = idx.search(
        chain_id, "cookie", lookback_limit=50, top_k=2, mode="hybrid", explain=True
    )
    assert matches
    for m in matches:
        assert m.score_breakdown is not None
        assert m.context_hint is not None
        assert 0.0 <= m.score_breakdown.fused <= 1.0


def test_role_filter_restricts_results(seeded_store, chain_id):
    idx = _index(seeded_store)
    matches, _ = idx.search(
        chain_id, "cookie", lookback_limit=50, top_k=5, roles=["tool"]
    )
    # Only one tool block exists in the sample; everything else is filtered out.
    assert all(m.role == "tool" for m in matches)


def test_min_relevance_filters(seeded_store, chain_id):
    idx = _index(seeded_store)
    matches, _ = idx.search(
        chain_id, "cookie", lookback_limit=50, top_k=10, min_relevance=0.999
    )
    # Nearly nothing should clear a 0.999 floor.
    assert all(m.relevance >= 0.999 for m in matches)


def test_time_range_excludes_recent_blocks(seeded_store, chain_id):
    # Filter to a window strictly before the seed time → no blocks should match.
    far_past = datetime(2000, 1, 1, tzinfo=timezone.utc)
    idx = _index(seeded_store)
    matches, _ = idx.search(
        chain_id,
        "cookie",
        lookback_limit=50,
        top_k=5,
        time_range=TimeRange(to=far_past),
    )
    assert matches == []


def test_diversify_spreads_results(seeded_store, chain_id):
    idx = _index(seeded_store)
    matches, _ = idx.search(
        chain_id, "auth dark mode chart", lookback_limit=50, top_k=4, diversify=True
    )
    # MMR shouldn't return 4 adjacent blocks for a 3-topic prompt.
    seqs = [m.sequence for m in matches]
    gap = max(seqs) - min(seqs)
    assert gap >= 3
