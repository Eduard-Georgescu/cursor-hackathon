"""Summarizer: extractive contract, detail levels, hallucination guard."""

from __future__ import annotations

from context_retrieval.config import Settings
from context_retrieval.summarizer import Summarizer, _validate_checkpoints
from context_retrieval.store import SQLiteStore


def _summarizer_without_claude() -> Summarizer:
    # No API key => extractive path. Settings() with no key ensures it.
    return Summarizer(Settings(anthropic_api_key=None))


def test_extractive_summary_emits_checkpoints(seeded_store: SQLiteStore, chain_id, session_id):
    blocks = seeded_store.list_chain_blocks(chain_id)
    summ = _summarizer_without_claude()
    summary = summ.summarize(chain_id, session_id, blocks, detail_level="standard")
    assert summary.engine == "extractive"
    assert summary.checkpoints, "extractive summary must include checkpoints"
    assert summary.block_count == len(blocks)
    # All checkpoint block_ids must reference real blocks in the chain.
    real_ids = {b.block_id for b in blocks}
    for cp in summary.checkpoints:
        assert cp.block_id in real_ids


def test_detail_levels_change_output_size(
    seeded_store: SQLiteStore, chain_id, session_id
):
    blocks = seeded_store.list_chain_blocks(chain_id)
    summ = _summarizer_without_claude()
    skim = summ.summarize(chain_id, session_id, blocks, detail_level="skim")
    standard = summ.summarize(chain_id, session_id, blocks, detail_level="standard")
    deep = summ.summarize(chain_id, session_id, blocks, detail_level="deep")
    # Skim caps tighter than the others.
    assert len(skim.checkpoints) <= len(standard.checkpoints) <= len(deep.checkpoints) + 1
    # Themes scale with detail.
    assert len(deep.themes) >= len(skim.themes)


def test_empty_chain_returns_placeholder(session_id):
    summ = _summarizer_without_claude()
    summary = summ.summarize("nonexistent", session_id, [], detail_level="standard")
    assert summary.checkpoints == []
    assert "no activity" in summary.plain_text.lower()


def test_validator_drops_hallucinated_block_ids():
    valid = {"b01", "b02"}
    raw = [
        {"block_id": "b01", "label": "ok", "description": "real anchor"},
        {"block_id": "b99", "label": "fake", "description": "halluc."},
        {"block_id": "b02", "label": "valid", "description": "real"},
    ]
    out = _validate_checkpoints(raw, valid)
    assert {cp["block_id"] for cp in out} == {"b01", "b02"}


def test_validator_clamps_confidence_and_drops_bad_category():
    valid = {"b01"}
    raw = [
        {
            "block_id": "b01",
            "label": "x",
            "description": "y",
            "confidence": 5.0,
            "category": "nope",
        }
    ]
    out = _validate_checkpoints(raw, valid)
    assert out[0]["confidence"] == 1.0
    assert out[0]["category"] is None


def test_validator_handles_garbage_input():
    assert _validate_checkpoints(None, set()) == []
    assert _validate_checkpoints("not a list", set()) == []
    assert _validate_checkpoints([1, 2, 3], set()) == []
