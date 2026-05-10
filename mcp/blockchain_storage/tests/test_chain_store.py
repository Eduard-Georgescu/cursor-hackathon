from pathlib import Path

import pytest

from blockchain_mcp.store.chain_store import ChainStore


@pytest.fixture()
def store(tmp_path: Path) -> ChainStore:
    return ChainStore(tmp_path)


def test_append_commit_next(store: ChainStore) -> None:
    bid, rem = store.append_context("c1", "hello", token_count=None)
    assert rem > 0
    committed_id, next_id, pruned = store.commit_block(bid)
    assert committed_id == bid
    assert next_id != bid
    assert pruned is None
    stats = store.get_chain_stats("c1")
    assert stats["block_count"] == 2


def test_edit_diff(store: ChainStore) -> None:
    bid, _ = store.append_context("c1", "a\nb\n")
    uid, diff = store.edit_block(bid, {"prepend": "z\n"})
    assert uid == bid
    assert "z" in diff


def test_cancel_open(store: ChainStore) -> None:
    bid, _ = store.append_context("c1", "x")
    ok, relinked = store.cancel_block(bid, "test")
    assert ok and relinked
    cur = store.current_open_block("c1")
    assert cur.block_id != bid
    assert cur.content == ""


def test_quota_blocks_append(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SUBSCRIPTION_BLOCK_QUOTA", "0")
    store = ChainStore(tmp_path)
    with pytest.raises(ValueError, match="quota"):
        store.append_context("q1", "nope")


def test_prune_on_commit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MAX_BLOCKS_PER_CHAIN", "2")
    store = ChainStore(tmp_path)
    for i in range(3):
        bid, _ = store.append_context("p1", f"b{i}")
        _, _, pruned = store.commit_block(bid)
        if i == 0:
            assert pruned is None
    chain = store.get_chain("p1")
    assert len(chain.committed_blocks) == 2


def test_validate_apply(store: ChainStore) -> None:
    from blockchain_mcp.tools.validate_block import run_validate_block

    bid, _ = store.append_context("c1", "hello world")
    out = run_validate_block(store, block_id=bid)
    assert out["valid"] is True
    blk = store.get_block(bid)
    assert blk is not None
    assert blk.validation_status == "valid"
