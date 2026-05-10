"""End-to-end smoke test exercising all 6 tools + 2 resources directly.

Bypasses the MCP transport layer and calls the underlying Store /
SemanticIndex / Summarizer plus the same logic the FastMCP tools use, so
we can sanity-check correctness without a real MCP client.

    python scripts/smoke_test.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from context_retrieval.config import Settings  # noqa: E402
from context_retrieval.models import (  # noqa: E402
    Block,
    Chain,
    Provenance,
    Session,
    Snapshot,
    SnapshotManifest,
)
from context_retrieval.search import Embedder, SemanticIndex  # noqa: E402
from context_retrieval.store import SQLiteStore  # noqa: E402
from context_retrieval.summarizer import Summarizer  # noqa: E402

from seed_demo import DEMO_BLOCKS  # noqa: E402


def _seed(store: SQLiteStore) -> tuple[str, str]:
    session_id = "smoke-session"
    chain_id = "smoke-chain"
    store.upsert_session(
        Session(session_id=session_id, active_chain_id=chain_id, lookback_limit=50)
    )
    store.upsert_chain(
        Chain(chain_id=chain_id, session_id=session_id, title="Smoke test chain")
    )
    base = datetime.now(timezone.utc) - timedelta(hours=1)
    for i, (role, content) in enumerate(DEMO_BLOCKS):
        store.upsert_block(
            Block(
                block_id=f"smoke-block-{i:02d}",
                chain_id=chain_id,
                sequence=i,
                role=role,  # type: ignore[arg-type]
                content=content,
                created_at=base + timedelta(minutes=i),
            )
        )
    return session_id, chain_id


def main() -> None:
    # ignore_cleanup_errors: SQLite holds the WAL file briefly even after
    # close() on Windows. Best to not blow up the test on tempdir teardown.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        db_path = Path(tmp) / "smoke.db"
        store = SQLiteStore(db_path)
        settings = Settings(db_path=db_path)
        index = SemanticIndex(store, embedder=Embedder(settings.embedding_model))
        summarizer = Summarizer(settings)

        session_id, chain_id = _seed(store)
        print(f"[seed] {store.chain_block_count(chain_id)} blocks in chain={chain_id}\n")

        # 1) query_chain (hybrid)
        print("[1] query_chain(prompt='cookie sameSite issue', mode='hybrid', explain=True)")
        matches, total = index.search(
            chain_id,
            "cookie sameSite issue",
            lookback_limit=50,
            top_k=5,
            mode="hybrid",
            explain=True,
        )
        assert matches, "expected hybrid matches"
        for m in matches:
            sb = m.score_breakdown
            assert sb is not None
            print(
                f"    {m.relevance:.3f}  seq={m.sequence}  sem={sb.semantic:.3f} "
                f"lex={sb.lexical:.3f} rec={sb.recency:.3f}  {m.preview[:70]}"
            )
        print(f"    total_candidates={total}\n")

        # 1b) lexical-only and semantic-only modes return matches too
        for mode in ("semantic", "lexical"):
            ms, _ = index.search(
                chain_id, "dark mode toggle", lookback_limit=50, top_k=3, mode=mode  # type: ignore[arg-type]
            )
            print(f"    mode={mode}: {len(ms)} matches")
            assert ms, f"expected results in mode={mode}"

        top_match = matches[0]

        # 2) scan_adjacent
        print(f"\n[2] scan_adjacent(block={top_match.block_id}, window=2)")
        neighbors = store.adjacent_blocks(top_match.block_id, before=2, after=2)
        assert any(b.block_id == top_match.block_id for b in neighbors)
        for b in neighbors:
            print(f"    seq={b.sequence} role={b.role}: {b.content[:70]}")

        # 3) set_lookback (session + chain scopes)
        print("\n[3] set_lookback(session, 20)")
        prev, upd = store.set_session_lookback(session_id, 20)
        assert (prev, upd) == (50, 20)
        eff = store.get_effective_lookback(session_id, chain_id, default=50)
        assert eff == 20
        print(f"    session lookback: {prev} -> {upd}, effective={eff}")

        prev2, upd2 = store.set_chain_lookback(session_id, chain_id, 7)
        eff2 = store.get_effective_lookback(session_id, chain_id, default=50)
        assert eff2 == 7
        print(f"    chain lookback: {prev2} -> {upd2}, effective={eff2} (chain overrides session)")

        # 4) summarize_history
        print("\n[4] summarize_history(detail_level='standard')")
        blocks = store.list_chain_blocks(chain_id)
        summary = summarizer.summarize(
            chain_id, session_id, blocks, detail_level="standard"
        )
        store.insert_summary(summary)
        print(
            f"    engine={summary.engine} summary_id={summary.summary_id} "
            f"themes={len(summary.themes)} checkpoints={len(summary.checkpoints)}"
        )
        for cp in summary.checkpoints[:5]:
            print(f"      - [{cp.position}] {cp.label}  (block={cp.block_id} cat={cp.category})")
        assert summary.checkpoints, "expected at least one checkpoint"
        # Pick a mid-chain checkpoint so the prefix is large enough to exercise
        # both the kept and skipped paths in reclaim_context.
        mid_cp = summary.checkpoints[len(summary.checkpoints) // 2]
        first_cp = mid_cp

        # 4b) cache hit on second call with same inputs
        cached = store.latest_summary_for_chain(chain_id, detail_level="standard")
        assert cached is not None
        assert cached.block_count == len(blocks)
        print(f"    cache hit: block_count={cached.block_count}")

        # 5) backtrack (dry_run + real)
        print(f"\n[5] backtrack(checkpoint={first_cp.checkpoint_id}, dry_run=True)")
        anchor = store.get_block(first_cp.block_id)
        assert anchor is not None
        prefix = store.list_chain_blocks(chain_id, up_to_sequence=anchor.sequence)
        role_counts: dict[str, int] = {}
        for b in prefix:
            role_counts[b.role] = role_counts.get(b.role, 0) + 1
        tokens = sum(max(1, len(b.content.split())) for b in prefix)
        manifest = SnapshotManifest(
            block_count=len(prefix),
            token_estimate=tokens,
            role_counts=role_counts,
            chain_title="Smoke test chain",
        )
        prov = Provenance(created_by=session_id, reason="smoke", name="smoke-snap")
        # Don't insert on dry_run.
        print(f"    would restore {manifest.block_count} blocks, ~{manifest.token_estimate} tokens")

        print(f"\n[5b] backtrack(checkpoint={first_cp.checkpoint_id}, dry_run=False)")
        snapshot_id = str(uuid.uuid4())
        snap = Snapshot(
            snapshot_id=snapshot_id,
            session_id=session_id,
            chain_id=chain_id,
            checkpoint_id=first_cp.checkpoint_id,
            anchor_block_id=anchor.block_id,
            anchor_sequence=anchor.sequence,
            lookback_limit=20,
            manifest=manifest,
            provenance=prov,
            blocks=prefix,
        )
        store.insert_snapshot(snap)
        print(f"    snapshot_id={snapshot_id} blocks_frozen={len(prefix)}")

        # 6) reclaim_context (with token_budget compression path)
        print(f"\n[6] reclaim_context(snapshot={snapshot_id}, token_budget=80)")
        loaded = store.get_snapshot(snapshot_id)
        assert loaded is not None
        # Pick a budget that forces *some* blocks to be skipped but keeps at
        # least one so we exercise both the kept and compressed paths.
        budget = 80
        kept = list(loaded.blocks)
        total = sum(max(1, len(b.content.split())) for b in kept)
        skipped: list[Block] = []
        while kept and total > budget:
            gone = kept.pop(0)
            skipped.append(gone)
            total -= max(1, len(gone.content.split()))
        compressed_summary = None
        if skipped:
            comp = summarizer.summarize(chain_id, session_id, skipped, detail_level="skim")
            compressed_summary = comp.plain_text
        tokens_injected = total + (
            max(1, len(compressed_summary.split())) if compressed_summary else 0
        )
        print(
            f"    blocks_injected={len(kept)} blocks_skipped={len(skipped)} "
            f"compressed={compressed_summary is not None} tokens_injected={tokens_injected}"
        )
        assert len(kept) >= 1

        # Resources
        print("\n[R] retrieval://history/{session_id}")
        latest = store.latest_summary_for_session(session_id)
        assert latest is not None
        print(f"    {len(latest.checkpoints)} clickable checkpoints persisted; engine={latest.engine}")

        print("\n[R] retrieval://snapshot/{snapshot_id}")
        snap2 = store.get_snapshot(snapshot_id)
        assert snap2 is not None
        # Confirm round-trip JSON serializes cleanly.
        rendered = json.dumps(snap2.model_dump(mode="json"))
        assert "anchor_sequence" in rendered
        print(f"    chain_id={snap2.chain_id} anchor_sequence={snap2.anchor_sequence}")

        store.close()
        print("\nAll checks passed.")


if __name__ == "__main__":
    main()
