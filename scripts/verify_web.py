"""End-to-end verification of the demo web API.

Runs through every endpoint with realistic payloads. Used by the dev to confirm
all six MCP tools are reachable over HTTP, plus the two resources and a few
read helpers.
"""

from __future__ import annotations

import json
import sys
import urllib.request as u


BASE = "http://127.0.0.1:8765"


def get(path: str):
    with u.urlopen(BASE + path) as r:
        return json.loads(r.read())


def post(path: str, body: dict):
    req = u.Request(
        BASE + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with u.urlopen(req) as r:
        return json.loads(r.read())


def main() -> int:
    print("== /api/health")
    print(json.dumps(get("/api/health"), indent=2))

    print("\n== /api/sessions")
    print(json.dumps(get("/api/sessions"), indent=2))

    print("\n== /api/chains")
    print(json.dumps(get("/api/chains"), indent=2))

    blocks = get("/api/chain/demo-chain/blocks")
    print(f"\n== /api/chain/demo-chain/blocks  -> {len(blocks)} blocks "
          f"(first seq={blocks[0]['sequence']}, last seq={blocks[-1]['sequence']})")

    print("\n== POST /api/query  (hybrid, explain=True)")
    q = post("/api/query", {
        "chain_id": "demo-chain",
        "prompt": "cookie sameSite issue",
        "top_k": 3,
        "explain": True,
        "session_id": "demo-session",
    })
    print(f"  mode={q['mode']} matches={len(q['matches'])} candidates={q['total_candidates']}")
    for m in q["matches"]:
        br = m.get("score_breakdown") or {}
        print(f"  - {m['block_id']} seq={m['sequence']} role={m['role']} "
              f"rel={m['relevance']:.3f} sem={br.get('semantic', 0):.2f} "
              f"lex={br.get('lexical', 0):.2f} rec={br.get('recency', 0):.2f}")
        print(f"      preview: {m['preview'][:80]}")

    anchor = q["matches"][0]["block_id"]
    print(f"\n== POST /api/adjacent  (anchor={anchor}, window=1)")
    adj = post("/api/adjacent", {"block_id": anchor, "window": 1})
    print(f"  blocks={len(adj['blocks'])} tokens={adj['total_tokens']} truncated={adj['truncated']}")

    print("\n== POST /api/lookback  (scope=session, limit=40)")
    lb = post("/api/lookback", {
        "session_id": "demo-session", "limit": 40, "scope": "session",
    })
    print(f"  previous={lb['previous_limit']} updated={lb['updated_limit']}")

    print("\n== POST /api/summarize  (detail=standard)")
    s = post("/api/summarize", {
        "chain_id": "demo-chain",
        "session_id": "demo-session",
        "detail_level": "standard",
        "force_refresh": True,
    })
    print(f"  engine={s['engine']} checkpoints={len(s['checkpoints'])} "
          f"themes={s['themes']} blocks={s['block_count']}")
    for cp in s["checkpoints"][:3]:
        print(f"  - {cp['label']} (block={cp['block_id']} cat={cp.get('category')})")

    print("\n== GET /api/history/demo-session")
    h = get("/api/history/demo-session")
    print(f"  detail_level={h.get('detail_level')} checkpoints={len(h.get('checkpoints', []))}")

    cp = s["checkpoints"][0]
    print(f"\n== POST /api/backtrack  (checkpoint={cp['checkpoint_id'][:8]}, dry_run=False)")
    bt = post("/api/backtrack", {
        "session_id": "demo-session",
        "checkpoint_id": cp["checkpoint_id"],
        "dry_run": False,
        "name": "verify",
        "reason": "automated verify_web.py",
    })
    print(f"  snapshot_id={bt['snapshot_id'][:8]} "
          f"blocks={bt['block_count']} tokens={bt['estimated_tokens']}")

    print(f"\n== GET /api/snapshot/{bt['snapshot_id'][:8]}…")
    snap = get(f"/api/snapshot/{bt['snapshot_id']}")
    print(f"  anchor={snap['anchor_block_id']} blocks_in_snapshot={len(snap.get('blocks', []))}")

    print("\n== POST /api/reclaim  (token_budget=60, compress=True)")
    rc = post("/api/reclaim", {
        "session_id": "demo-session",
        "snapshot_id": bt["snapshot_id"],
        "token_budget": 60,
        "compress_if_over": True,
    })
    print(f"  injected={rc['blocks_injected']} skipped={rc['blocks_skipped']} "
          f"tokens={rc['tokens_injected']} compressed={rc['compressed']}")
    if rc.get("compression_summary"):
        print(f"  compression: {rc['compression_summary'][:120]}…")

    print("\n== GET /api/prompts")
    print(json.dumps([p["name"] for p in get("/api/prompts")]))

    print("\nAll endpoints OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
