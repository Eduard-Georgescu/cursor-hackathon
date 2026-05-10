"""Seed the demo chain and launch the web demo on http://localhost:8765.

Usage:

    python scripts/run_demo.py
    python scripts/run_demo.py --port 9000 --host 0.0.0.0 --no-seed --no-open

Requires: `pip install -e ".[web]"` (uvicorn + fastapi).
"""

from __future__ import annotations

import argparse
import os
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the Context Retrieval demo website.")
    p.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    p.add_argument("--port", type=int, default=8765, help="Bind port (default: 8765)")
    p.add_argument(
        "--db",
        default=None,
        help="SQLite path (default: $CONTEXT_RETRIEVAL_DB or ./data/context.db)",
    )
    p.add_argument("--no-seed", action="store_true", help="Skip seeding demo data")
    p.add_argument("--no-open", action="store_true", help="Don't open the browser")
    p.add_argument("--reload", action="store_true", help="Enable uvicorn auto-reload")
    return p.parse_args()


def _maybe_seed(db_path: str) -> None:
    from context_retrieval.store import SQLiteStore

    store = SQLiteStore(db_path)
    try:
        with store._lock:  # type: ignore[attr-defined]
            row = store._conn.execute(  # type: ignore[attr-defined]
                "SELECT COUNT(*) FROM blocks"
            ).fetchone()
        count = row[0] if row else 0
    finally:
        store.close()
    if count > 0:
        print(f"[run_demo] db already has {count} block(s) — skipping seed")
        return

    print("[run_demo] seeding demo chain …")
    from scripts.seed_demo import seed  # type: ignore[import-not-found]

    session_id, chain_id = seed()
    print(f"[run_demo] seeded session={session_id} chain={chain_id}")


def main() -> int:
    args = _parse_args()

    db_path = args.db or os.environ.get("CONTEXT_RETRIEVAL_DB") or str(
        ROOT / "data" / "context.db"
    )
    os.environ["CONTEXT_RETRIEVAL_DB"] = db_path
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    if not args.no_seed:
        # Allow `from scripts.seed_demo import seed`
        sys.path.insert(0, str(ROOT))
        _maybe_seed(db_path)

    try:
        import uvicorn  # noqa: F401
    except ModuleNotFoundError:
        print(
            "[run_demo] uvicorn not installed. Install the web extra:\n"
            "    pip install -e \".[web]\"",
            file=sys.stderr,
        )
        return 1

    url = f"http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}"
    print(f"[run_demo] starting server on {url}  (db={db_path})")

    if not args.no_open:
        try:
            webbrowser.open(url)
        except Exception:  # pragma: no cover
            pass

    import uvicorn

    uvicorn.run(
        "context_retrieval.web:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
