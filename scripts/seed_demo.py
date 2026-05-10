"""Seed a realistic demo chain so Server 3 has something to retrieve.

Run once before pointing an MCP client at the server:

    python scripts/seed_demo.py

By default writes to ./data/context.db. Set CONTEXT_RETRIEVAL_DB to override.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from context_retrieval.models import Block, Chain, Session  # noqa: E402
from context_retrieval.store import SQLiteStore  # noqa: E402


DEMO_BLOCKS: list[tuple[str, str]] = [
    ("user", "I'm building a Next.js dashboard and the auth flow is broken — clicking 'Sign in' redirects back to /login instead of /dashboard."),
    ("agent", "Let's look at the middleware. Can you share the contents of `middleware.ts`?"),
    ("user", "Here it is: it checks for a 'session' cookie and redirects to /login when missing."),
    ("agent", "Got it. I suspect the cookie isn't being set on the redirect response. Try logging document.cookie right after `signIn()` returns."),
    ("tool", "[browser console] document.cookie -> '' (empty)"),
    ("agent", "Confirmed — the cookie isn't being persisted. The likely cause is a sameSite policy mismatch. Set `sameSite: 'lax'` on the cookie in your NextAuth config."),
    ("user", "Changed sameSite to 'lax'. Now I get redirected to /dashboard but it flashes /login first."),
    ("agent", "That's a hydration race — middleware runs before the client has the cookie. Move the auth check into a server component layout instead of middleware."),
    ("user", "Switching to a server component layout fixed the flash. Auth is working end to end now."),
    ("agent", "Nice. Want me to add an integration test that signs in and asserts we land on /dashboard with no /login flash?"),
    ("user", "Yes — use Playwright. And while you're at it, add a test for sign-out too."),
    ("agent", "Done. Two Playwright specs added: signin.spec.ts and signout.spec.ts. Both pass locally."),
    ("user", "Now let's switch topics — I want to add a dark mode toggle to the settings page."),
    ("agent", "Sure. Are you using `next-themes`, or should we roll our own with a context provider?"),
    ("user", "Use next-themes. The settings page is at app/settings/page.tsx."),
    ("agent", "Installed next-themes, wrapped the app in ThemeProvider, and added a Toggle component on the settings page that switches between 'light' and 'dark'."),
    ("user", "Toggle works but the chart on /analytics doesn't respect dark mode."),
    ("agent", "Chart needs a theme-aware palette. I'll update the Recharts config to read CSS variables instead of hardcoded hex values."),
    ("tool", "[build] pnpm typecheck -> 0 errors. pnpm test -> 18 passed."),
    ("user", "Looks great. Can you also fix the contrast on the sidebar in dark mode?"),
    ("agent", "Updated the sidebar tokens: background goes from slate-900 to zinc-900, text from gray-400 to gray-200. WCAG AA contrast verified."),
]


def seed() -> tuple[str, str]:
    db_path = os.environ.get("CONTEXT_RETRIEVAL_DB") or str(ROOT / "data" / "context.db")
    store = SQLiteStore(db_path)

    session_id = "demo-session"
    chain_id = "demo-chain"

    store.upsert_session(
        Session(session_id=session_id, active_chain_id=chain_id, lookback_limit=50)
    )
    store.upsert_chain(
        Chain(chain_id=chain_id, session_id=session_id, title="Auth fix, dark mode, a11y")
    )

    base_time = datetime.now(timezone.utc) - timedelta(hours=2)
    for i, (role, content) in enumerate(DEMO_BLOCKS):
        store.upsert_block(
            Block(
                block_id=f"demo-block-{i:02d}",
                chain_id=chain_id,
                sequence=i,
                role=role,  # type: ignore[arg-type]
                content=content,
                metadata={"seeded": True},
                created_at=base_time + timedelta(minutes=i * 3),
            )
        )

    store.close()
    return session_id, chain_id


def main() -> None:
    session_id, chain_id = seed()
    print(f"Seeded {len(DEMO_BLOCKS)} blocks.")
    print(f"  session_id = {session_id}")
    print(f"  chain_id   = {chain_id}")
    print()
    print("Try these tool calls from your MCP client:")
    print(f"  query_chain(chain_id='{chain_id}', prompt='cookie sameSite issue', explain=True)")
    print(f"  scan_adjacent(block_id='demo-block-05', window=2)")
    print(f"  set_lookback(session_id='{session_id}', limit=20)")
    print(f"  summarize_history(chain_id='{chain_id}', session_id='{session_id}', detail_level='standard')")
    print(f"  backtrack(session_id='{session_id}', block_id='demo-block-08', dry_run=True)")


if __name__ == "__main__":
    main()
