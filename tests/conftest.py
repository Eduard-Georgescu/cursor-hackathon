"""Shared pytest fixtures."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from context_retrieval.config import Settings
from context_retrieval.models import Block, Chain, Session
from context_retrieval.store import SQLiteStore


SAMPLE_BLOCKS: list[tuple[str, str]] = [
    ("user", "I'm building a Next.js dashboard and the auth flow is broken."),
    ("agent", "Let's look at the middleware. Share your middleware.ts."),
    ("user", "It checks for a 'session' cookie and redirects on miss."),
    ("agent", "Set sameSite: 'lax' on the cookie in NextAuth config."),
    ("tool", "[build] pnpm typecheck -> 0 errors."),
    ("user", "Now let's add a dark mode toggle on the settings page."),
    ("agent", "I'll install next-themes and add a Toggle component."),
    ("user", "The chart on /analytics doesn't respect dark mode."),
    ("agent", "Updating Recharts config to read CSS variables instead of hex."),
    ("user", "Great, please also fix the sidebar contrast for WCAG AA."),
]


@pytest.fixture
def store(tmp_path: Path) -> SQLiteStore:
    db_path = tmp_path / "test.db"
    return SQLiteStore(db_path)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(db_path=tmp_path / "test.db")


@pytest.fixture
def session_id() -> str:
    return "test-session"


@pytest.fixture
def chain_id() -> str:
    return "test-chain"


@pytest.fixture
def seeded_store(
    store: SQLiteStore, session_id: str, chain_id: str
) -> SQLiteStore:
    store.upsert_session(
        Session(session_id=session_id, active_chain_id=chain_id, lookback_limit=50)
    )
    store.upsert_chain(
        Chain(chain_id=chain_id, session_id=session_id, title="Test chain")
    )
    base = datetime.now(timezone.utc) - timedelta(hours=1)
    for i, (role, content) in enumerate(SAMPLE_BLOCKS):
        store.upsert_block(
            Block(
                block_id=f"b{i:02d}",
                chain_id=chain_id,
                sequence=i,
                role=role,  # type: ignore[arg-type]
                content=content,
                created_at=base + timedelta(minutes=i),
            )
        )
    return store
