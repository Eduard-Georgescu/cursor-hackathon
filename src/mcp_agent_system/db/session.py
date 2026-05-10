"""Database engine + session factory.

Uses synchronous SQLAlchemy 2.0; tool handlers call repositories via
``with session_scope() as s: ...``. SQLite is the default; PostgreSQL is
selected automatically by the URL scheme.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session as OrmSession
from sqlalchemy.orm import sessionmaker

from .models import Base

_engine: Engine | None = None
_SessionFactory: sessionmaker[OrmSession] | None = None


def _build_engine(url: str) -> Engine:
    connect_args: dict[str, object] = {}
    if url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    return create_engine(url, future=True, connect_args=connect_args)


def init_engine(url: str | None = None) -> Engine:
    """Initialize (or replace) the global engine. Call once at startup or in tests."""
    global _engine, _SessionFactory
    resolved = url or os.environ.get("DATABASE_URL", "sqlite:///./dev.db")
    _engine = _build_engine(resolved)
    _SessionFactory = sessionmaker(
        bind=_engine, autoflush=False, autocommit=False, expire_on_commit=False
    )
    return _engine


def get_engine() -> Engine:
    if _engine is None:
        return init_engine()
    return _engine


@contextmanager
def session_scope() -> Iterator[OrmSession]:
    """Transactional session context. Commits on success, rolls back on error."""
    if _SessionFactory is None:
        init_engine()
    assert _SessionFactory is not None
    session = _SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def create_all() -> None:
    """Create all tables from the metadata. Used by tests and quick-start mode."""
    Base.metadata.create_all(get_engine())


def drop_all() -> None:
    Base.metadata.drop_all(get_engine())
