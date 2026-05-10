"""Repository for the ``codebase_links`` table (Greptile-indexed repos)."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import desc, select
from sqlalchemy.orm import Session as OrmSession

from ..models import CodebaseLink


class CodebaseLinkRepo:
    @staticmethod
    def upsert(
        db: OrmSession,
        *,
        link_id: str,
        session_id: str,
        repo_remote: str,
        repo_branch: str,
        greptile_repo_id: str,
    ) -> CodebaseLink:
        existing = db.execute(
            select(CodebaseLink)
            .where(CodebaseLink.session_id == session_id)
            .where(CodebaseLink.repo_remote == repo_remote)
            .where(CodebaseLink.repo_branch == repo_branch)
        ).scalar_one_or_none()
        if existing is not None:
            existing.greptile_repo_id = greptile_repo_id
            existing.linked_at = datetime.now(UTC).isoformat()
            db.flush()
            return existing
        row = CodebaseLink(
            link_id=link_id,
            session_id=session_id,
            repo_remote=repo_remote,
            repo_branch=repo_branch,
            greptile_repo_id=greptile_repo_id,
            linked_at=datetime.now(UTC).isoformat(),
        )
        db.add(row)
        db.flush()
        return row

    @staticmethod
    def latest_for_session(db: OrmSession, session_id: str) -> CodebaseLink | None:
        return db.execute(
            select(CodebaseLink)
            .where(CodebaseLink.session_id == session_id)
            .order_by(desc(CodebaseLink.linked_at))
            .limit(1)
        ).scalar_one_or_none()

    @staticmethod
    def record_query(
        db: OrmSession, link_id: str, *, query: str, answer: str
    ) -> None:
        row = db.get(CodebaseLink, link_id)
        if row is not None:
            row.last_query = query
            row.last_answer = answer
            db.flush()
