"""``agent://codebase/{session_id}`` resource — Greptile-linked repo state."""

from __future__ import annotations

from ..db.repositories.codebase_repo import CodebaseLinkRepo
from ..db.session import session_scope


def render_codebase_payload(session_id: str) -> dict:
    with session_scope() as db:
        row = CodebaseLinkRepo.latest_for_session(db, session_id)
        if row is None:
            return {
                "session_id": session_id,
                "linked": False,
                "message": "No codebase has been linked via query_codebase yet.",
            }
        return {
            "session_id": session_id,
            "linked": True,
            "repo_remote": row.repo_remote,
            "repo_branch": row.repo_branch,
            "greptile_repo_id": row.greptile_repo_id,
            "linked_at": row.linked_at,
            "last_query": row.last_query,
            "last_answer": row.last_answer,
        }
