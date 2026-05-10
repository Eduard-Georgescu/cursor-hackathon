"""``query_codebase`` tool handler.

Powered by Greptile (https://greptile.com). Lets the Active Agent ask a
natural-language question grounded in a connected GitHub repository.

On first call for a session, the supplied ``repo_remote`` / ``repository``
are submitted to Greptile for indexing and remembered in the
``codebase_links`` table. Subsequent calls in the same session may omit
those fields and re-use the cached repo.
"""

from __future__ import annotations

import uuid

from ..agents.orchestrator import AgentOrchestrator
from ..db.repositories.codebase_repo import CodebaseLinkRepo
from ..db.repositories.session_repo import SessionRepo
from ..db.session import session_scope
from ..integrations.greptile import GreptileClient
from ..schemas.tools import QueryCodebaseInput, QueryCodebaseOutput


async def handle_query_codebase(
    payload: QueryCodebaseInput,
    orchestrator: AgentOrchestrator,
    greptile: GreptileClient | None = None,
) -> QueryCodebaseOutput:
    with session_scope() as db:
        if SessionRepo.find(db, payload.session_id) is None:
            raise ValueError(f"Session not found: {payload.session_id}")

    client = greptile or GreptileClient()
    indexed_now = False

    repo_remote = payload.repo_remote
    repository = payload.repository
    branch = payload.branch

    with session_scope() as db:
        existing = CodebaseLinkRepo.latest_for_session(db, payload.session_id)
        if existing is not None and not (repo_remote and repository):
            repo_remote = existing.repo_remote.split(":", 1)[0]
            repository = existing.repo_remote.split(":", 1)[1] if ":" in existing.repo_remote else existing.repo_remote
            branch = existing.repo_branch
            link_id = existing.link_id
            greptile_repo_id = existing.greptile_repo_id
        else:
            if not (repo_remote and repository):
                raise ValueError(
                    "First call for this session must include 'repo_remote' and 'repository'."
                )
            link_id = str(uuid.uuid4())
            greptile_repo_id = ""

    if not greptile_repo_id:
        greptile_repo_id = await client.index_repo(
            remote=repo_remote, repository=repository, branch=branch
        )
        indexed_now = True
        with session_scope() as db:
            CodebaseLinkRepo.upsert(
                db,
                link_id=link_id,
                session_id=payload.session_id,
                repo_remote=f"{repo_remote}:{repository}",
                repo_branch=branch,
                greptile_repo_id=greptile_repo_id,
            )

    answer = await client.query(
        question=payload.question,
        repositories=[
            {"remote": repo_remote, "repository": repository, "branch": branch}
        ],
        session_id=payload.session_id,
    )

    with session_scope() as db:
        CodebaseLinkRepo.record_query(
            db, link_id, query=payload.question, answer=answer.answer
        )

    return QueryCodebaseOutput(
        answer=answer.answer,
        sources=answer.sources,
        repo_id=answer.repo_id,
        indexed=indexed_now,
    )
