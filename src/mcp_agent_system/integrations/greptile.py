"""Greptile codebase-search client.

Greptile (https://greptile.com) provides natural-language semantic search
over indexed repositories. We expose two operations the MCP agent uses:

    1. ``index_repo``  - kicks off (or refreshes) indexing of a repo.
    2. ``query``       - asks a natural-language question grounded in the repo.

Repos are referenced by ``{remote}:{repository}:{branch}`` per Greptile's
docs (e.g. ``github:owner/name:main``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from ..utils.env import get_settings
from ..utils.logger import get_logger
from ..utils.retry import with_retry

log = get_logger(__name__)


@dataclass
class GreptileAnswer:
    answer: str
    sources: list[dict[str, Any]]
    repo_id: str


class GreptileClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        github_token: str | None = None,
        base_url: str | None = None,
    ) -> None:
        s = get_settings()
        self.api_key = api_key or s.greptile_api_key
        self.github_token = github_token or s.greptile_github_token
        self.base_url = (base_url or s.greptile_base_url).rstrip("/")

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.github_token:
            headers["X-GitHub-Token"] = self.github_token
        return headers

    @staticmethod
    def repo_id(remote: str, repository: str, branch: str) -> str:
        return f"{remote}:{repository}:{branch}"

    async def index_repo(
        self, *, remote: str, repository: str, branch: str = "main"
    ) -> str:
        """Submit a repo for indexing. Returns the canonical repo_id."""
        body = {
            "remote": remote,
            "repository": repository,
            "branch": branch,
            "reload": False,
            "notify": False,
        }

        async def _do() -> str:
            async with httpx.AsyncClient(timeout=60.0) as http:
                resp = await http.post(
                    f"{self.base_url}/repositories",
                    headers=self._headers(),
                    json=body,
                )
                if resp.status_code in (409, 200, 201, 202):
                    log.info(
                        "greptile.index", status=resp.status_code, repo=repository
                    )
                    return self.repo_id(remote, repository, branch)
                resp.raise_for_status()
                return self.repo_id(remote, repository, branch)

        return await with_retry(_do)

    async def query(
        self,
        *,
        question: str,
        repositories: list[dict[str, str]],
        session_id: str,
        genius: bool = True,
    ) -> GreptileAnswer:
        """Run a natural-language query against one or more indexed repos."""
        body = {
            "messages": [{"id": session_id, "content": question, "role": "user"}],
            "repositories": repositories,
            "sessionId": session_id,
            "genius": genius,
            "stream": False,
        }

        async def _do() -> dict[str, Any]:
            async with httpx.AsyncClient(timeout=120.0) as http:
                resp = await http.post(
                    f"{self.base_url}/query",
                    headers=self._headers(),
                    json=body,
                )
                resp.raise_for_status()
                return resp.json()

        data = await with_retry(_do)
        repo = repositories[0]
        return GreptileAnswer(
            answer=str(data.get("message") or data.get("answer") or ""),
            sources=list(data.get("sources") or data.get("citations") or []),
            repo_id=self.repo_id(
                repo["remote"], repo["repository"], repo.get("branch", "main")
            ),
        )
