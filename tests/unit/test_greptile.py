from __future__ import annotations

import httpx
import pytest
import respx

from mcp_agent_system.integrations.greptile import GreptileClient


@pytest.mark.asyncio
async def test_index_repo_returns_repo_id() -> None:
    client = GreptileClient(
        api_key="k", github_token="ghp_x", base_url="https://api.example/v2"
    )
    with respx.mock:
        respx.post("https://api.example/v2/repositories").mock(
            return_value=httpx.Response(202, json={"status": "queued"})
        )
        repo_id = await client.index_repo(
            remote="github", repository="owner/name", branch="main"
        )
    assert repo_id == "github:owner/name:main"


@pytest.mark.asyncio
async def test_query_returns_answer_and_sources() -> None:
    client = GreptileClient(
        api_key="k", github_token="ghp_x", base_url="https://api.example/v2"
    )
    with respx.mock:
        respx.post("https://api.example/v2/query").mock(
            return_value=httpx.Response(
                200,
                json={
                    "message": "It does X by calling Y.",
                    "sources": [{"path": "src/x.py", "lines": "10-20"}],
                },
            )
        )
        out = await client.query(
            question="how does X work?",
            repositories=[
                {"remote": "github", "repository": "owner/name", "branch": "main"}
            ],
            session_id="sess",
        )
    assert "X by calling Y" in out.answer
    assert out.sources[0]["path"] == "src/x.py"
    assert out.repo_id == "github:owner/name:main"
