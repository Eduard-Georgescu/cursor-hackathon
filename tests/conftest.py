"""Shared pytest fixtures.

Each test gets a fresh in-memory SQLite database and a fresh
``AgentOrchestrator`` whose CLōD client is replaced by a deterministic
fake. Sponsor integrations are stubbed by default so we never hit the
network in unit/integration tests.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest

from mcp_agent_system.agents.orchestrator import AgentOrchestrator
from mcp_agent_system.db.session import create_all, drop_all, init_engine
from mcp_agent_system.integrations.clod import ClodClient, ClodCompletion, ClodUsage


@dataclass
class FakeClod(ClodClient):
    """In-memory CLōD double. Records calls and returns canned text."""

    chat_text: str = "I'm a faked CLōD response."
    summary_text: str = "Summary: the conversation discussed a topic and reached a conclusion."
    chat_input_tokens: int = 100
    chat_output_tokens: int = 50
    calls: list[dict[str, Any]] = field(default_factory=list)
    active_model: str = "anthropic/claude-opus-4"
    summary_model: str = "meta-llama/llama-3.1-8b-instruct"
    max_output_tokens: int = 8096

    def __init__(self, **overrides: Any) -> None:  # type: ignore[no-untyped-def]
        # Skip the real ClodClient.__init__ (it builds an OpenAI client).
        self.calls = []
        self.chat_text = overrides.get("chat_text", "I'm a faked CLōD response.")
        self.summary_text = overrides.get(
            "summary_text",
            "Summary: the conversation discussed a topic and reached a conclusion.",
        )
        self.chat_input_tokens = overrides.get("chat_input_tokens", 100)
        self.chat_output_tokens = overrides.get("chat_output_tokens", 50)
        self.active_model = overrides.get("active_model", "anthropic/claude-opus-4")
        self.summary_model = overrides.get(
            "summary_model", "meta-llama/llama-3.1-8b-instruct"
        )
        self.max_output_tokens = overrides.get("max_output_tokens", 8096)

    async def chat(  # type: ignore[override]
        self,
        *,
        messages: list[dict[str, str]],
        system: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> ClodCompletion:
        self.calls.append(
            {"op": "chat", "messages": messages, "system": system, "model": model}
        )
        return ClodCompletion(
            text=self.chat_text,
            usage=ClodUsage(
                input_tokens=self.chat_input_tokens,
                output_tokens=self.chat_output_tokens,
            ),
            model=model or self.active_model,
            raw=None,
        )

    async def summarize(  # type: ignore[override]
        self, messages: list[dict[str, str]]
    ) -> ClodCompletion:
        self.calls.append({"op": "summarize", "messages": messages})
        return ClodCompletion(
            text=self.summary_text,
            usage=ClodUsage(input_tokens=20, output_tokens=20),
            model=self.summary_model,
            raw=None,
        )


@pytest.fixture(autouse=True)
def _isolated_db() -> Iterator[None]:
    init_engine("sqlite:///:memory:")
    create_all()
    try:
        yield
    finally:
        drop_all()


@pytest.fixture
def fake_clod() -> FakeClod:
    return FakeClod()


@pytest.fixture
def fake_chain() -> MagicMock:
    chain = MagicMock()
    chain.enabled = False
    chain.chain_id = 97
    chain.record_correction.return_value = None
    chain.record_context_block.return_value = None
    return chain


@pytest.fixture
def orchestrator(fake_clod: FakeClod, fake_chain: MagicMock) -> AgentOrchestrator:
    return AgentOrchestrator(clod=fake_clod, chain=fake_chain)
