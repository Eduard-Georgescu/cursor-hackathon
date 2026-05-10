from __future__ import annotations

import uuid

from mcp_agent_system.integrations.chain import (
    ChainClient,
    sha256_bytes32,
    uuid_to_bytes16,
)


def test_uuid_to_bytes16_valid_uuid() -> None:
    raw = uuid_to_bytes16(str(uuid.uuid4()))
    assert isinstance(raw, bytes)
    assert len(raw) == 16


def test_uuid_to_bytes16_invalid_falls_back() -> None:
    raw = uuid_to_bytes16("not-a-uuid")
    assert isinstance(raw, bytes)
    assert len(raw) == 16


def test_sha256_bytes32_length() -> None:
    assert len(sha256_bytes32("hello world")) == 32


def test_chain_disabled_when_missing_credentials(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    client = ChainClient(
        rpc_url="https://example",
        chain_id=97,
        private_key="",
        contract_address="",
        explorer_base="https://x",
    )
    assert client.enabled is False
    assert client.record_correction(session_id="s", original="a", correction="b") is None
    assert (
        client.record_context_block(session_id="s", block_id="b", summary="c") is None
    )
