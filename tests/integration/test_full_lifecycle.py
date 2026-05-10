"""Integration tests covering the full session lifecycle.

Each test runs against an in-memory SQLite database populated by the
``_isolated_db`` autouse fixture and uses a faked CLōD client + mocked
sponsor SDKs so no real network calls are made.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import respx

from mcp_agent_system.agents.orchestrator import AgentOrchestrator
from mcp_agent_system.integrations.allscale import AllScaleClient
from mcp_agent_system.integrations.chain import ChainTx
from mcp_agent_system.integrations.greptile import GreptileAnswer, GreptileClient
from mcp_agent_system.resources.codebase_resource import render_codebase_payload
from mcp_agent_system.resources.corrections_resource import render_corrections_payload
from mcp_agent_system.resources.session_resource import render_session_payload
from mcp_agent_system.schemas.tools import (
    CreateInvoiceInput,
    GetContextUsageInput,
    GetSessionStateInput,
    LogCorrectionInput,
    QueryCodebaseInput,
    SendPromptInput,
    StartSessionInput,
    TransferContextInput,
)
from mcp_agent_system.tools.create_invoice import handle_create_invoice
from mcp_agent_system.tools.get_context_usage import handle_get_context_usage
from mcp_agent_system.tools.get_session_state import handle_get_session_state
from mcp_agent_system.tools.log_correction import handle_log_correction
from mcp_agent_system.tools.query_codebase import handle_query_codebase
from mcp_agent_system.tools.send_prompt import handle_send_prompt
from mcp_agent_system.tools.start_session import handle_start_session
from mcp_agent_system.tools.transfer_context import handle_transfer_context


@pytest.mark.asyncio
async def test_start_session_creates_row_and_in_memory_pair(
    orchestrator: AgentOrchestrator,
) -> None:
    out = await handle_start_session(
        StartSessionInput(user_id="u1", model="anthropic/claude-opus-4"), orchestrator
    )
    assert out.session_id
    assert out.agent_config.context_limit == 200_000
    assert orchestrator.get_active(out.session_id) is not None
    assert orchestrator.get_history(out.session_id) is not None


@pytest.mark.asyncio
async def test_send_prompt_updates_tokens_and_records_exchange(
    orchestrator: AgentOrchestrator, fake_clod
) -> None:
    started = await handle_start_session(
        StartSessionInput(user_id="u1"), orchestrator
    )
    fake_clod.chat_input_tokens = 200
    fake_clod.chat_output_tokens = 80
    out = await handle_send_prompt(
        SendPromptInput(session_id=started.session_id, message="hello"), orchestrator
    )
    assert out.response == fake_clod.chat_text
    assert out.tokens_used == 280
    assert out.transfer_triggered is False

    usage = await handle_get_context_usage(
        GetContextUsageInput(session_id=started.session_id), orchestrator
    )
    assert usage.tokens_used == 280
    assert usage.transfer_threshold == pytest.approx(0.80, rel=1e-3)


@pytest.mark.asyncio
async def test_send_prompt_auto_transfers_when_threshold_crossed(
    orchestrator: AgentOrchestrator, fake_clod
) -> None:
    started = await handle_start_session(
        StartSessionInput(user_id="u1"), orchestrator
    )
    # Force a single call to push past the 80% threshold by overshooting.
    fake_clod.chat_input_tokens = 100_000
    fake_clod.chat_output_tokens = 100_000

    out = await handle_send_prompt(
        SendPromptInput(session_id=started.session_id, message="long input"),
        orchestrator,
    )
    assert out.transfer_triggered is True
    assert out.transfer_block_pointer is not None
    # After transfer, tokens_used should be reset to a small estimate.
    assert out.tokens_used < 1000

    state = await handle_get_session_state(
        GetSessionStateInput(session_id=started.session_id)
    )
    assert state.blocks_stored == 1
    assert state.block_pointer == out.transfer_block_pointer


@pytest.mark.asyncio
async def test_log_correction_writes_row_and_uses_chain_proof(
    orchestrator: AgentOrchestrator, fake_chain: MagicMock
) -> None:
    started = await handle_start_session(
        StartSessionInput(user_id="u1"), orchestrator
    )
    fake_chain.record_correction.return_value = ChainTx(
        tx_hash="0xabc123",
        chain_id=97,
        explorer_url="https://testnet.bscscan.com/tx/0xabc123",
    )
    out = await handle_log_correction(
        LogCorrectionInput(
            session_id=started.session_id,
            original="The capital of France is London.",
            correction="The capital of France is Paris.",
            severity="high",
            category="factual",
        ),
        orchestrator,
        fake_chain,
    )
    assert out.tx_hash == "0xabc123"
    assert out.chain_id == 97
    assert out.explorer_url and "0xabc123" in out.explorer_url

    state = await handle_get_session_state(
        GetSessionStateInput(session_id=started.session_id)
    )
    assert state.correction_count == 1

    payload = render_corrections_payload(started.session_id)
    assert payload["total"] == 1
    assert payload["corrections"][0]["tx_hash"] == "0xabc123"
    assert payload["corrections"][0]["chain_explorer_url"].endswith("0xabc123")
    assert payload["training_pairs"][0]["output"].endswith("Paris.")


@pytest.mark.asyncio
async def test_transfer_context_rolls_back_status_on_failure(
    orchestrator: AgentOrchestrator,
) -> None:
    started = await handle_start_session(
        StartSessionInput(user_id="u1"), orchestrator
    )
    pair = orchestrator.get_pair(started.session_id)
    assert pair is not None
    # Force the history agent to blow up mid-transfer.
    pair.history.store_context_block = AsyncMock(  # type: ignore[method-assign]
        side_effect=RuntimeError("boom")
    )
    with pytest.raises(RuntimeError):
        await handle_transfer_context(
            TransferContextInput(session_id=started.session_id), orchestrator
        )
    state = await handle_get_session_state(
        GetSessionStateInput(session_id=started.session_id)
    )
    assert state.status == "active"


@pytest.mark.asyncio
async def test_create_invoice_persists_billing_row(
    orchestrator: AgentOrchestrator, fake_clod
) -> None:
    started = await handle_start_session(
        StartSessionInput(user_id="u1"), orchestrator
    )
    fake_clod.chat_input_tokens = 200
    fake_clod.chat_output_tokens = 300
    await handle_send_prompt(
        SendPromptInput(session_id=started.session_id, message="hi"), orchestrator
    )

    allscale = AllScaleClient(
        api_key="pub",
        api_secret="sec",
        base_url="https://api.example",
        usd_per_1k_tokens=0.04,
        stablecoin=1,
    )
    with respx.mock:
        respx.post("https://api.example/v1/checkout_intents/").mock(
            return_value=httpx.Response(
                200,
                json={
                    "data": {
                        "intent_id": "ci_77",
                        "payment_url": "https://pay.example/ci_77",
                    }
                },
            )
        )
        out = await handle_create_invoice(
            CreateInvoiceInput(session_id=started.session_id),
            orchestrator,
            allscale,
        )
    assert out.invoice_ref == "ci_77"
    assert out.payment_url == "https://pay.example/ci_77"
    assert out.stablecoin == "USDT"
    assert out.usd_amount == pytest.approx(500 * 0.04 / 1000, rel=1e-3)

    state = await handle_get_session_state(
        GetSessionStateInput(session_id=started.session_id)
    )
    assert state.outstanding_invoices_usd > 0


@pytest.mark.asyncio
async def test_query_codebase_indexes_once_and_reuses(
    orchestrator: AgentOrchestrator,
) -> None:
    started = await handle_start_session(
        StartSessionInput(user_id="u1"), orchestrator
    )
    greptile = MagicMock(spec=GreptileClient)
    greptile.index_repo = AsyncMock(return_value="github:owner/repo:main")
    greptile.query = AsyncMock(
        return_value=GreptileAnswer(
            answer="It works by reading the schema.",
            sources=[{"path": "src/db/models.py"}],
            repo_id="github:owner/repo:main",
        )
    )

    first = await handle_query_codebase(
        QueryCodebaseInput(
            session_id=started.session_id,
            question="how does the schema work?",
            repo_remote="github",
            repository="owner/repo",
            branch="main",
        ),
        orchestrator,
        greptile,
    )
    assert first.indexed is True
    assert greptile.index_repo.call_count == 1

    second = await handle_query_codebase(
        QueryCodebaseInput(
            session_id=started.session_id,
            question="and the migrations?",
        ),
        orchestrator,
        greptile,
    )
    assert second.indexed is False
    assert greptile.index_repo.call_count == 1  # not called again

    payload = render_codebase_payload(started.session_id)
    assert payload["linked"] is True
    assert payload["repo_remote"] == "github:owner/repo"
    assert payload["last_query"] == "and the migrations?"


@pytest.mark.asyncio
async def test_session_resource_exposes_outstanding_invoices(
    orchestrator: AgentOrchestrator, fake_clod
) -> None:
    started = await handle_start_session(
        StartSessionInput(user_id="u1"), orchestrator
    )
    payload = render_session_payload(started.session_id)
    assert payload["session_id"] == started.session_id
    assert payload["outstanding_invoices_usd"] == 0.0


@pytest.mark.asyncio
async def test_rehydrate_active_sessions_after_restart(
    orchestrator: AgentOrchestrator, fake_clod, fake_chain: MagicMock
) -> None:
    started = await handle_start_session(
        StartSessionInput(user_id="u1"), orchestrator
    )
    # Fill some context, force a transfer, then create a fresh orchestrator.
    fake_clod.chat_input_tokens = 100_000
    fake_clod.chat_output_tokens = 100_000
    await handle_send_prompt(
        SendPromptInput(session_id=started.session_id, message="long"), orchestrator
    )
    fresh = AgentOrchestrator(clod=fake_clod, chain=fake_chain)
    recovered = fresh.rehydrate_active_sessions()
    assert started.session_id in recovered
    pair = fresh.get_pair(started.session_id)
    assert pair is not None
    # Messages from the archived context_block.raw_context are restored
    # (per doc §9.4 server-crash recovery semantics) and the History agent's
    # block_index is restored so the next transfer keeps a monotonic sequence.
    assert len(pair.active.messages) > 0
    assert pair.history.block_index >= 1
