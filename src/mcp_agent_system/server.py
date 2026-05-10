"""MCP Agent System — server entrypoint.

Exposes 8 tools and 3 resources over either ``stdio`` (default, used by
Claude Desktop) or ``sse`` (used by remote MCP hosts).

Tools:
    start_session, send_prompt, get_context_usage, log_correction,
    transfer_context, get_session_state, query_codebase, create_invoice

Resources:
    agent://session/{id}, agent://corrections/{id}, agent://codebase/{session_id}

Sponsor integrations are wired here:
    * CLōD     - LLM gateway (replaces Anthropic SDK)
    * Greptile - codebase search (query_codebase tool)
    * AllScale - stablecoin invoicing (create_invoice tool)
    * BGA      - on-chain correction registry (log_correction + transfer_context)
"""

from __future__ import annotations

import json
from typing import Any

from mcp.server.fastmcp import FastMCP

from .agents.orchestrator import AgentOrchestrator
from .db.session import init_engine
from .integrations.allscale import AllScaleClient
from .integrations.chain import ChainClient
from .integrations.clod import ClodClient
from .integrations.greptile import GreptileClient
from .resources.codebase_resource import render_codebase_payload
from .resources.corrections_resource import render_corrections_payload
from .resources.session_resource import render_session_payload
from .schemas.tools import (
    CreateInvoiceInput,
    GetContextUsageInput,
    GetSessionStateInput,
    LogCorrectionInput,
    QueryCodebaseInput,
    SendPromptInput,
    StartSessionInput,
    TransferContextInput,
)
from .tools.create_invoice import handle_create_invoice
from .tools.get_context_usage import handle_get_context_usage
from .tools.get_session_state import handle_get_session_state
from .tools.log_correction import handle_log_correction
from .tools.query_codebase import handle_query_codebase
from .tools.send_prompt import handle_send_prompt
from .tools.start_session import handle_start_session
from .tools.transfer_context import handle_transfer_context
from .utils.env import get_settings
from .utils.logger import configure_logging, get_logger


def build_server() -> tuple[FastMCP, AgentOrchestrator, dict[str, Any]]:
    """Construct the MCP server with sponsor clients wired in.

    Returns the FastMCP instance, the orchestrator, and a dict of sponsor
    clients so callers (tests, scripts) can introspect or replace them.
    """
    settings = get_settings()
    configure_logging(settings.log_level)
    log = get_logger("mcp_agent_system.server")

    init_engine(settings.database_url)

    clod = ClodClient()
    chain = ChainClient()
    greptile = GreptileClient()
    allscale = AllScaleClient()

    orchestrator = AgentOrchestrator(clod=clod, chain=chain)

    log.info(
        "boot.sponsors_wired",
        clod_active_model=clod.active_model,
        clod_summary_model=clod.summary_model,
        greptile_base=greptile.base_url,
        allscale_base=allscale.base_url,
        allscale_signed=bool(allscale.api_key and allscale.api_secret),
        bga_chain_id=chain.chain_id,
        bga_enabled=chain.enabled,
    )

    recovered = orchestrator.rehydrate_active_sessions(
        transfer_threshold=settings.context_transfer_threshold
    )
    if recovered:
        log.info("boot.recovered_sessions", session_ids=recovered)

    server = FastMCP(
        name="mcp-agent-system",
        instructions=(
            "Agent System MCP server. Call start_session first, then send_prompt, "
            "log_correction, query_codebase (Greptile), transfer_context, "
            "create_invoice (AllScale), and get_session_state. Corrections are "
            "anchored on-chain via the BGA CorrectionRegistry contract."
        ),
    )

    # ── Tools ────────────────────────────────────────────────────────────

    @server.tool()
    async def start_session(
        user_id: str,
        model: str = "anthropic/claude-opus-4",
        system_prompt: str | None = None,
    ) -> str:
        """Initialize a new ActiveAgent + HistoryAgent pair for a user session."""
        out = await handle_start_session(
            StartSessionInput(
                user_id=user_id, model=model, system_prompt=system_prompt
            ),
            orchestrator,
        )
        return out.model_dump_json()

    @server.tool()
    async def send_prompt(session_id: str, message: str) -> str:
        """Route a user message to the Active Agent. Auto-fires transfer at threshold."""
        out = await handle_send_prompt(
            SendPromptInput(session_id=session_id, message=message), orchestrator
        )
        return out.model_dump_json()

    @server.tool()
    async def get_context_usage(session_id: str) -> str:
        """Return the current token consumption stats for the session."""
        out = await handle_get_context_usage(
            GetContextUsageInput(session_id=session_id), orchestrator
        )
        return out.model_dump_json()

    @server.tool()
    async def log_correction(
        session_id: str,
        original: str,
        correction: str,
        severity: str = "low",
        category: str | None = None,
    ) -> str:
        """Record a user correction. Hashed proof is anchored on the BGA registry."""
        out = await handle_log_correction(
            LogCorrectionInput(
                session_id=session_id,
                original=original,
                correction=correction,
                severity=severity,  # type: ignore[arg-type]
                category=category,  # type: ignore[arg-type]
            ),
            orchestrator,
            chain,
        )
        return out.model_dump_json()

    @server.tool()
    async def transfer_context(session_id: str) -> str:
        """Trigger the context handoff protocol manually."""
        out = await handle_transfer_context(
            TransferContextInput(session_id=session_id), orchestrator
        )
        return out.model_dump_json()

    @server.tool()
    async def get_session_state(session_id: str) -> str:
        """Return a complete snapshot of a session."""
        out = await handle_get_session_state(
            GetSessionStateInput(session_id=session_id)
        )
        return out.model_dump_json()

    @server.tool()
    async def query_codebase(
        session_id: str,
        question: str,
        repo_remote: str | None = None,
        repository: str | None = None,
        branch: str = "main",
    ) -> str:
        """Ask a Greptile-grounded question about a connected GitHub repo."""
        out = await handle_query_codebase(
            QueryCodebaseInput(
                session_id=session_id,
                question=question,
                repo_remote=repo_remote,
                repository=repository,
                branch=branch,
            ),
            orchestrator,
            greptile,
        )
        return out.model_dump_json()

    @server.tool()
    async def create_invoice(session_id: str, stablecoin: str | None = None) -> str:
        """Create an AllScale stablecoin invoice for the session's token usage."""
        out = await handle_create_invoice(
            CreateInvoiceInput(session_id=session_id, stablecoin=stablecoin),
            orchestrator,
            allscale,
        )
        return out.model_dump_json()

    # ── Resources ────────────────────────────────────────────────────────

    @server.resource("agent://session/{session_id}")
    def session_resource(session_id: str) -> str:
        return json.dumps(render_session_payload(session_id), indent=2)

    @server.resource("agent://corrections/{session_id}")
    def corrections_resource(session_id: str) -> str:
        return json.dumps(render_corrections_payload(session_id), indent=2)

    @server.resource("agent://codebase/{session_id}")
    def codebase_resource(session_id: str) -> str:
        return json.dumps(render_codebase_payload(session_id), indent=2)

    return server, orchestrator, {
        "clod": clod,
        "greptile": greptile,
        "allscale": allscale,
        "chain": chain,
    }


def main() -> None:
    """CLI entrypoint. Selects stdio or SSE based on ``MCP_TRANSPORT``."""
    settings = get_settings()
    server, _, _ = build_server()
    if settings.mcp_transport == "sse":
        import uvicorn

        app = server.sse_app()
        uvicorn.run(app, host=settings.sse_host, port=settings.sse_port)
    else:
        import anyio

        anyio.run(server.run_stdio_async)


if __name__ == "__main__":
    main()
