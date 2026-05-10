"""Seed dev data: a single demo session and one correction.

Usage::

    python scripts/seed.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("DATABASE_URL", f"sqlite:///{ROOT}/dev.db")

from mcp_agent_system.agents.orchestrator import AgentOrchestrator  # noqa: E402
from mcp_agent_system.db.session import init_engine  # noqa: E402
from mcp_agent_system.schemas.tools import (  # noqa: E402
    LogCorrectionInput,
    SendPromptInput,
    StartSessionInput,
)
from mcp_agent_system.tools.log_correction import handle_log_correction  # noqa: E402
from mcp_agent_system.tools.send_prompt import handle_send_prompt  # noqa: E402
from mcp_agent_system.tools.start_session import handle_start_session  # noqa: E402


async def run() -> None:
    init_engine()
    orchestrator = AgentOrchestrator()
    started = await handle_start_session(
        StartSessionInput(user_id="seed_user"), orchestrator
    )
    print(f"Created session: {started.session_id}")
    try:
        await handle_send_prompt(
            SendPromptInput(session_id=started.session_id, message="Hello there"),
            orchestrator,
        )
        await handle_log_correction(
            LogCorrectionInput(
                session_id=started.session_id,
                original="The capital of France is London.",
                correction="The capital of France is Paris.",
                severity="high",
                category="factual",
            ),
            orchestrator,
        )
        print("Seed data ready.")
    except Exception as exc:  # noqa: BLE001
        print(f"(skipped LLM-dependent seed: {exc})")


if __name__ == "__main__":
    asyncio.run(run())
