"""Tiny liveness probe: spawn the MCP server over stdio, run the
initialize handshake, list tools/resources/prompts, then disconnect.

Confirms the wire protocol is healthy end-to-end without standing up
a real client.

    python scripts/stdio_handshake.py
"""

from __future__ import annotations

import asyncio
import sys


async def main() -> None:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "context_retrieval", "--transport", "stdio"],
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            print(f"server: {init.serverInfo.name} v{init.serverInfo.version}")
            tools = await session.list_tools()
            print(f"tools ({len(tools.tools)}): " + ", ".join(t.name for t in tools.tools))
            resources = await session.list_resource_templates()
            print(
                f"resource templates ({len(resources.resourceTemplates)}): "
                + ", ".join(r.uriTemplate for r in resources.resourceTemplates)
            )
            prompts = await session.list_prompts()
            print(
                f"prompts ({len(prompts.prompts)}): "
                + ", ".join(p.name for p in prompts.prompts)
            )


if __name__ == "__main__":
    asyncio.run(main())
