"""Console entrypoint: `python -m context_retrieval` (or `mcp-context-retrieval`).

Runs the FastMCP server over stdio by default. Use `--transport sse` or
`--transport streamable-http` for HTTP-based transports.
"""

from __future__ import annotations

import argparse

from .server import mcp


def main() -> None:
    parser = argparse.ArgumentParser(description="MCP Server 3: Context Retrieval")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="MCP transport (default: stdio)",
    )
    args = parser.parse_args()
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
