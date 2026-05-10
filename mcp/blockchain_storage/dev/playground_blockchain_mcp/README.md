# Playground (dev only)

Interactive tester for `ChainStore` and PRMD2 tool handlers used by the MCP server.

## Run

**If your terminal is in the parent folder** (`Cursor Hackathon`), `dev/playground_blockchain_mcp/app.py` is the wrong path. Either:

- Run: `streamlit run mcp/blockchain_storage/dev/playground_blockchain_mcp/app.py`, or
- From repo root use: `run_blockchain_playground.ps1` / `run_blockchain_playground.bat` (in the workspace root).

From `mcp/blockchain_storage`:

```bash
pip install -e ".[dev-ui]"
streamlit run dev/playground_blockchain_mcp/app.py
```

By default JSON persistence goes to `dev/playground_blockchain_mcp/_testdata/` via `DATA_DIR`.

Override:

```bash
set PLAYGROUND_DATA_DIR=C:\path\to\folder
streamlit run dev/playground_blockchain_mcp/app.py
```

Do not point this at production `DATA_DIR`. This UI is for hackathon debugging only.
