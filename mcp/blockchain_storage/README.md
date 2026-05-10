# Blockchain MCP (PRMD2)

Python **stdio MCP server** implementing **MCP Server 2: Blockchain Storage** — tools for append / validate / commit / edit / cancel blocks and JSON-backed chain metadata.

## Install

From this directory:

```bash
pip install -e .
```

Optional playground UI:

```bash
pip install -e ".[dev-ui]"
```

Tests:

```bash
pip install -e ".[test]"
python -m pytest tests -q
```

## Run (stdio)

```bash
python -m blockchain_mcp
```

Or:

```bash
blockchain-mcp
```

## Cursor `mcp.json`

Use your Python interpreter path as needed:

```json
{
  "mcpServers": {
    "blockchain-storage": {
      "command": "python",
      "args": ["-m", "blockchain_mcp"],
      "cwd": "C:\\path\\to\\Cursor Hackathon\\mcp\\blockchain_storage",
      "env": {
        "DATA_DIR": "C:\\path\\to\\chain-data",
        "DEFAULT_CHAIN_ID": "my-chain",
        "MAX_BLOCKS_PER_CHAIN": "256",
        "SUBSCRIPTION_BLOCK_QUOTA": "10000"
      }
    }
  }
}
```

### Environment variables

| Variable | Meaning |
|----------|---------|
| `DATA_DIR` | Directory containing `chains.json` (created automatically). |
| `DEFAULT_CHAIN_ID` | Required for the static resource `blockchain://current` (see below). |
| `MAX_BLOCKS_PER_CHAIN` | After commit, oldest committed blocks are pruned when count exceeds this (if pruning enabled on the chain). |
| `SUBSCRIPTION_BLOCK_QUOTA` | Max committed blocks allowed before appends are rejected (`quota_remaining` in stats). |

### Tools (PRMD2)

- **`append_context`** — `chain_id`, `content`, optional `token_count` (otherwise heuristic ~4 chars/token).
- **`validate_block`** — stub checks only (normalize whitespace, empty-block error, placeholder warnings); sets validation fields on the open block.
- **`commit_block`** — seals the **open** block matching `block_id`, opens a new empty block, returns optional `pruned_block_id`.
- **`edit_block`** — uncommitted blocks only. `edits` object:
  - `content`: full replacement (optional), and/or
  - `prepend`, `append`: strings applied if `content` is omitted.
- **`cancel_block`** — replaces the matching **open** block with a fresh empty block (`success`, `chain_relinked`).
- **`get_chain_stats`** — `block_count`, `total_tokens`, `quota_remaining`, `pruning_enabled`.

### Resources

| URI | Notes |
|-----|--------|
| `blockchain://chain/{chain_id}` | Chain metadata + summary stats. |
| `blockchain://block/{block_id}` | Block payload and validation fields. |
| `blockchain://current/{chain_id}` | Current open block for that chain. |
| `blockchain://current` | Same as above using **`DEFAULT_CHAIN_ID`**. |

## Token counts

Per-block capacity is **500,000 tokens** (constant `MAX_BLOCK_TOKEN_CAPACITY`). Default counting is a **heuristic** (`len(text) // 4`, minimum rules in `blockchain_mcp/store/tokens.py`), unless callers pass explicit `token_count` on append.

## Dev playground

See [dev/playground_blockchain_mcp/README.md](dev/playground_blockchain_mcp/README.md). Streamlit drives the same `ChainStore` and tool handlers as the MCP server (no stdio bridge).
