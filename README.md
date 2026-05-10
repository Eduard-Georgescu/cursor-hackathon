# MCP Server 3: Context Retrieval

A production-grade FastMCP server that lets an Active Agent search across its own past, summarize what's happened in plain language, and rewind to a previous moment in the conversation.

It exposes **6 tools**, **2 resources**, and **3 prompts** built on top of a shared SQLite store that Servers 1 & 2 read and write into. Everything works zero-deps (extractive summaries + hashed-TF-IDF semantic search) and gets sharper when you add `sentence-transformers` and an `ANTHROPIC_API_KEY`.

## Quick start

```bash
# Python 3.10+
pip install -e .                  # zero-deps install
pip install -e ".[all]"           # + sentence-transformers + anthropic + pytest

python scripts/seed_demo.py       # seed a realistic demo chain
python scripts/smoke_test.py      # exercise all 6 tools end-to-end

python -m context_retrieval       # run the MCP server over stdio
```

To wire into Cursor, drop the contents of [docs/cursor-mcp.example.json](docs/cursor-mcp.example.json) into `~/.cursor/mcp.json`.

## Demo website

A single-page browser UI that exercises every tool, both resources, and all three prompts is included. It hits a thin FastAPI bridge (`src/context_retrieval/web.py`) that calls the same `Store` / `SemanticIndex` / `Summarizer` code the MCP tools use — no MCP transport required.

```bash
pip install -e ".[web]"           # fastapi + uvicorn
python scripts/run_demo.py        # seeds demo data, opens http://localhost:8765
```

Flags: `--port 9000`, `--no-seed`, `--no-open`, `--reload`, `--db path/to/file.db`.

What's on the page:

- **Left pane — chain timeline.** All blocks in the active chain. Click to scan adjacent neighbors. Double-click to preview a backtrack from that block.
- **Center pane — `query_chain` + `scan_adjacent`.** Free-text search with mode (`hybrid` / `semantic` / `lexical`), `top_k`, `min_relevance`, role filters, MMR diversification, and `explain` toggle. Each result shows a score breakdown (semantic / lexical / recency) and a context hint. Clicking a result populates the adjacent-block view.
- **Right pane — `summarize_history`, `backtrack`, `reclaim_context`, prompts.** Generate skim / standard / deep summaries (cached or `force_refresh`). Themes, decisions, open questions, and clickable checkpoints are rendered. Clicking a checkpoint opens a dry-run backtrack modal; confirming writes the snapshot and auto-runs `reclaim_context` with an optional token budget + compression.
- **Top bar — `set_lookback`.** Slider + Save persists the lookback for the current session.
- **Prompt cards.** The three MCP prompts (`recap_history`, `find_related_context`, `restore_to_decision`) are exposed as one-click recipes that wire the underlying tool sequence.

Verify the API end-to-end without the browser:

```bash
python scripts/verify_web.py
```

## Architecture

```
┌─────────────────┐                     ┌─────────────────────────────┐
│   MCP Client    │ ── stdio / sse ───▶ │   Server 3: context-retrieval│
│ Cursor / app    │ ◀── notifications ──│  ┌────────────────────────┐  │
└─────────────────┘                     │  │ Tools  Resources Prompts│  │
                                        │  └──────────┬──────────────┘  │
                                        │  ┌──────────▼──────────────┐  │
                                        │  │ Hybrid Search           │  │
                                        │  │  semantic + BM25 + MMR  │  │
                                        │  └──────────┬──────────────┘  │
                                        │  ┌──────────▼──────────────┐  │
                                        │  │ Multi-level Summarizer  │  │
                                        │  │  Claude / extractive    │  │
                                        │  └──────────┬──────────────┘  │
                                        │  ┌──────────▼──────────────┐  │
                                        │  │ Pluggable Store         │  │
                                        │  └──────────┬──────────────┘  │
                                        └─────────────┼─────────────────┘
                                                      │
                                              ┌───────▼────────┐
                                              │ Shared SQLite  │
                                              │ data/context.db│
                                              └───────┬────────┘
                                                      │
                                            ┌─────────┴─────────┐
                                            │   Server 1 / 2    │
                                            │ chains / blocks   │
                                            └───────────────────┘
```

Shared SQLite is the integration contract for the hackathon. The `Store` Protocol lets you swap in Postgres or Redis later without touching tool code. See [docs/INTEGRATION.md](docs/INTEGRATION.md) for the schema Servers 1 & 2 write into.

## Tools

| Tool | What it does |
| --- | --- |
| `query_chain` | Hybrid semantic + lexical search over a chain's blocks. Optional `mode`, `roles`, `time_range`, `min_relevance`, `diversify`, `explain`. |
| `scan_adjacent` | Fetch neighbors around a matched block. Asymmetric `before_window` / `after_window`, optional `roles` filter, optional `max_tokens` budget. |
| `set_lookback` | Cap how far back the agent searches. `scope="session"` (default) or `scope="chain"`. |
| `summarize_history` | Plain-language recap with themes, decisions, open questions, and reasoned checkpoints. `detail_level` is `"skim"` / `"standard"` / `"deep"`. |
| `backtrack` | Freeze the chain prefix at a checkpoint or block. Supports `dry_run=True` so the UI can preview. |
| `reclaim_context` | Re-inject a snapshot. Honors `token_budget` and (when `compress_if_over=True`) compresses dropped blocks via Claude. |

### Tool details

#### `query_chain(chain_id, prompt, *, lookback_limit?, session_id?, top_k=10, mode="hybrid", diversify=False, roles?, time_range?, min_relevance=0.0, explain=False)`

Hybrid retrieval combines a semantic leg (cosine over embeddings) with a lexical leg (SQLite FTS5 BM25) using Reciprocal Rank Fusion, then a recency boost favors fresher blocks. With `diversify=True`, results are re-ranked with MMR (λ=0.7) so you don't get five neighboring blocks back.

```jsonc
// Match with explain=True
{
  "block_id": "b03",
  "chain_id": "c1",
  "sequence": 3,
  "role": "agent",
  "relevance": 0.7421,
  "preview": "Set sameSite='lax' on the auth cookie ...",
  "score_breakdown": { "semantic": 0.81, "lexical": 0.66, "recency": 0.92, "fused": 0.74 },
  "context_hint": "matches on: cookie, samesite — Set sameSite='lax' ..."
}
```

#### `scan_adjacent(block_id, *, window=1, before_window?, after_window?, roles?, max_tokens?)`

Returns the anchor block plus its neighbors. `merged_context` is the formatted plain-text concatenation ready to hand to the agent. When `max_tokens` is set, the oldest blocks are dropped first; the anchor is never dropped unless that's the only way to fit.

#### `set_lookback(session_id, limit, *, scope="session", chain_id?)`

`scope="session"` writes to `sessions.lookback_limit`. `scope="chain"` writes to a per-chain override that takes precedence when both exist. Returns `previous_limit` and `updated_limit`.

#### `summarize_history(chain_id, session_id, *, detail_level="standard", force_refresh=False, model?)`

Three detail levels:

- **skim** — 2-3 paragraph recap + 3-5 checkpoints (one Claude call).
- **standard** (default) — recap + 4-10 checkpoints + themes + top decisions (one Claude call).
- **deep** — two-pass: pass 1 extracts themes/decisions/anchor_candidates; pass 2 narrates around themes with 6-12 reasoned checkpoints (`reason`, `confidence`, `category`).

Cached by `(chain_id, detail_level, block_count)`. Re-call with the same chain length returns the cached summary unless `force_refresh=True`. Every successful call fires `notifications/resources/updated` for `retrieval://history/{session_id}`.

#### `backtrack(session_id, *, checkpoint_id?, block_id?, name?, reason?, dry_run=False)`

Provide either `checkpoint_id` (the UI path) or `block_id` (advanced). Returns a manifest of what would be / was frozen: block count, token estimate, per-role counts. With `dry_run=True` nothing is persisted.

#### `reclaim_context(session_id, snapshot_id, *, token_budget?, roles?, compress_if_over=True)`

Drops oldest blocks until the snapshot fits `token_budget`. With `compress_if_over=True`, the dropped blocks are summarized (skim level) into one synthetic prefix string so nothing is silently lost. Updates `sessions.active_chain_id` so subsequent tool calls see the restored chain.

## Resources

| URI | Body |
| --- | --- |
| `retrieval://history/{session_id}` | Latest plain-language history as JSON (themes, decisions, open_questions, checkpoints with reasoning). |
| `retrieval://snapshot/{point_id}` | Frozen agent state at a chain point: manifest, provenance, lookback, blocks. |

## Prompts

| Prompt | Use |
| --- | --- |
| `recap_history(session_id, detail_level)` | One-click recap of the session, narrated to the user. |
| `find_related_context(chain_id, topic)` | Hybrid search a topic and explain the top matches. |
| `restore_to_decision(session_id, decision_query)` | Find a decision moment and walk through `backtrack` → `reclaim_context`. |

## Environment variables

| Var | Default | Notes |
| --- | --- | --- |
| `CONTEXT_RETRIEVAL_DB` | `./data/context.db` | Path to the shared SQLite database. |
| `ANTHROPIC_API_KEY` | _(unset)_ | When set, summaries use Claude. |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-5` | Model id for Claude summaries. |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model. Falls back to hashed-TF-IDF if missing. |
| `LOG_LEVEL` | `INFO` | Python log level. Logs go to stderr. |
| `MAX_LOOKBACK` | `1000` | Upper bound on `set_lookback`. |
| `MAX_BLOCKS_TO_SUMMARIZE` | `200` | Cap on blocks per Claude summarize call. |
| `MAX_CHARS_PER_BLOCK` | `1200` | Per-block truncation before summarization. |

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The suite covers store CRUD + FTS sync, hybrid search ordering + MMR + filters, summarizer hallucination guards + detail-level contracts, and in-process tool calls (dry_run, token_budget, cache hits).

## Project layout

```
src/context_retrieval/
  config.py         Settings, env-driven
  logging_utils.py  Structured per-call logger
  models.py         Pydantic models (Block, Match, Summary, Snapshot, ...)
  store.py          Store Protocol + SQLiteStore + FTS5 + embedding cache
  search.py         Hybrid retrieval (semantic + BM25 + RRF + MMR)
  prompts.py        Claude system prompts (skim / standard / deep)
  summarizer.py     Multi-level summarizer with extractive fallback
  server.py         FastMCP wiring: 6 tools, 2 resources, 3 prompts
  __main__.py       CLI entrypoint (stdio / sse / streamable-http)

scripts/
  seed_demo.py      Realistic demo chain
  smoke_test.py     End-to-end exercise of every tool

tests/
  conftest.py
  test_store.py
  test_search.py
  test_summarizer.py
  test_server_tools.py

docs/
  INTEGRATION.md             Shared SQLite contract for Servers 1 & 2
  cursor-mcp.example.json    Drop-in Cursor MCP client config
```
