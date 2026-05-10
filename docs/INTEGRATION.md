# Integration contract for Servers 1 & 2

Server 3 (Context Retrieval) reads from a shared SQLite database. Servers 1 & 2 own the **write** side of three tables; Server 3 reads them. Server 3 owns everything else.

Default path: `./data/context.db`. Override with `CONTEXT_RETRIEVAL_DB`.

## Tables owned by Servers 1 & 2 (writes here)

### `chains`

| Column | Type | Notes |
| --- | --- | --- |
| `chain_id` | TEXT PK | Stable id for a chain. |
| `session_id` | TEXT | Owning session. |
| `title` | TEXT | Optional human label shown in the UI. |
| `created_at` | TEXT | ISO 8601 UTC. |

### `sessions`

| Column | Type | Notes |
| --- | --- | --- |
| `session_id` | TEXT PK | |
| `active_chain_id` | TEXT | Server 3 updates this on `reclaim_context`. Keep it as the chain the agent is currently working on. |
| `lookback_limit` | INTEGER | Default 50. Server 3 writes this through `set_lookback`. |
| `created_at` | TEXT | ISO 8601 UTC. |

### `blocks`

| Column | Type | Notes |
| --- | --- | --- |
| `block_id` | TEXT PK | Stable id. |
| `chain_id` | TEXT | FK → `chains.chain_id`. |
| `sequence` | INTEGER | 0-based, monotonically increasing within a chain. `UNIQUE(chain_id, sequence)`. |
| `role` | TEXT | One of `user` / `agent` / `tool` / `system`. |
| `content` | TEXT | Raw text. Server 3 truncates to `MAX_CHARS_PER_BLOCK` before sending to Claude. |
| `metadata` | TEXT | JSON object. Free-form for Servers 1 & 2 — Server 3 ignores it. |
| `created_at` | TEXT | ISO 8601 UTC. |

An FTS5 mirror (`blocks_fts`) is kept current via triggers — Servers 1 & 2 just write to `blocks`; the index updates itself.

## Tables owned by Server 3 (do not write here)

- `chain_lookbacks` — per-chain lookback overrides.
- `block_embeddings` — cached embeddings keyed by `(block_id, model)`.
- `summaries`, `summary_themes`, `summary_decisions`, `summary_open_questions`, `checkpoints` — Claude-generated history.
- `snapshots` — frozen agent state for `backtrack` / `reclaim_context`.

## Recommended write order

```text
upsert chains(chain_id, session_id, ...)
upsert sessions(session_id, active_chain_id=chain_id, ...)
for each block in order:
    insert blocks(block_id, chain_id, sequence, role, content, ...)
```

Sequence numbers must be strictly increasing per chain. If you need to edit a block, write a new one with a bumped sequence rather than mutating the old row — Server 3 caches embeddings keyed on `block_id` and assumes block content is immutable for the life of that id.

## Concurrency

Server 3 opens SQLite in WAL mode with `foreign_keys=ON`. Servers 1 & 2 are free to do the same. A single shared writer is simplest; multiple writers are safe as long as transactions are small.

## Schema migrations

Server 3 issues `CREATE TABLE IF NOT EXISTS` for every table on every startup, so it's safe to point it at a database created by Servers 1 & 2 (or vice versa). Any new column added later must be additive (`ALTER TABLE ... ADD COLUMN`).
