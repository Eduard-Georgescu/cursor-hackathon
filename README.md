# MCP Agent System

A Python implementation of the **MCP Agent System** framework: a Model Context Protocol server that manages the full lifecycle of paired AI agent sessions (Active Agent + History Agent) with on-chain auditability and stablecoin billing.

Built for a hackathon with four sponsors wired in:

| Sponsor | Role | Where |
| --- | --- | --- |
| **CLōD** (`clod.io`) | LLM gateway — OpenAI-compatible API, energy-smart routing across 30+ models | `integrations/clod.py`, used by both agents |
| **Greptile** | Natural-language codebase search | `integrations/greptile.py`, exposed as the `query_codebase` tool and `agent://codebase/{id}` resource |
| **AllScale** | Stablecoin invoicing for token consumption | `integrations/allscale.py`, exposed as the `create_invoice` tool |
| **Blockchain for Good Alliance (BGA)** | On-chain registry of every correction (SDG 4 + SDG 16) | `contracts/CorrectionRegistry.sol`, `integrations/chain.py`, attached to every `log_correction` |

## What's exposed

**8 MCP tools**

| Tool | Description |
| --- | --- |
| `start_session` | Allocate a new ActiveAgent + HistoryAgent pair |
| `send_prompt` | Route a user message; auto-fires `transfer_context` at threshold |
| `get_context_usage` | Token-usage stats for the current window |
| `log_correction` | Persist a correction; **anchor a SHA-256 proof on the BGA registry** |
| `transfer_context` | Run the full context handoff protocol manually |
| `get_session_state` | Full session snapshot incl. `outstanding_invoices_usd` |
| `query_codebase` | **Greptile**-grounded natural-language repo Q&A |
| `create_invoice` | **AllScale** stablecoin invoice for the session's tokens |

**3 MCP resources**

| URI | Description |
| --- | --- |
| `agent://session/{id}` | Live snapshot of session state |
| `agent://corrections/{id}` | All corrections + `training_pairs` + on-chain `tx_hash` proofs |
| `agent://codebase/{session_id}` | Linked Greptile repo + last query/answer |

## Quickstart (local stdio transport)

```bash
cd mcp-agent-system
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env          # then fill in CLOD_API_KEY etc.
alembic upgrade head           # creates ./dev.db with 5 tables
python -m mcp_agent_system.server
```

Run the test suite:

```bash
pytest -q
```

## Sponsor setup

### CLōD (`clod.io`)
1. Sign up at <https://clod.io>, generate an API key.
2. Set `CLOD_API_KEY=clod-...`. The default `CLOD_BASE_URL` (`https://api.clod.io/v1`) is OpenAI-compatible.
3. Active Agent uses `CLOD_ACTIVE_MODEL` (premium, e.g. `anthropic/claude-opus-4`); History Agent uses `CLOD_SUMMARY_MODEL` (cheap or free model) — this showcases CLōD's energy-smart routing.

### Greptile (`greptile.com`)
1. Get a Greptile API key and a GitHub fine-grained token with read access to the repo you want to query.
2. Set `GREPTILE_API_KEY` and `GREPTILE_GITHUB_TOKEN`.
3. Call `query_codebase` with `repo_remote="github"`, `repository="owner/name"`, `branch="main"` on the first call; subsequent calls in the same session may omit those.

### AllScale (`allscale.io`)
1. Register a merchant at <https://app.allscale.io>, then **Settings → Commerce** to enable Checkout and reveal credentials.
2. Set both halves of the pair:
   - `ALLSCALE_API_KEY` — public identifier (sent as `X-API-Key`)
   - `ALLSCALE_API_SECRET` — secret used to sign every request with HMAC-SHA256 (shown once; cannot be retrieved later)
3. `ALLSCALE_BASE_URL` defaults to `https://openapi.allscale.io`. Currently AllScale only settles in **USDT** (`ALLSCALE_STABLECOIN=1`); minimum settlement is **0.1 USDT** which the client enforces as a floor.
4. Tune `ALLSCALE_USD_PER_1K_TOKENS` (default `0.02`); set `ALLSCALE_REDIRECT_URL` if you want users sent back to your site post-payment.
5. Call `create_invoice` to create a `POST /v1/checkout_intents/` checkout for the session's accumulated token cost. The response's `payment_url` is the hosted page the user pays on.

### BGA on-chain registry
1. Fund a hot wallet on **BNB Chain testnet** (chain id `97`) — get free testnet BNB from <https://www.bnbchain.org/en/testnet-faucet>.
2. Grab a free **Alchemy** API key from <https://dashboard.alchemy.com/> (the free tier includes 30M compute units/month and full archive access on BNB testnet) and set it as `ALCHEMY_API_KEY=...`. The default `BGA_RPC_URL=https://bnb-testnet.g.alchemy.com/v2/{ALCHEMY_API_KEY}` template is substituted at runtime.
   - Prefer a different provider? Just override `BGA_RPC_URL` with a fully-formed URL (Infura, QuickNode, Ankr, public node, etc.) — the placeholder is only used when present.
3. Set `BGA_PRIVATE_KEY=0x...` for your funded hot wallet.
4. Deploy the contract:
   ```bash
   python scripts/deploy_contract.py
   ```
5. Copy the printed address into `.env` as `BGA_CONTRACT_ADDRESS=0x...`.
6. Every `log_correction` call now writes a `CorrectionRecorded(sessionId, originalHash, correctionHash, timestamp)` event. The `tx_hash` is persisted on the row and surfaced on `agent://corrections/{id}` with an explorer URL.

If `BGA_ENABLED=false`, `ALCHEMY_API_KEY` is unset (and `BGA_RPC_URL` still contains the `{ALCHEMY_API_KEY}` placeholder), or any required field is missing, the chain client becomes a graceful no-op so tools never fail because of chain issues.

## Claude Desktop config

```json
{
  "mcpServers": {
    "agent-system": {
      "command": "/path/to/mcp-agent-system/.venv/bin/python",
      "args": ["-m", "mcp_agent_system.server"],
      "env": {
        "DATABASE_URL": "sqlite:////path/to/mcp-agent-system/dev.db",
        "CLOD_API_KEY": "clod-...",
        "GREPTILE_API_KEY": "grpt-...",
        "GREPTILE_GITHUB_TOKEN": "ghp_...",
        "ALLSCALE_API_KEY": "as_pub_...",
        "ALLSCALE_API_SECRET": "as_sec_...",
        "BGA_PRIVATE_KEY": "0x...",
        "BGA_CONTRACT_ADDRESS": "0x..."
      }
    }
  }
}
```

## SSE deployment

```bash
MCP_TRANSPORT=sse SSE_PORT=3001 python -m mcp_agent_system.server
```

Client config:

```json
{
  "mcpServers": {
    "agent-system": { "url": "http://your-host:3001/sse" }
  }
}
```

## Architecture

```
mcp-agent-system/
├── src/mcp_agent_system/
│   ├── server.py                 # FastMCP server + tool/resource registration
│   ├── tools/                    # 8 tool handlers
│   ├── resources/                # 3 resource renderers
│   ├── agents/                   # ActiveAgent, HistoryAgent, AgentOrchestrator
│   ├── integrations/             # clod / greptile / allscale / chain
│   ├── db/                       # SQLAlchemy models, session, repositories
│   ├── schemas/                  # Pydantic v2 input/output models
│   └── utils/                    # env, prompts, retry, logger, token_counter
├── contracts/CorrectionRegistry.sol
├── alembic/                      # migrations
├── scripts/
│   ├── deploy_contract.py
│   └── seed.py
└── tests/{unit,integration,e2e}/
```

## SDG framing (BGA track)

This system advances two UN Sustainable Development Goals:

- **SDG 4 — Quality Education.** Each correction logged through `log_correction` is recorded as a public, hash-verified training pair. Educational AI projects can pull `agent://corrections/{id}` to bootstrap fine-tuning datasets that are *provably* user-curated, not hallucinated.
- **SDG 16 — Peace, Justice & Strong Institutions.** The `CorrectionRegistry` contract gives every AI assistant deployment a tamper-proof audit trail. When the AI is wrong, the correction is on-chain. When the AI is challenged, the on-chain hash is the receipt. Institutions deploying AI to citizens (education, healthcare, civic services) can prove they are listening to corrections rather than ignoring them.

Privacy-first: only SHA-256 hashes — never the raw text — are written on chain.

## Hackathon judging quick links

- CLōD usage: [`integrations/clod.py`](src/mcp_agent_system/integrations/clod.py)
- Greptile usage: [`integrations/greptile.py`](src/mcp_agent_system/integrations/greptile.py) + [`tools/query_codebase.py`](src/mcp_agent_system/tools/query_codebase.py)
- AllScale usage: [`integrations/allscale.py`](src/mcp_agent_system/integrations/allscale.py) + [`tools/create_invoice.py`](src/mcp_agent_system/tools/create_invoice.py)
- BGA on-chain: [`contracts/CorrectionRegistry.sol`](contracts/CorrectionRegistry.sol) + [`integrations/chain.py`](src/mcp_agent_system/integrations/chain.py) + [`tools/log_correction.py`](src/mcp_agent_system/tools/log_correction.py)
