"""FastMCP server exposing 6 tools, 2 resources, and 3 prompts.

Tools
    query_chain        Hybrid semantic + lexical search across blocks
    scan_adjacent      Neighbor-block fetch with merged_context formatter
    set_lookback       Configure session- or chain-scoped lookback caps
    summarize_history  Claude-generated plain-language chain recap
    backtrack          Freeze the agent state at a checkpoint (dry-run aware)
    reclaim_context    Re-inject a snapshot, with token-budget compression

Resources
    retrieval://history/{session_id}        Latest plain-language history
    retrieval://snapshot/{point_id}         Frozen agent state

Prompts
    recap_history(session_id, detail_level)
    find_related_context(chain_id, topic)
    restore_to_decision(session_id, decision_query)
"""

from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Annotated, Any, AsyncIterator, Literal

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.prompts.base import Message, UserMessage
from pydantic import AnyUrl, Field

from .config import Settings
from .logging_utils import configure_logging, get_logger, log_call
from .models import (
    AdjacentResult,
    BacktrackResult,
    Block,
    DetailLevel,
    LookbackResult,
    Provenance,
    QueryResult,
    ReclaimResult,
    Role,
    SearchMode,
    Session,
    Snapshot,
    SnapshotManifest,
    TimeRange,
)
from .search import Embedder, SemanticIndex
from .store import SQLiteStore, Store
from .summarizer import Summarizer


_log = get_logger("server")


# ---------------------------------------------------------------- lifecycle


@dataclass
class AppState:
    settings: Settings
    store: Store
    index: SemanticIndex
    summarizer: Summarizer


# Resources don't receive a Context, so we mirror the lifespan-owned state
# on a module global. The lifespan is the single writer; resource handlers
# only read.
_APP_STATE: AppState | None = None


@asynccontextmanager
async def lifespan(_: FastMCP) -> AsyncIterator[AppState]:
    global _APP_STATE
    settings = Settings.from_env()
    configure_logging(settings.log_level)
    _log.info(
        f"event=server.start db={settings.db_path} "
        f"claude={settings.claude_enabled} embedding={settings.embedding_model}"
    )
    store = SQLiteStore(settings.db_path)
    state = AppState(
        settings=settings,
        store=store,
        index=SemanticIndex(store, embedder=Embedder(settings.embedding_model)),
        summarizer=Summarizer(settings),
    )
    _APP_STATE = state
    try:
        yield state
    finally:
        _APP_STATE = None
        store.close()
        _log.info("event=server.stop")


mcp = FastMCP("context-retrieval", lifespan=lifespan)


def _state(ctx: Context) -> AppState:
    return ctx.request_context.lifespan_context  # type: ignore[return-value]


def _state_for_resource() -> AppState:
    if _APP_STATE is None:
        raise RuntimeError("server lifespan has not initialized state yet")
    return _APP_STATE


# ----------------------------------------------------------------- helpers


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_session(state: AppState, session_id: str) -> Session:
    sess = state.store.get_session(session_id)
    if sess is None:
        sess = Session(session_id=session_id)
        state.store.upsert_session(sess)
    return sess


def _block_role_counts(blocks: list[Block]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for b in blocks:
        counts[b.role] = counts.get(b.role, 0) + 1
    return counts


def _estimate_block_tokens(content: str) -> int:
    return max(1, len(content.split()))


def _format_blocks_as_text(blocks: list[Block]) -> str:
    parts = []
    for b in blocks:
        header = f"[{b.sequence:>4} | {b.role}]"
        parts.append(f"{header} {b.content.strip()}")
    return "\n\n".join(parts)


async def _notify_resource_updated(ctx: Context, uri: str) -> None:
    """Best-effort resources/updated notification.

    Failure here should never sink the tool call — for example, transports
    that don't support notifications (or test harnesses without a live
    session) should still let the tool return successfully.
    """
    try:
        await ctx.session.send_resource_updated(AnyUrl(uri))
    except Exception as exc:
        _log.debug(f"event=resource.notify.skipped uri={uri} error={exc!r}")


# --------------------------------------------------------------------- tools


@mcp.tool(
    title="Query Chain",
    description=(
        "Hybrid semantic + lexical search across the blocks of a chain. "
        "Returns ranked matches with relevance in [0,1]. Set explain=True for "
        "per-leg score breakdown and a one-line context_hint."
    ),
)
def query_chain(
    chain_id: str,
    prompt: str,
    ctx: Context,
    lookback_limit: int | None = None,
    session_id: str | None = None,
    top_k: Annotated[int, Field(ge=1, le=100)] = 10,
    mode: SearchMode = "hybrid",
    diversify: bool = False,
    roles: list[Role] | None = None,
    time_range: TimeRange | None = None,
    min_relevance: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0,
    explain: bool = False,
) -> QueryResult:
    state = _state(ctx)
    limit = (
        lookback_limit
        if lookback_limit and lookback_limit > 0
        else state.store.get_effective_lookback(
            session_id or "", chain_id, default=state.settings.max_lookback
        )
    )
    limit = min(limit, state.settings.max_lookback)

    with log_call(
        "query_chain",
        session=session_id,
        chain=chain_id,
        mode=mode,
        top_k=top_k,
    ) as fields:
        matches, total = state.index.search(
            chain_id=chain_id,
            prompt=prompt,
            lookback_limit=limit,
            top_k=top_k,
            mode=mode,
            diversify=diversify,
            roles=roles,
            time_range=time_range,
            min_relevance=min_relevance,
            explain=explain,
        )
        fields["matches"] = len(matches)
        fields["candidates"] = total

    return QueryResult(
        chain_id=chain_id,
        prompt=prompt,
        lookback_limit=limit,
        mode=mode,
        total_candidates=total,
        matches=matches,
    )


@mcp.tool(
    title="Scan Adjacent Blocks",
    description=(
        "Fetch neighboring blocks around a match to capture tasks that spanned "
        "block boundaries. Returns the blocks in order plus a merged plain-text "
        "context string ready to hand to the agent. Honors `max_tokens` by "
        "dropping the oldest blocks first."
    ),
)
def scan_adjacent(
    block_id: str,
    ctx: Context,
    window: Annotated[int, Field(ge=0, le=200)] = 1,
    before_window: int | None = None,
    after_window: int | None = None,
    roles: list[Role] | None = None,
    max_tokens: Annotated[int | None, Field(ge=1)] = None,
) -> AdjacentResult:
    state = _state(ctx)
    before = before_window if before_window is not None else window
    after = after_window if after_window is not None else window
    before = max(0, before)
    after = max(0, after)

    with log_call(
        "scan_adjacent", block=block_id, before=before, after=after
    ) as fields:
        anchor = state.store.get_block(block_id)
        if anchor is None:
            raise ValueError(f"unknown block_id: {block_id}")
        blocks = state.store.adjacent_blocks(block_id, before=before, after=after)
        if roles:
            role_set = set(roles)
            # Always keep the anchor block itself even when role-filtered.
            blocks = [b for b in blocks if b.role in role_set or b.block_id == block_id]

        total_chain_length = state.store.chain_block_count(anchor.chain_id)
        truncated = False
        if max_tokens is not None:
            blocks, dropped = _trim_to_token_budget(
                blocks, anchor_block_id=block_id, max_tokens=max_tokens
            )
            truncated = dropped > 0
            fields["dropped"] = dropped

        merged = _format_blocks_as_text(blocks)
        total_tokens = sum(_estimate_block_tokens(b.content) for b in blocks)
        fields["blocks"] = len(blocks)

    return AdjacentResult(
        block_id=block_id,
        before_window=before,
        after_window=after,
        blocks=blocks,
        merged_context=merged,
        total_tokens=total_tokens,
        total_chain_length=total_chain_length,
        truncated=truncated,
    )


@mcp.tool(
    title="Set Lookback",
    description=(
        "Configure the maximum number of blocks the Active Agent is allowed to "
        "search back inside a chain. Scope can be the session (default) or a "
        "specific chain; chain-scoped values take precedence when both exist."
    ),
)
def set_lookback(
    session_id: str,
    limit: int,
    ctx: Context,
    scope: Literal["session", "chain"] = "session",
    chain_id: str | None = None,
) -> LookbackResult:
    state = _state(ctx)
    if limit < 1 or limit > state.settings.max_lookback:
        raise ValueError(
            f"limit must be in [1, {state.settings.max_lookback}] "
            f"(MAX_LOOKBACK env var); got {limit}"
        )
    if scope == "chain" and not chain_id:
        raise ValueError("chain_id is required when scope='chain'")

    with log_call(
        "set_lookback",
        session=session_id,
        scope=scope,
        chain=chain_id,
        limit=limit,
    ) as fields:
        if scope == "session":
            prev, updated = state.store.set_session_lookback(session_id, limit)
        else:
            assert chain_id is not None
            prev, updated = state.store.set_chain_lookback(session_id, chain_id, limit)
        fields["previous"] = prev

    return LookbackResult(
        session_id=session_id,
        scope=scope,
        chain_id=chain_id,
        previous_limit=prev,
        updated_limit=updated,
    )


@mcp.tool(
    title="Summarize History",
    description=(
        "Convert the raw chain into a plain-language history with reasoned "
        "checkpoints, themes, decisions, and open questions, and store it for "
        "display in the History tab. Uses Claude when ANTHROPIC_API_KEY is set; "
        "otherwise falls back to an extractive summary. Cached by "
        "(chain_id, detail_level, block_count) unless force_refresh=True."
    ),
)
async def summarize_history(
    chain_id: str,
    session_id: str,
    ctx: Context,
    detail_level: DetailLevel = "standard",
    force_refresh: bool = False,
    model: str | None = None,
) -> dict[str, Any]:
    state = _state(ctx)
    with log_call(
        "summarize_history",
        session=session_id,
        chain=chain_id,
        detail_level=detail_level,
        force_refresh=force_refresh,
    ) as fields:
        blocks = state.store.list_chain_blocks(chain_id)
        block_count = len(blocks)
        cached = state.store.latest_summary_for_chain(chain_id, detail_level=detail_level)
        if (
            cached is not None
            and not force_refresh
            and cached.block_count == block_count
        ):
            fields["cached"] = True
            return {
                "summary_id": cached.summary_id,
                "plain_text": cached.plain_text,
                "checkpoints": [cp.model_dump(mode="json") for cp in cached.checkpoints],
                "themes": cached.themes,
                "decisions": [d.model_dump(mode="json") for d in cached.decisions],
                "open_questions": cached.open_questions,
                "engine": cached.engine,
                "detail_level": cached.detail_level,
                "generated_at": cached.created_at.isoformat(),
                "cached": True,
                "block_count": cached.block_count,
            }

        # Summarization can be CPU+IO heavy. Run it in a worker thread so we
        # don't block the FastMCP event loop.
        summary = await asyncio.to_thread(
            state.summarizer.summarize,
            chain_id,
            session_id,
            blocks,
            detail_level=detail_level,
            model_override=model,
        )
        state.store.insert_summary(summary)
        fields["cached"] = False
        fields["checkpoints"] = len(summary.checkpoints)
        fields["engine"] = summary.engine

        # Live clients re-fetch the resource on this signal.
        await _notify_resource_updated(ctx, f"retrieval://history/{session_id}")

        return {
            "summary_id": summary.summary_id,
            "plain_text": summary.plain_text,
            "checkpoints": [cp.model_dump(mode="json") for cp in summary.checkpoints],
            "themes": summary.themes,
            "decisions": [d.model_dump(mode="json") for d in summary.decisions],
            "open_questions": summary.open_questions,
            "engine": summary.engine,
            "detail_level": summary.detail_level,
            "generated_at": summary.created_at.isoformat(),
            "cached": False,
            "block_count": summary.block_count,
        }


@mcp.tool(
    title="Backtrack",
    description=(
        "Restore agent context to a specific historical point. Accepts either a "
        "checkpoint_id (the UI path) or a raw block_id (advanced). Freezes a "
        "snapshot of every block up to and including the anchor; reclaim_context "
        "will re-inject it. With dry_run=True we compute everything but don't "
        "write the snapshot."
    ),
)
def backtrack(
    session_id: str,
    ctx: Context,
    checkpoint_id: str | None = None,
    block_id: str | None = None,
    name: str | None = None,
    reason: str | None = None,
    dry_run: bool = False,
) -> BacktrackResult:
    state = _state(ctx)
    if not checkpoint_id and not block_id:
        raise ValueError("provide either checkpoint_id or block_id")

    with log_call(
        "backtrack",
        session=session_id,
        checkpoint=checkpoint_id,
        block=block_id,
        dry_run=dry_run,
    ) as fields:
        if checkpoint_id is not None:
            cp = state.store.get_checkpoint(checkpoint_id)
            if cp is None:
                raise ValueError(f"unknown checkpoint_id: {checkpoint_id}")
            anchor = state.store.get_block(cp.block_id)
            chain_id = cp.chain_id
            anchor_label = cp.label
        else:
            assert block_id is not None
            anchor = state.store.get_block(block_id)
            if anchor is None:
                raise ValueError(f"unknown block_id: {block_id}")
            cp = None
            chain_id = anchor.chain_id
            anchor_label = f"block {anchor.sequence}"

        if anchor is None:
            raise ValueError("anchor block not found")

        prefix = state.store.list_chain_blocks(chain_id, up_to_sequence=anchor.sequence)
        sess = _ensure_session(state, session_id)
        role_counts = _block_role_counts(prefix)
        token_estimate = sum(_estimate_block_tokens(b.content) for b in prefix)
        chain = state.store.get_chain(chain_id)
        chain_title = chain.title if chain else None
        manifest = SnapshotManifest(
            block_count=len(prefix),
            token_estimate=token_estimate,
            role_counts=role_counts,
            chain_title=chain_title,
        )
        provenance = Provenance(
            created_by=session_id, reason=reason, dry_run=dry_run, name=name
        )
        snapshot_id: str | None = None
        if not dry_run:
            snapshot_id = str(uuid.uuid4())
            snap = Snapshot(
                snapshot_id=snapshot_id,
                session_id=session_id,
                chain_id=chain_id,
                checkpoint_id=cp.checkpoint_id if cp else None,
                anchor_block_id=anchor.block_id,
                anchor_sequence=anchor.sequence,
                lookback_limit=sess.lookback_limit,
                manifest=manifest,
                provenance=provenance,
                blocks=prefix,
            )
            state.store.insert_snapshot(snap)

        fields["blocks"] = len(prefix)
        fields["tokens"] = token_estimate
        fields["snapshot"] = snapshot_id

        restored_summary = (
            f"{'Would restore' if dry_run else 'Restored'} {len(prefix)} block(s) "
            f"up to '{anchor_label}' (sequence {anchor.sequence})."
        )
        return BacktrackResult(
            snapshot_id=snapshot_id,
            anchor_block_id=anchor.block_id,
            anchor_sequence=anchor.sequence,
            block_count=len(prefix),
            estimated_tokens=token_estimate,
            role_counts=role_counts,
            restored_summary=restored_summary,
            dry_run=dry_run,
        )


@mcp.tool(
    title="Reclaim Context",
    description=(
        "After backtracking, re-inject the selected historical context into the "
        "Active Agent so it can resume work. Honors `token_budget` by dropping "
        "the oldest blocks first; with `compress_if_over=True` the dropped "
        "blocks are summarized into a single synthetic prefix block so nothing "
        "is silently lost."
    ),
)
async def reclaim_context(
    session_id: str,
    snapshot_id: str,
    ctx: Context,
    token_budget: Annotated[int | None, Field(ge=1)] = None,
    roles: list[Role] | None = None,
    compress_if_over: bool = True,
) -> ReclaimResult:
    state = _state(ctx)
    with log_call(
        "reclaim_context",
        session=session_id,
        snapshot=snapshot_id,
        budget=token_budget,
    ) as fields:
        snap = state.store.get_snapshot(snapshot_id)
        if snap is None:
            raise ValueError(f"unknown snapshot_id: {snapshot_id}")
        if snap.session_id != session_id:
            raise ValueError(
                f"snapshot {snapshot_id} belongs to session {snap.session_id}, "
                f"not {session_id}"
            )

        blocks = list(snap.blocks)
        if roles:
            role_set = set(roles)
            blocks = [b for b in blocks if b.role in role_set]

        skipped: list[Block] = []
        if token_budget is not None:
            blocks, skipped = _trim_to_token_budget_pair(blocks, max_tokens=token_budget)

        compressed = False
        compression_summary: str | None = None
        if skipped and compress_if_over:
            # Summarize the dropped blocks into one synthetic block. We use
            # the existing Summarizer in a worker thread so we don't block.
            compress_summary = await asyncio.to_thread(
                state.summarizer.summarize,
                snap.chain_id,
                session_id,
                skipped,
                detail_level="skim",
            )
            compression_summary = compress_summary.plain_text
            compressed = True

        tokens_injected = sum(_estimate_block_tokens(b.content) for b in blocks)
        if compression_summary:
            tokens_injected += _estimate_block_tokens(compression_summary)

        sess = _ensure_session(state, session_id)
        sess.active_chain_id = snap.chain_id
        state.store.upsert_session(sess)

        fields["blocks_injected"] = len(blocks)
        fields["blocks_skipped"] = len(skipped)
        fields["compressed"] = compressed

        return ReclaimResult(
            success=True,
            snapshot_id=snapshot_id,
            tokens_injected=tokens_injected,
            blocks_injected=len(blocks),
            blocks_skipped=len(skipped),
            compressed=compressed,
            compression_summary=compression_summary,
            active_chain_id=snap.chain_id,
        )


# ----------------------------------------------------------------- resources


@mcp.resource(
    "retrieval://history/{session_id}",
    title="Plain-language History",
    description=(
        "Latest Claude-generated plain-language history for this session, with "
        "themes, decisions, open questions, and clickable checkpoints."
    ),
    mime_type="application/json",
)
def history_resource(session_id: str) -> str:
    state = _state_for_resource()
    summary = state.store.latest_summary_for_session(session_id)
    if summary is None:
        return json.dumps(
            {"session_id": session_id, "summary": None}, ensure_ascii=False
        )
    payload = summary.model_dump(mode="json")
    payload["session_id"] = session_id
    return json.dumps(payload, ensure_ascii=False, indent=2)


@mcp.resource(
    "retrieval://snapshot/{point_id}",
    title="Frozen Agent Snapshot",
    description=(
        "Frozen agent state at a specific chain point. Used by `backtrack` and "
        "`reclaim_context`. The body is JSON: manifest, provenance, blocks, "
        "lookback config, and timing."
    ),
    mime_type="application/json",
)
def snapshot_resource(point_id: str) -> str:
    state = _state_for_resource()
    snap = state.store.get_snapshot(point_id)
    if snap is None:
        return json.dumps(
            {"snapshot_id": point_id, "error": "not_found"}, ensure_ascii=False
        )
    return json.dumps(snap.model_dump(mode="json"), ensure_ascii=False, indent=2)


# ------------------------------------------------------------------ prompts


@mcp.prompt(
    title="Recap History",
    description="Generate a plain-language recap of the session and narrate it to the user.",
)
def recap_history(session_id: str, detail_level: DetailLevel = "standard") -> list[Message]:
    return [
        UserMessage(
            "Recap what's happened in this session for me.\n\n"
            "Steps:\n"
            f"1. Call `summarize_history(chain_id=<active chain for session '{session_id}'>, "
            f"session_id='{session_id}', detail_level='{detail_level}')`.\n"
            "2. Read the result and tell me the story in 4-6 short paragraphs.\n"
            "3. End with a numbered list of the checkpoints I can click to backtrack, "
            "each with its label and one-line reason.\n"
            "Use the themes and decisions from the result; don't re-summarize raw blocks."
        ),
    ]


@mcp.prompt(
    title="Find Related Context",
    description="Search the chain for context related to a topic and explain the matches.",
)
def find_related_context(chain_id: str, topic: str) -> list[Message]:
    return [
        UserMessage(
            f"I want context about \"{topic}\" from chain {chain_id}.\n\n"
            "Steps:\n"
            f"1. Call `query_chain(chain_id='{chain_id}', prompt='{topic}', "
            "mode='hybrid', explain=True, diversify=True, top_k=5)`.\n"
            "2. For each match, summarize what was happening in 1-2 sentences "
            "and cite the `sequence` and `context_hint`.\n"
            "3. If two matches are adjacent, call `scan_adjacent(block_id=..., window=1)` "
            "to merge them before summarizing.\n"
            "4. End by suggesting whether I should backtrack to one of them and which."
        ),
    ]


@mcp.prompt(
    title="Restore to Decision",
    description="Find the moment a key decision was made and backtrack the agent there.",
)
def restore_to_decision(session_id: str, decision_query: str) -> list[Message]:
    return [
        UserMessage(
            f"I want to backtrack to when we decided about \"{decision_query}\".\n\n"
            "Steps:\n"
            f"1. Call `summarize_history(session_id='{session_id}', detail_level='deep')`.\n"
            "2. From the result, find the checkpoint whose label/reason best matches "
            f"\"{decision_query}\".\n"
            "3. Call `backtrack(session_id=..., checkpoint_id=..., dry_run=True)` and "
            "show me the manifest. Wait for my confirmation.\n"
            "4. On confirmation, call `backtrack` without dry_run and then "
            "`reclaim_context(session_id=..., snapshot_id=..., compress_if_over=True)`.\n"
            "5. Confirm the session is now anchored to that decision."
        ),
    ]


# ----------------------------------------------------------------- internals


def _trim_to_token_budget(
    blocks: list[Block], *, anchor_block_id: str, max_tokens: int
) -> tuple[list[Block], int]:
    """Drop blocks until total <= max_tokens, but never drop the anchor.

    Strategy: drop from the front (oldest) first; if the front is the
    anchor, drop from the back instead. If we run out of non-anchor
    blocks and the anchor alone still exceeds the budget, we stop —
    callers preferred preserving the anchor over honoring the budget.
    """
    kept = list(blocks)
    total = sum(_estimate_block_tokens(b.content) for b in kept)
    dropped = 0
    while kept and total > max_tokens:
        if len(kept) == 1:
            break  # only the anchor left; keep it even if over budget
        # Drop from the front, unless that would remove the anchor; in
        # which case drop from the back. If both ends are the anchor
        # (impossible given len>1) we'd bail.
        if kept[0].block_id == anchor_block_id:
            gone = kept.pop()
        else:
            gone = kept.pop(0)
        total -= _estimate_block_tokens(gone.content)
        dropped += 1
    return kept, dropped


def _trim_to_token_budget_pair(
    blocks: list[Block], *, max_tokens: int
) -> tuple[list[Block], list[Block]]:
    """Drop oldest first, returning (kept, dropped)."""
    kept = list(blocks)
    total = sum(_estimate_block_tokens(b.content) for b in kept)
    dropped: list[Block] = []
    while kept and total > max_tokens:
        gone = kept.pop(0)
        dropped.append(gone)
        total -= _estimate_block_tokens(gone.content)
    return kept, dropped
