"""Hybrid semantic + lexical search over chain blocks.

The default `mode="hybrid"` combines two ranked lists:

    1. Semantic leg: cosine similarity between the prompt embedding and
       per-block embeddings. Uses `sentence-transformers` if installed,
       otherwise a deterministic hashed-token TF-IDF vector so the
       server runs zero-deps.

    2. Lexical leg: SQLite FTS5 BM25 over `blocks.content`.

The two rankings are fused with Reciprocal Rank Fusion (RRF), then a
recency boost (`exp(-age_days/tau)`) nudges fresh blocks. MMR
re-ranking diversifies results when `diversify=True` so a user doesn't
get five neighboring blocks back when one would do.

Modes `semantic` and `lexical` skip the other leg entirely.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Iterable

import numpy as np

from .models import (
    Block,
    Match,
    Role,
    ScoreBreakdown,
    SearchMode,
    TimeRange,
)
from .store import Store


_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
_FALLBACK_DIM = 256
_FALLBACK_MODEL = f"hashed-tfidf-{_FALLBACK_DIM}"
_RRF_K = 60  # RRF damping constant; 60 is the value in the original RRF paper.
_RECENCY_TAU_DAYS = 30.0  # weeks-old blocks still count, but freshness wins.


# --------------------------------------------------------------- tokenizer


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _WORD_RE.findall(text)]


def _hash_idx(token: str, dim: int) -> int:
    h = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(h, "big") % dim


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n == 0.0:
        return v
    return v / n


# --------------------------------------------------------------- embedder


class Embedder:
    """Embedding interface with lazy model loading + graceful fallback."""

    def __init__(self, prefer_model: str | None = None):
        self._st_model = None
        self._st_name: str | None = None
        if prefer_model is None:
            prefer_model = "sentence-transformers/all-MiniLM-L6-v2"
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._st_model = SentenceTransformer(prefer_model)
            self._st_name = prefer_model
        except Exception:
            # No torch, no model on disk, no network — fall back silently.
            self._st_model = None
            self._st_name = None

    @property
    def model_name(self) -> str:
        return self._st_name or _FALLBACK_MODEL

    @property
    def dim(self) -> int:
        if self._st_model is not None:
            return int(self._st_model.get_sentence_embedding_dimension())
        return _FALLBACK_DIM

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if self._st_model is not None:
            vecs = self._st_model.encode(
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return vecs.astype(np.float32, copy=False)
        return np.stack([self._hashed_tfidf(t) for t in texts], axis=0)

    # -------------------------------------------------------------- fallback

    def _hashed_tfidf(self, text: str) -> np.ndarray:
        toks = _tokens(text)
        if not toks:
            return np.zeros(_FALLBACK_DIM, dtype=np.float32)
        counts = Counter(toks)
        vec = np.zeros(_FALLBACK_DIM, dtype=np.float32)
        for tok, c in counts.items():
            idx = _hash_idx(tok, _FALLBACK_DIM)
            sign = 1.0 if (hash(tok) & 1) == 0 else -1.0
            vec[idx] += sign * (1.0 + math.log(c))
        return _normalize(vec)


# ----------------------------------------------------------------- index


class SemanticIndex:
    """Hybrid retrieval over chain blocks."""

    def __init__(self, store: Store, embedder: Embedder | None = None):
        self.store = store
        self.embedder = embedder or Embedder()

    # ------------------------------------------------------- embedding cache

    def _cached_embedding(self, block: Block) -> np.ndarray:
        cached = self.store.get_embedding(block.block_id)
        if cached is not None:
            model, vec = cached
            if model == self.embedder.model_name and vec.shape[0] == self.embedder.dim:
                return vec
        vec = self.embedder.encode([block.content])[0]
        self.store.put_embedding(block.block_id, self.embedder.model_name, vec)
        return vec

    def ensure_embedded(self, blocks: Iterable[Block]) -> None:
        """Batch-encode any blocks that are missing or out-of-model embeddings."""
        missing: list[Block] = []
        for b in blocks:
            cached = self.store.get_embedding(b.block_id)
            if (
                cached is None
                or cached[0] != self.embedder.model_name
                or cached[1].shape[0] != self.embedder.dim
            ):
                missing.append(b)
        if not missing:
            return
        vecs = self.embedder.encode([b.content for b in missing])
        for b, v in zip(missing, vecs):
            self.store.put_embedding(b.block_id, self.embedder.model_name, v)

    # ----------------------------------------------------------------- search

    def search(
        self,
        chain_id: str,
        prompt: str,
        *,
        lookback_limit: int,
        top_k: int = 10,
        mode: SearchMode = "hybrid",
        diversify: bool = False,
        roles: list[Role] | None = None,
        time_range: TimeRange | None = None,
        min_relevance: float = 0.0,
        explain: bool = False,
    ) -> tuple[list[Match], int]:
        """Return (matches, total_candidates_before_filtering)."""
        prompt = prompt.strip()
        if not prompt:
            return ([], 0)

        candidates = self.store.list_chain_blocks(chain_id, limit=lookback_limit)
        # Apply role / time filters up front so cheaper legs don't waste work.
        filtered = _apply_filters(candidates, roles=roles, time_range=time_range)
        if not filtered:
            return ([], len(candidates))

        block_by_id = {b.block_id: b for b in filtered}
        valid_ids = set(block_by_id.keys())

        # ---- semantic leg
        semantic_scores: dict[str, float] = {}
        if mode in ("semantic", "hybrid"):
            self.ensure_embedded(filtered)
            q = self.embedder.encode([prompt])[0]
            vectors = np.stack(
                [self._cached_embedding(b) for b in filtered], axis=0
            )
            sims = vectors @ q  # both unit-norm
            sims_clamped = (sims + 1.0) / 2.0  # map cosine [-1,1] -> [0,1]
            for b, s in zip(filtered, sims_clamped):
                semantic_scores[b.block_id] = float(s)

        # ---- lexical leg
        lexical_scores: dict[str, float] = {}
        if mode in ("lexical", "hybrid"):
            # bm25() returns lower=better; convert to higher=better.
            raw = self.store.lexical_search(chain_id, prompt, limit=max(top_k * 4, 40))
            raw = [(b, s) for (b, s) in raw if b.block_id in valid_ids]
            if raw:
                worst = max(s for _, s in raw)
                # Add a small epsilon so the best result still beats zero.
                for b, s in raw:
                    lexical_scores[b.block_id] = float(worst - s + 1e-3)

        # ---- score fusion
        sem_rank = _rank_dict(semantic_scores)
        lex_rank = _rank_dict(lexical_scores)
        rrf: dict[str, float] = {}
        for bid in valid_ids:
            score = 0.0
            if bid in sem_rank:
                score += 1.0 / (_RRF_K + sem_rank[bid])
            if bid in lex_rank:
                score += 1.0 / (_RRF_K + lex_rank[bid])
            if score > 0:
                rrf[bid] = score

        # ---- recency boost
        now = datetime.now(timezone.utc)
        recency_scores: dict[str, float] = {}
        for bid, b in block_by_id.items():
            age_days = max(
                0.0,
                (now - _ensure_aware(b.created_at)).total_seconds() / 86400.0,
            )
            recency_scores[bid] = math.exp(-age_days / _RECENCY_TAU_DAYS)

        # ---- finalize fused score
        if not rrf:
            return ([], len(candidates))
        max_rrf = max(rrf.values())
        max_rec = max(recency_scores.values()) if recency_scores else 1.0
        fused: dict[str, float] = {}
        for bid in rrf:
            base = rrf[bid] / max_rrf if max_rrf else 0.0
            rec = (recency_scores.get(bid, 0.0) / max_rec) if max_rec else 0.0
            # 85% fused retrieval, 15% recency boost. Clamps to [0,1].
            fused[bid] = max(0.0, min(1.0, 0.85 * base + 0.15 * rec))

        # ---- MMR diversification (optional)
        ordered_ids: list[str]
        if diversify and mode in ("semantic", "hybrid") and semantic_scores:
            ordered_ids = _mmr(
                fused=fused,
                vectors={bid: self._cached_embedding(block_by_id[bid]) for bid in fused},
                k=top_k,
                lambda_=0.7,
            )
        else:
            ordered_ids = sorted(fused, key=lambda b: -fused[b])[:top_k]

        # ---- min_relevance filter
        ordered_ids = [bid for bid in ordered_ids if fused[bid] >= min_relevance]

        # ---- build Match objects
        results: list[Match] = []
        for bid in ordered_ids:
            b = block_by_id[bid]
            preview = b.content.strip().replace("\n", " ")
            if len(preview) > 220:
                preview = preview[:217] + "..."
            breakdown: ScoreBreakdown | None = None
            hint: str | None = None
            if explain:
                breakdown = ScoreBreakdown(
                    semantic=round(semantic_scores.get(bid, 0.0), 4),
                    lexical=round(lexical_scores.get(bid, 0.0), 4),
                    recency=round(recency_scores.get(bid, 0.0), 4),
                    fused=round(fused[bid], 4),
                )
                hint = _build_context_hint(b, prompt)
            results.append(
                Match(
                    block_id=b.block_id,
                    chain_id=b.chain_id,
                    sequence=b.sequence,
                    role=b.role,
                    created_at=b.created_at,
                    relevance=round(fused[bid], 4),
                    preview=preview,
                    score_breakdown=breakdown,
                    context_hint=hint,
                )
            )
        return (results, len(candidates))


# --------------------------------------------------------------- helpers


def _apply_filters(
    blocks: list[Block],
    *,
    roles: list[Role] | None,
    time_range: TimeRange | None,
) -> list[Block]:
    out = blocks
    if roles:
        role_set = set(roles)
        out = [b for b in out if b.role in role_set]
    if time_range is not None:
        if time_range.from_ is not None:
            lower = _ensure_aware(time_range.from_)
            out = [b for b in out if _ensure_aware(b.created_at) >= lower]
        if time_range.to is not None:
            upper = _ensure_aware(time_range.to)
            out = [b for b in out if _ensure_aware(b.created_at) <= upper]
    return out


def _rank_dict(scores: dict[str, float]) -> dict[str, int]:
    """Map id -> 1-based rank (best=1). Stable on ties via insertion order."""
    if not scores:
        return {}
    ordered = sorted(scores.items(), key=lambda kv: -kv[1])
    return {bid: i + 1 for i, (bid, _) in enumerate(ordered)}


def _ensure_aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def _mmr(
    *,
    fused: dict[str, float],
    vectors: dict[str, np.ndarray],
    k: int,
    lambda_: float,
) -> list[str]:
    """Greedy MMR. Maximizes lambda*relevance - (1-lambda)*max_sim_to_chosen."""
    remaining = list(fused.keys())
    chosen: list[str] = []
    while remaining and len(chosen) < k:
        best_id: str | None = None
        best_score = -math.inf
        for bid in remaining:
            rel = fused[bid]
            if not chosen:
                penalty = 0.0
            else:
                v = vectors[bid]
                penalty = max(
                    float(np.dot(v, vectors[other])) for other in chosen
                )
            score = lambda_ * rel - (1.0 - lambda_) * penalty
            if score > best_score:
                best_score = score
                best_id = bid
        if best_id is None:
            break
        chosen.append(best_id)
        remaining.remove(best_id)
    return chosen


def _build_context_hint(block: Block, prompt: str) -> str:
    """One-line explanation of why this block matched."""
    p_terms = {t for t in _tokens(prompt) if len(t) > 2}
    b_terms = _tokens(block.content)
    hits = [t for t in dict.fromkeys(b_terms) if t in p_terms]
    lead = block.content.strip().splitlines()[0] if block.content.strip() else ""
    if len(lead) > 120:
        lead = lead[:117] + "..."
    if hits:
        joined = ", ".join(hits[:5])
        return f"matches on: {joined} — {lead}"
    return lead or "(no preview)"
