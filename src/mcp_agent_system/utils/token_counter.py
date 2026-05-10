"""Token counting helpers.

CLōD routes to many backend models. We keep a known-limits table for the
common ones and fall back to a configurable default.
"""

from __future__ import annotations

MODEL_CONTEXT_LIMITS: dict[str, int] = {
    # Anthropic via CLōD
    "anthropic/claude-opus-4": 200_000,
    "anthropic/claude-opus-4-20250514": 200_000,
    "anthropic/claude-sonnet-4": 200_000,
    "anthropic/claude-sonnet-4-20250514": 200_000,
    "anthropic/claude-haiku-4-5": 200_000,
    # OpenAI via CLōD
    "openai/gpt-4o": 128_000,
    "openai/gpt-4o-mini": 128_000,
    # Meta via CLōD
    "meta-llama/llama-3.1-8b-instruct": 128_000,
    "meta-llama/llama-3.1-70b-instruct": 128_000,
}


def get_context_limit(model_id: str, default: int = 200_000) -> int:
    return MODEL_CONTEXT_LIMITS.get(model_id, default)


def compute_context_pct(tokens_used: int, limit: int) -> float:
    if limit <= 0:
        return 0.0
    return min(tokens_used / limit, 1.0)


def estimate_tokens_from_text(text: str) -> int:
    """Rough estimate (~4 chars per token). Used only for summary-injection sizing."""
    return max(1, (len(text) + 3) // 4)
