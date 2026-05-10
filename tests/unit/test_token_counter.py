from __future__ import annotations

from mcp_agent_system.utils.token_counter import (
    compute_context_pct,
    estimate_tokens_from_text,
    get_context_limit,
)


def test_known_model_limit() -> None:
    assert get_context_limit("anthropic/claude-opus-4") == 200_000


def test_unknown_model_falls_back_to_default() -> None:
    assert get_context_limit("totally-unknown-model") == 200_000


def test_compute_context_pct_basic() -> None:
    assert compute_context_pct(160_000, 200_000) == 0.8


def test_compute_context_pct_caps_at_one() -> None:
    assert compute_context_pct(250_000, 200_000) == 1.0


def test_compute_context_pct_zero_limit() -> None:
    assert compute_context_pct(100, 0) == 0.0


def test_estimate_tokens_minimum_one() -> None:
    assert estimate_tokens_from_text("") == 1
    assert estimate_tokens_from_text("abcd") == 1
    assert estimate_tokens_from_text("a" * 8) == 2
