"""Approximate token counts for PRMD2 blocks (v1 heuristic)."""


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 characters per token."""
    if not text:
        return 0
    return max(1, len(text) // 4) if len(text) >= 4 else 1
