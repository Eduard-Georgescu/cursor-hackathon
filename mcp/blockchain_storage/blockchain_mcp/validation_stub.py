"""Stub 'Greptile' validation — lightweight checks only (no external API)."""

from __future__ import annotations

from blockchain_mcp.models import MAX_BLOCK_TOKEN_CAPACITY


def run_stub_validation(content: str) -> tuple[bool, list[str], str]:
    issues: list[str] = []
    cleaned = content.replace("\r\n", "\n").strip("\n")

    if len(content.encode("utf-8")) > MAX_BLOCK_TOKEN_CAPACITY * 8:
        issues.append("warn:block exceeds heuristic byte ceiling")

    if not cleaned.strip():
        issues.append("error:block content is empty after normalization")

    placeholder_patterns = ("FIXME:GREPTILE", "<<REPLACE_ME>>")
    for p in placeholder_patterns:
        if p in content:
            issues.append(f"warn:forbidden_placeholder_pattern:{p}")

    valid = not any(i.startswith("error:") for i in issues)
    return valid, issues, cleaned
