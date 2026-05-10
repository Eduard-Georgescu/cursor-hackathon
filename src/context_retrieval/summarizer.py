"""Plain-language history generation.

Three detail levels, each a different prompt chain:

    skim      — single Claude call, 2-3 paragraphs, 3-5 checkpoints
    standard  — single Claude call, 3-6 paragraphs, themes, top decisions,
                4-10 checkpoints (default)
    deep      — two-pass:
                pass 1 extracts themes/decisions/anchor_candidates
                pass 2 narrates around themes and emits 6-12 reasoned
                checkpoints citing block_ids

Without `ANTHROPIC_API_KEY` (or when a Claude call fails), we fall back
to a deterministic extractive summarizer so the server keeps working.

Robustness:
    - Validate Claude's JSON shape; drop any field that doesn't conform.
    - Hallucination guard: every checkpoint `block_id` must be in the
      input. Hallucinated ids are dropped and back-filled from the
      extractive picks so the user still has clickable history.
    - Per-block truncation to `MAX_CHARS_PER_BLOCK` and global cap of
      `MAX_BLOCKS_TO_SEND` to keep Claude calls bounded.
    - Summaries are keyed by (chain_id, detail_level, block_count) so a
      re-call with the same inputs returns cached output unless
      `force_refresh=True`.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any

from .config import Settings
from .logging_utils import get_logger
from .models import Checkpoint, Decision, DetailLevel, Summary
from .prompts import (
    deep_pass1_prompt,
    deep_pass2_prompt,
    skim_prompt,
    standard_prompt,
)


_log = get_logger("summarizer")

VALID_CATEGORIES = {"decision", "question", "result", "transition", "blocker"}


# ----------------------------------------------------------------- helpers


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\n", "", text)
        text = re.sub(r"\n```$", "", text)
    return text.strip()


def _truncate_blocks(
    blocks: list[Any],
    *,
    max_blocks: int,
    max_chars: int,
) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    # Keep the most recent N blocks so context near the user's "now" wins.
    for b in blocks[-max_blocks:]:
        content = b.content
        if len(content) > max_chars:
            content = content[:max_chars] + " ... [truncated]"
        payload.append(
            {
                "block_id": b.block_id,
                "sequence": b.sequence,
                "role": b.role,
                "content": content,
            }
        )
    return payload


_SENT_RE = re.compile(r"[^.!?\n]+[.!?]?")


def _first_sentence(text: str, max_len: int = 200) -> str:
    text = text.strip().replace("\n", " ")
    if not text:
        return ""
    m = _SENT_RE.search(text)
    sentence = (m.group(0) if m else text).strip()
    if len(sentence) > max_len:
        sentence = sentence[: max_len - 1].rstrip() + "\u2026"
    return sentence


def _estimate_tokens(text: str) -> int:
    # Whitespace tokens are a rough but stable proxy. Good enough for the UI.
    return max(1, len(text.split()))


# ----------------------------------------------------------------- summarizer


class Summarizer:
    """Wraps Claude (when available) with an extractive fallback."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings.from_env()
        self._client = None
        if self.settings.anthropic_api_key:
            try:
                import anthropic  # type: ignore

                self._client = anthropic.Anthropic(
                    api_key=self.settings.anthropic_api_key
                )
            except Exception as exc:
                _log.warning(
                    f"event=anthropic.unavailable error={type(exc).__name__}"
                )
                self._client = None

    @property
    def using_claude(self) -> bool:
        return self._client is not None

    @property
    def engine_id(self) -> str:
        return (
            f"claude-{self.settings.anthropic_model}"
            if self._client is not None
            else "extractive"
        )

    # ------------------------------------------------------------------ main

    def summarize(
        self,
        chain_id: str,
        session_id: str,
        blocks: list[Any],
        *,
        detail_level: DetailLevel = "standard",
        model_override: str | None = None,
    ) -> Summary:
        summary_id = str(uuid.uuid4())
        if not blocks:
            return Summary(
                summary_id=summary_id,
                chain_id=chain_id,
                session_id=session_id,
                plain_text="This chain has no activity yet.",
                detail_level=detail_level,
                engine=self.engine_id,
                token_count=0,
                block_count=0,
            )

        # Try Claude first; on any failure (network, bad JSON, etc.) fall
        # back to the extractive summarizer. We log the reason so it's
        # visible to operators.
        plain_text = ""
        themes: list[str] = []
        decisions: list[Decision] = []
        open_questions: list[str] = []
        cp_dicts: list[dict[str, Any]] = []
        engine = self.engine_id

        if self._client is not None:
            try:
                plain_text, themes, decisions, open_questions, cp_dicts = (
                    self._summarize_with_claude(
                        blocks,
                        detail_level=detail_level,
                        model_override=model_override,
                    )
                )
            except Exception as exc:
                _log.warning(
                    f"event=claude.summarize.failed error={type(exc).__name__} "
                    f'message="{exc}" detail_level={detail_level}'
                )
                engine = "extractive"
                plain_text, themes, decisions, open_questions, cp_dicts = (
                    self._summarize_extractive(blocks, detail_level=detail_level)
                )
        else:
            plain_text, themes, decisions, open_questions, cp_dicts = (
                self._summarize_extractive(blocks, detail_level=detail_level)
            )

        # Build Checkpoint models, assigning ids and positions.
        checkpoints: list[Checkpoint] = []
        for i, cp in enumerate(cp_dicts):
            checkpoints.append(
                Checkpoint(
                    checkpoint_id=str(uuid.uuid4()),
                    summary_id=summary_id,
                    block_id=cp["block_id"],
                    chain_id=chain_id,
                    position=i,
                    label=cp["label"],
                    description=cp.get("description", ""),
                    reason=cp.get("reason"),
                    confidence=cp.get("confidence"),
                    category=cp.get("category"),
                )
            )

        token_count = _estimate_tokens(plain_text) + sum(
            _estimate_tokens(b.content) for b in blocks
        )
        return Summary(
            summary_id=summary_id,
            chain_id=chain_id,
            session_id=session_id,
            plain_text=plain_text,
            detail_level=detail_level,
            engine=engine,
            token_count=token_count,
            block_count=len(blocks),
            themes=themes,
            decisions=decisions,
            open_questions=open_questions,
            checkpoints=checkpoints,
        )

    # --------------------------------------------------------------- Claude

    def _call_claude(
        self,
        *,
        system_prompt: str,
        user_payload: str,
        model_override: str | None,
        max_tokens: int = 2048,
    ) -> str:
        assert self._client is not None
        model = model_override or self.settings.anthropic_model
        start = time.perf_counter()
        resp = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_payload}],
        )
        elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
        _log.info(
            f"event=claude.call model={model} max_tokens={max_tokens} "
            f"duration_ms={elapsed_ms}"
        )
        return "".join(
            block.text
            for block in resp.content
            if getattr(block, "type", None) == "text"
        )

    def _summarize_with_claude(
        self,
        blocks: list[Any],
        *,
        detail_level: DetailLevel,
        model_override: str | None,
    ) -> tuple[str, list[str], list[Decision], list[str], list[dict[str, Any]]]:
        payload = _truncate_blocks(
            blocks,
            max_blocks=self.settings.max_blocks_to_summarize,
            max_chars=self.settings.max_chars_per_block,
        )
        valid_ids = {b["block_id"] for b in payload}

        if detail_level == "skim":
            text = self._call_claude(
                system_prompt=skim_prompt(),
                user_payload=(
                    "Here is the chain in JSON. Produce the history JSON now.\n\n"
                    + json.dumps(payload, ensure_ascii=False)
                ),
                model_override=model_override,
                max_tokens=1500,
            )
            data = self._parse_json(text)
            return (
                str(data.get("summary", "")).strip(),
                _coerce_str_list(data.get("themes")),
                [],
                [],
                _validate_checkpoints(data.get("checkpoints"), valid_ids),
            )

        if detail_level == "standard":
            text = self._call_claude(
                system_prompt=standard_prompt(),
                user_payload=(
                    "Here is the chain in JSON. Produce the history JSON now.\n\n"
                    + json.dumps(payload, ensure_ascii=False)
                ),
                model_override=model_override,
                max_tokens=2500,
            )
            data = self._parse_json(text)
            return (
                str(data.get("summary", "")).strip(),
                _coerce_str_list(data.get("themes")),
                _validate_decisions(data.get("decisions"), valid_ids),
                _coerce_str_list(data.get("open_questions")),
                _validate_checkpoints(data.get("checkpoints"), valid_ids),
            )

        # detail_level == "deep" — two-pass
        pass1_text = self._call_claude(
            system_prompt=deep_pass1_prompt(),
            user_payload=(
                "Here is the chain in JSON.\n\n"
                + json.dumps(payload, ensure_ascii=False)
            ),
            model_override=model_override,
            max_tokens=2500,
        )
        pass1 = self._parse_json(pass1_text)
        themes = _coerce_str_list(pass1.get("themes"))
        decisions = _validate_decisions(pass1.get("decisions"), valid_ids)
        open_q = _coerce_str_list(pass1.get("open_questions"))

        pass2_text = self._call_claude(
            system_prompt=deep_pass2_prompt(_strip_code_fence(pass1_text)),
            user_payload=(
                "Here is the chain in JSON. Produce the pass-2 JSON now.\n\n"
                + json.dumps(payload, ensure_ascii=False)
            ),
            model_override=model_override,
            max_tokens=3000,
        )
        pass2 = self._parse_json(pass2_text)
        narrative = str(pass2.get("summary", "")).strip()
        cps = _validate_checkpoints(pass2.get("checkpoints"), valid_ids)
        return (narrative, themes, decisions, open_q, cps)

    # ---------------------------------------------------------- parse + guard

    def _parse_json(self, text: str) -> dict[str, Any]:
        cleaned = _strip_code_fence(text)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Claude returned non-JSON: {exc.msg}") from exc
        if not isinstance(data, dict):
            raise ValueError("Claude returned non-object JSON")
        return data

    # --------------------------------------------------------- extractive

    def _summarize_extractive(
        self,
        blocks: list[Any],
        *,
        detail_level: DetailLevel,
    ) -> tuple[str, list[str], list[Decision], list[str], list[dict[str, Any]]]:
        # Group into "turns": consecutive blocks with the same role.
        turns: list[list[Any]] = []
        for b in blocks:
            if turns and turns[-1][-1].role == b.role:
                turns[-1].append(b)
            else:
                turns.append([b])

        first_user = next((b for b in blocks if b.role == "user"), blocks[0])
        opener = _first_sentence(first_user.content)
        paragraphs = [
            f"This session opened with you asking: \u201c{opener}\u201d.",
        ]

        notable = [t for t in turns if t[0].role in ("user", "agent")]
        mid_count = {"skim": 1, "standard": 4, "deep": 6}.get(detail_level, 4)
        mid = notable[1:-1][:mid_count] if len(notable) > 2 else []
        for turn in mid:
            head = turn[0]
            actor = "You" if head.role == "user" else "The agent"
            paragraphs.append(f"{actor} then: {_first_sentence(head.content)}")

        last = blocks[-1]
        tail_actor = "you" if last.role == "user" else "the agent"
        paragraphs.append(
            f"Most recently, {tail_actor}: {_first_sentence(last.content)}"
        )

        summary_text = "\n\n".join(paragraphs)

        # Lightweight theme extraction: most common content words.
        theme_cap = {"skim": 3, "standard": 6, "deep": 8}.get(detail_level, 6)
        themes = _extract_themes(blocks, cap=theme_cap)

        # Decisions: any agent block whose first sentence starts with a verb
        # of decision is a candidate; otherwise empty.
        decisions: list[Decision] = []
        for b in blocks:
            if b.role != "agent":
                continue
            sent = _first_sentence(b.content).lower()
            if any(sent.startswith(v) for v in ("set ", "use ", "switch ", "move ", "add ", "install ", "drop ")):
                decisions.append(Decision(summary=_first_sentence(b.content), block_id=b.block_id))
            if len(decisions) >= 3:
                break

        # Open questions: trailing user block ending in `?`.
        open_questions: list[str] = []
        if last.role == "user" and last.content.strip().endswith("?"):
            open_questions.append(_first_sentence(last.content))

        # Checkpoints: head of each "turn" up to cap, plus the final block.
        cp_cap = {"skim": 5, "standard": 8, "deep": 12}.get(detail_level, 8)
        candidates = [turn[0] for turn in turns]
        if blocks[-1] is not candidates[-1]:
            candidates.append(blocks[-1])
        if len(candidates) > cp_cap:
            step = len(candidates) / cp_cap
            candidates = [candidates[int(i * step)] for i in range(cp_cap)]

        cp_dicts: list[dict[str, Any]] = []
        seen: set[str] = set()
        for b in candidates:
            if b.block_id in seen:
                continue
            seen.add(b.block_id)
            label = _first_sentence(b.content)
            if len(label) > 60:
                label = label[:57] + "..."
            actor = {
                "user": "You",
                "agent": "Agent",
                "tool": "Tool",
                "system": "System",
            }.get(b.role, b.role.capitalize())
            cp_dicts.append(
                {
                    "block_id": b.block_id,
                    "label": label or f"{actor} turn",
                    "description": f"{actor}: {_first_sentence(b.content) or '(empty block)'}",
                    "reason": "turn boundary in the chain",
                    "confidence": 0.5,
                    "category": _guess_category(b),
                }
            )
        return (summary_text, themes, decisions, open_questions, cp_dicts)


# ------------------------------------------------------------- validators


def _coerce_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for v in value:
        if isinstance(v, str) and v.strip():
            out.append(v.strip())
    return out


def _validate_decisions(value: Any, valid_ids: set[str]) -> list[Decision]:
    if not isinstance(value, list):
        return []
    out: list[Decision] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        summary = str(item.get("summary", "")).strip()
        if not summary:
            continue
        block_id = item.get("block_id")
        if block_id is not None and block_id not in valid_ids:
            block_id = None  # drop hallucinated id, keep the decision
        out.append(Decision(summary=summary, block_id=block_id))
    return out


def _validate_checkpoints(
    value: Any, valid_ids: set[str]
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        block_id = item.get("block_id")
        if not isinstance(block_id, str) or block_id not in valid_ids:
            continue  # hallucination guard
        label = str(item.get("label", "Checkpoint"))[:60].strip() or "Checkpoint"
        description = str(item.get("description", "")).strip()
        reason = item.get("reason")
        if isinstance(reason, str):
            reason = reason.strip() or None
        else:
            reason = None
        confidence: float | None
        try:
            confidence = float(item.get("confidence")) if item.get("confidence") is not None else None
            if confidence is not None:
                confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            confidence = None
        category = item.get("category")
        if not isinstance(category, str) or category not in VALID_CATEGORIES:
            category = None
        out.append(
            {
                "block_id": block_id,
                "label": label,
                "description": description,
                "reason": reason,
                "confidence": confidence,
                "category": category,
            }
        )
    return out


# ----------------------------------------------------------- helpers


_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "you", "your", "from", "into",
    "have", "are", "was", "were", "but", "not", "any", "all", "can", "out",
    "use", "via", "per", "let", "got", "now", "one", "two", "ill", "its", "ive",
    "they", "them", "their", "there", "then", "than", "what", "when", "which",
    "who", "how", "why", "where", "would", "could", "should", "about", "after",
    "before", "back", "here", "just", "like", "make", "more", "most", "much",
    "some", "such", "want", "well", "will", "yes", "your", "yourself",
    "im", "ive", "id", "ll", "didnt", "doesnt", "isnt",
}


def _extract_themes(blocks: list[Any], cap: int) -> list[str]:
    from collections import Counter

    counts: Counter[str] = Counter()
    for b in blocks:
        for token in _word_tokens(b.content):
            if token in _STOPWORDS or len(token) < 4:
                continue
            counts[token] += 1
    return [w for w, _ in counts.most_common(cap)]


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


def _word_tokens(text: str) -> list[str]:
    return [t.lower() for t in _WORD_RE.findall(text)]


def _guess_category(block: Any) -> str:
    role = getattr(block, "role", "")
    text = (getattr(block, "content", "") or "").lower()
    if role == "user" and text.strip().endswith("?"):
        return "question"
    if role == "tool":
        return "result"
    if any(kw in text for kw in ("blocked", "stuck", "error", "fail")):
        return "blocker"
    if any(text.startswith(v) for v in ("set ", "use ", "switch ", "move ")):
        return "decision"
    return "transition"
