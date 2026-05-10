"""Claude system prompts used by the summarizer.

Kept separate from `summarizer.py` so we can A/B test prompts without
churning code, and so the wording stays diff-able across runs.

Each prompt is a strict-JSON contract — we validate Claude's output
against it and drop any checkpoints whose `block_id` isn't in the input.
"""

from __future__ import annotations


SHARED_PREAMBLE = """You are the History view of an agent IDE.

You will receive a chain of blocks (each one prompt / response / tool
call inside an agent session). Your job is to convert this raw chain
into a plain-language recap that the user will read in a "History" tab,
plus a small set of clickable checkpoints they can use to backtrack the
agent to earlier points.

Return ONLY valid JSON. Do not wrap the JSON in markdown fences. Never
invent block_ids — every `block_id` must reference one from the input.
"""


SKIM_SCHEMA = """Schema:
{
  "summary": string,            // 2-3 short paragraphs, second-person ("you asked...")
  "themes": [string, ...],      // 1-3 short tags
  "checkpoints": [              // 3-5 entries
    {
      "block_id": string,       // must reference an input block_id
      "label": string,          // <= 60 chars, headline-style
      "description": string,    // one sentence, plain English
      "reason": string,         // one sentence: why this is a useful anchor
      "confidence": number,     // 0..1
      "category": string        // "decision" | "question" | "result" | "transition" | "blocker"
    }
  ]
}"""


STANDARD_SCHEMA = """Schema:
{
  "summary": string,            // 3-6 short paragraphs, second-person
  "themes": [string, ...],      // 3-6 short tags
  "decisions": [
    { "summary": string, "block_id": string }   // top decisions, anchored to blocks
  ],
  "open_questions": [string, ...], // unresolved threads, may be empty
  "checkpoints": [              // 4-10 entries
    {
      "block_id": string,
      "label": string,
      "description": string,
      "reason": string,
      "confidence": number,
      "category": string
    }
  ]
}"""


DEEP_PASS1_SCHEMA = """Schema (Pass 1 — extraction only):
{
  "themes": [string, ...],          // 3-8 tags
  "decisions": [
    { "summary": string, "block_id": string }
  ],
  "open_questions": [string, ...],
  "anchor_candidates": [
    {
      "block_id": string,
      "category": string,           // see categories
      "reason": string              // why this block could anchor a checkpoint
    }
  ]
}
Categories: "decision" | "question" | "result" | "transition" | "blocker".
Aim for 8-15 anchor candidates spread across the chain."""


DEEP_PASS2_SCHEMA = """Schema (Pass 2 — narrate around the themes/anchors from pass 1):
{
  "summary": string,                // 4-8 short paragraphs, second-person, organized by theme
  "checkpoints": [                  // 6-12 entries, drawn primarily from anchor_candidates
    {
      "block_id": string,
      "label": string,              // <= 60 chars
      "description": string,
      "reason": string,
      "confidence": number,
      "category": string
    }
  ]
}"""


def skim_prompt() -> str:
    return SHARED_PREAMBLE + "\n" + SKIM_SCHEMA + "\n\nGuidance:\n" + _shared_guidance(short=True)


def standard_prompt() -> str:
    return SHARED_PREAMBLE + "\n" + STANDARD_SCHEMA + "\n\nGuidance:\n" + _shared_guidance(short=False)


def deep_pass1_prompt() -> str:
    return (
        SHARED_PREAMBLE
        + "\n"
        + DEEP_PASS1_SCHEMA
        + "\n\nDo not write the narrative yet — only the extraction. "
        + "Cover the entire chain, not just the most recent blocks."
    )


def deep_pass2_prompt(pass1_json: str) -> str:
    return (
        SHARED_PREAMBLE
        + "\n"
        + DEEP_PASS2_SCHEMA
        + "\n\nYou previously produced this extraction (pass 1):\n"
        + pass1_json
        + "\n\nNow write the user-facing narrative and select checkpoints. "
        + "Anchor every checkpoint to a block_id from the input. Cite "
        + "anchor_candidates where they fit; you may also pick other blocks."
    )


def _shared_guidance(short: bool) -> str:
    lines = [
        "- Write second-person, friendly, concrete. No corporate jargon.",
        "- Anchor every checkpoint to a real block_id from the input.",
        "- Pick checkpoints that are *useful entry points for backtracking*: "
        "decisions, question moments, key results, topic transitions, blockers.",
        "- Spread checkpoints across the chain. Don't cluster them all near the end.",
        "- Confidence should reflect how certain you are the user will want to backtrack to this point.",
    ]
    if not short:
        lines.append(
            "- Surface themes that span multiple blocks, not labels for single blocks."
        )
        lines.append(
            "- Surface open_questions only when there is a genuinely unresolved thread."
        )
    return "\n".join(lines)
