"""CLōD LLM gateway client.

CLōD (https://clod.io) exposes an OpenAI-compatible REST API at
``https://api.clod.io/v1`` that fans out to 30+ models with energy-smart
routing. We use the official ``openai`` Python SDK pointed at this base URL,
so every call to ``client.chat.completions.create`` is automatically routed
to the lowest-cost backend that satisfies the requested model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI

from ..utils.env import get_settings
from ..utils.logger import get_logger
from ..utils.prompts import SUMMARIZER_SYSTEM_PROMPT
from ..utils.retry import with_retry

log = get_logger(__name__)


@dataclass
class ClodUsage:
    input_tokens: int
    output_tokens: int

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class ClodCompletion:
    text: str
    usage: ClodUsage
    model: str
    raw: Any


class ClodClient:
    """Thin async wrapper around the OpenAI-compatible CLōD API."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        active_model: str | None = None,
        summary_model: str | None = None,
        max_output_tokens: int | None = None,
    ) -> None:
        s = get_settings()
        self.active_model = active_model or s.clod_active_model
        self.summary_model = summary_model or s.clod_summary_model
        self.max_output_tokens = max_output_tokens or s.clod_max_output_tokens
        self._client = AsyncOpenAI(
            api_key=api_key or s.clod_api_key or "missing-clod-api-key",
            base_url=base_url or s.clod_base_url,
        )

    async def chat(
        self,
        *,
        messages: list[dict[str, str]],
        system: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> ClodCompletion:
        """Run a chat completion. ``messages`` must use OpenAI-style roles."""
        model_id = model or self.active_model
        body_messages: list[dict[str, str]] = []
        if system:
            body_messages.append({"role": "system", "content": system})
        body_messages.extend(messages)

        async def _do() -> Any:
            return await self._client.chat.completions.create(
                model=model_id,
                messages=body_messages,  # type: ignore[arg-type]
                max_tokens=max_tokens or self.max_output_tokens,
            )

        resp = await with_retry(_do)
        text = (resp.choices[0].message.content or "").strip()
        usage = ClodUsage(
            input_tokens=getattr(resp.usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(resp.usage, "completion_tokens", 0) or 0,
        )
        log.info(
            "clod.chat",
            model=model_id,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
        )
        return ClodCompletion(text=text, usage=usage, model=model_id, raw=resp)

    async def summarize(self, messages: list[dict[str, str]]) -> ClodCompletion:
        """Summarize a conversation using the cheap CLōD model."""
        import json as _json

        return await self.chat(
            model=self.summary_model,
            system=SUMMARIZER_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _json.dumps(messages)}],
            max_tokens=512,
        )
