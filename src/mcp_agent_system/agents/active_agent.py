"""ActiveAgent: the user-facing LLM interface.

Holds the live message array for the current context window, tracks
cumulative token usage, and exposes the methods consumed by the tool
handlers. Talks to the model through the CLōD gateway (OpenAI-compatible).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..integrations.clod import ClodClient, ClodCompletion, ClodUsage
from ..utils.prompts import DEFAULT_SYSTEM_PROMPT, SUMMARY_INJECTION_TEMPLATE
from ..utils.token_counter import estimate_tokens_from_text


@dataclass
class ActiveAgentConfig:
    session_id: str
    model_id: str
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    context_limit: int = 200_000
    transfer_threshold: float = 0.80


@dataclass
class SendResult:
    response: str
    usage: ClodUsage


@dataclass
class ActiveAgent:
    cfg: ActiveAgentConfig
    clod: ClodClient
    messages: list[dict[str, str]] = field(default_factory=list)
    token_count: int = 0

    @property
    def session_id(self) -> str:
        return self.cfg.session_id

    @property
    def model_id(self) -> str:
        return self.cfg.model_id

    @property
    def system_prompt(self) -> str:
        return self.cfg.system_prompt

    @property
    def context_limit(self) -> int:
        return self.cfg.context_limit

    @property
    def transfer_threshold(self) -> float:
        return self.cfg.transfer_threshold

    async def send_message(self, user_message: str) -> SendResult:
        """Append the user's message, call the model via CLōD, store the reply."""
        self.messages.append({"role": "user", "content": user_message})
        completion: ClodCompletion = await self.clod.chat(
            messages=self.messages,
            system=self.cfg.system_prompt,
            model=self.cfg.model_id,
        )
        self.messages.append({"role": "assistant", "content": completion.text})
        self.token_count += completion.usage.total
        return SendResult(response=completion.text, usage=completion.usage)

    async def send_system(self, system_message: str) -> SendResult:
        """Run a one-shot internal prompt that does not pollute the user-visible history."""
        completion = await self.clod.chat(
            messages=[{"role": "user", "content": system_message}],
            system=self.cfg.system_prompt,
            model=self.cfg.model_id,
            max_tokens=256,
        )
        self.token_count += completion.usage.total
        return SendResult(response=completion.text, usage=completion.usage)

    def get_full_context(self) -> list[dict[str, str]]:
        return list(self.messages)

    def reset_context(self) -> None:
        self.messages = []
        self.token_count = 0

    def inject_context_summary(
        self,
        *,
        summary: str,
        block_pointer: str,
        block_index: int,
        tokens_transferred: int,
    ) -> None:
        injection = SUMMARY_INJECTION_TEMPLATE.format(
            block_pointer=block_pointer,
            block_index=block_index,
            tokens_transferred=tokens_transferred,
            summary=summary,
        )
        self.messages.append({"role": "user", "content": injection})
        self.messages.append(
            {
                "role": "assistant",
                "content": "Understood. Continuing from the context summary.",
            }
        )
        self.token_count = estimate_tokens_from_text(injection)
