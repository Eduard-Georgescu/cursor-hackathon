"""Retry helpers built on tenacity.

Retries transient HTTP / network failures with exponential backoff.
"""

from __future__ import annotations

from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

RETRYABLE_HTTP_STATUSES: frozenset[int] = frozenset({408, 425, 429, 500, 502, 503, 504, 529})


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRYABLE_HTTP_STATUSES
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError)):
        return True
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if isinstance(status, int) and status in RETRYABLE_HTTP_STATUSES:
        return True
    return False


def retry_async(max_attempts: int = 3, base_seconds: float = 0.5) -> AsyncRetrying:
    return AsyncRetrying(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=base_seconds, max=8.0),
        reraise=True,
    )


async def with_retry(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Call ``await fn(*args, **kwargs)`` with the default retry policy."""
    async for attempt in retry_async():
        with attempt:
            return await fn(*args, **kwargs)
    raise RuntimeError("Unreachable")
