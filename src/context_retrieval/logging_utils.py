"""Structured logging for the Context Retrieval server.

Every tool call emits a single line of `key=value` pairs at INFO level so
operators can grep without parsing JSON. Errors include `error_type` and
`error_message`. Timing is reported in milliseconds.

We log to stderr (stdout is reserved for the MCP stdio transport).
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from typing import Any, Iterator

_CONFIGURED = False
_LOGGER_NAME = "context_retrieval"


def configure_logging(level: str = "INFO") -> None:
    """Idempotent root config. Safe to call multiple times."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(level.upper())
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
    )
    logger.addHandler(handler)
    logger.propagate = False
    _CONFIGURED = True


def get_logger(name: str | None = None) -> logging.Logger:
    full = _LOGGER_NAME if name is None else f"{_LOGGER_NAME}.{name}"
    return logging.getLogger(full)


def _format_fields(fields: dict[str, Any]) -> str:
    parts: list[str] = []
    for k, v in fields.items():
        if v is None:
            continue
        s = str(v)
        if any(c in s for c in (" ", "=", '"')):
            s = '"' + s.replace('"', '\\"') + '"'
        parts.append(f"{k}={s}")
    return " ".join(parts)


@contextmanager
def log_call(tool: str, **fields: Any) -> Iterator[dict[str, Any]]:
    """Log the start, end, and timing of a tool call.

    Usage::

        with log_call("query_chain", session=session_id, chain=chain_id) as ctx:
            result = do_work()
            ctx["matches"] = len(result.matches)

    Anything added to the yielded dict is included in the completion log
    line. Exceptions are caught, logged with `result=error`, and re-raised.
    """
    log = get_logger("tool")
    extras: dict[str, Any] = {}
    start = time.perf_counter()
    log.info(_format_fields({"event": "tool.start", "tool": tool, **fields}))
    try:
        yield extras
    except Exception as exc:
        duration_ms = round((time.perf_counter() - start) * 1000, 1)
        log.exception(
            _format_fields(
                {
                    "event": "tool.error",
                    "tool": tool,
                    "result": "error",
                    "duration_ms": duration_ms,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    **fields,
                    **extras,
                }
            )
        )
        raise
    else:
        duration_ms = round((time.perf_counter() - start) * 1000, 1)
        log.info(
            _format_fields(
                {
                    "event": "tool.end",
                    "tool": tool,
                    "result": "ok",
                    "duration_ms": duration_ms,
                    **fields,
                    **extras,
                }
            )
        )
