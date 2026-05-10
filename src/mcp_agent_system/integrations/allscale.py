"""AllScale Checkout client.

AllScale (https://allscale.io) is a stablecoin checkout/invoicing OS. Auth
uses HMAC-SHA256 request signing with four headers:

    X-API-Key    -- public api_key (identifier)
    X-Timestamp  -- Unix seconds
    X-Nonce      -- random unique string
    X-Signature  -- "v1=" + base64(HMAC_SHA256(api_secret, canonical))

Where ``canonical`` is::

    METHOD\\n
    PATH\\n
    QUERY_STRING\\n
    TIMESTAMP\\n
    NONCE\\n
    sha256_hex(BODY)

We use ``POST /v1/checkout_intents/`` to create a stablecoin checkout
priced in cents. Currently AllScale only supports USDT (enum ``1``).
Token usage from CLōD is converted to USDT via ``ALLSCALE_USD_PER_1K_TOKENS``.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import time
from dataclasses import dataclass
from typing import Any

import httpx

from ..utils.env import get_settings
from ..utils.logger import get_logger
from ..utils.retry import with_retry

log = get_logger(__name__)


# AllScale's stable_coin field is an integer enum. Today the only supported
# coin is USDT (enum value 1). When AllScale adds more, extend this map.
STABLECOIN_NAME_TO_ENUM: dict[str, int] = {"USDT": 1}
STABLECOIN_ENUM_TO_NAME: dict[int, str] = {v: k for k, v in STABLECOIN_NAME_TO_ENUM.items()}


def stablecoin_name(value: int | str | None, default: int = 1) -> str:
    """Resolve a stablecoin spec (int enum or string) to its display name."""
    if value is None:
        return STABLECOIN_ENUM_TO_NAME.get(default, str(default))
    if isinstance(value, int):
        return STABLECOIN_ENUM_TO_NAME.get(value, str(value))
    return value.upper()


def stablecoin_enum(value: int | str | None, default: int = 1) -> int:
    """Resolve a stablecoin spec (int enum or string) to its int enum value."""
    if value is None:
        return default
    if isinstance(value, int):
        return value
    return STABLECOIN_NAME_TO_ENUM.get(value.upper(), default)


def _sha256_hex(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _build_canonical(
    *,
    method: str,
    path: str,
    query: str,
    timestamp: str,
    nonce: str,
    body_sha256_hex: str,
) -> str:
    return "\n".join([method.upper(), path, query, timestamp, nonce, body_sha256_hex])


def _sign(api_secret: str, canonical: str) -> str:
    digest = hmac.new(
        api_secret.encode("utf-8"), canonical.encode("utf-8"), hashlib.sha256
    ).digest()
    return base64.b64encode(digest).decode("ascii")


@dataclass
class AllScaleInvoice:
    invoice_ref: str
    payment_url: str | None
    usd_amount: float
    stablecoin: int
    raw: dict[str, Any]


class AllScaleClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_secret: str | None = None,
        base_url: str | None = None,
        usd_per_1k_tokens: float | None = None,
        stablecoin: int | None = None,
        redirect_url: str | None = None,
    ) -> None:
        s = get_settings()
        self.api_key = api_key or s.allscale_api_key
        self.api_secret = api_secret if api_secret is not None else s.allscale_api_secret
        self.base_url = (base_url or s.allscale_base_url).rstrip("/")
        self.usd_per_1k_tokens = (
            usd_per_1k_tokens
            if usd_per_1k_tokens is not None
            else s.allscale_usd_per_1k_tokens
        )
        self.stablecoin = stablecoin if stablecoin is not None else s.allscale_stablecoin
        self.redirect_url = redirect_url if redirect_url is not None else s.allscale_redirect_url

    def tokens_to_usd(self, tokens: int) -> float:
        return round((tokens / 1000.0) * self.usd_per_1k_tokens, 6)

    @staticmethod
    def usd_to_cents(usd: float) -> int:
        # AllScale rejects fractional cents; banker's rounding is fine here
        # because the 10-cent floor below dominates any sub-cent edge case.
        return max(0, round(usd * 100))

    def _headers_for(
        self, *, method: str, path: str, query: str, body: bytes
    ) -> dict[str, str]:
        timestamp = str(int(time.time()))
        nonce = secrets.token_hex(16)
        canonical = _build_canonical(
            method=method,
            path=path,
            query=query,
            timestamp=timestamp,
            nonce=nonce,
            body_sha256_hex=_sha256_hex(body),
        )
        signature = _sign(self.api_secret, canonical)
        return {
            "X-API-Key": self.api_key,
            "X-Timestamp": timestamp,
            "X-Nonce": nonce,
            "X-Signature": f"v1={signature}",
            "Content-Type": "application/json",
        }

    async def create_invoice(
        self,
        *,
        session_id: str,
        tokens_billed: int,
        customer_ref: str,
        stablecoin: int | str | None = None,
        description: str | None = None,
    ) -> AllScaleInvoice:
        """Create a stablecoin checkout intent for the session's token cost.

        AllScale's minimum payment is 0.1 USDT (10 cents). Smaller amounts
        are rounded up to the floor before submission.
        """
        coin = stablecoin_enum(stablecoin, default=self.stablecoin)
        usd_amount = self.tokens_to_usd(tokens_billed)
        cents = max(self.usd_to_cents(usd_amount), 10)  # 10 cents = 0.1 USDT floor

        path = "/v1/checkout_intents/"
        body_obj: dict[str, Any] = {
            "amount_cents": cents,
            "stable_coin": coin,
            "order_id": f"mcp-{session_id}-{int(time.time())}",
            "user_id": customer_ref,
            "order_description": description
            or f"MCP Agent System usage for session {session_id} ({tokens_billed} tokens).",
            "extra": {
                "session_id": session_id,
                "tokens_billed": tokens_billed,
                "rate_usd_per_1k_tokens": self.usd_per_1k_tokens,
            },
        }
        if self.redirect_url:
            body_obj["redirect_url"] = self.redirect_url

        import json as _json

        body_bytes = _json.dumps(body_obj, separators=(",", ":"), sort_keys=True).encode("utf-8")
        headers = self._headers_for(
            method="POST", path=path, query="", body=body_bytes
        )

        async def _do() -> dict[str, Any]:
            async with httpx.AsyncClient(timeout=30.0) as http:
                resp = await http.post(
                    f"{self.base_url}{path}",
                    headers=headers,
                    content=body_bytes,
                )
                resp.raise_for_status()
                return resp.json()

        data = await with_retry(_do)
        # AllScale wraps successful responses in a {data: {...}} envelope per
        # their docs; we tolerate both shapes.
        payload = data.get("data") if isinstance(data, dict) and "data" in data else data
        if not isinstance(payload, dict):
            payload = {}
        invoice_ref = str(
            payload.get("intent_id")
            or payload.get("id")
            or payload.get("checkout_intent_id")
            or ""
        )
        payment_url = (
            payload.get("payment_url")
            or payload.get("checkout_url")
            or payload.get("hosted_url")
        )
        log.info(
            "allscale.checkout_intent_created",
            session_id=session_id,
            invoice_ref=invoice_ref,
            cents=cents,
            stablecoin=coin,
        )
        return AllScaleInvoice(
            invoice_ref=invoice_ref,
            payment_url=payment_url,
            usd_amount=usd_amount,
            stablecoin=coin,
            raw=data if isinstance(data, dict) else {"raw": data},
        )
