from __future__ import annotations

import base64
import hashlib
import hmac
import json

import httpx
import pytest
import respx

from mcp_agent_system.integrations.allscale import (
    AllScaleClient,
    _build_canonical,
    _sha256_hex,
    stablecoin_enum,
    stablecoin_name,
)


def test_tokens_to_usd() -> None:
    client = AllScaleClient(
        api_key="pub",
        api_secret="sec",
        base_url="https://api.example",
        usd_per_1k_tokens=0.05,
        stablecoin=1,
    )
    assert client.tokens_to_usd(1000) == 0.05
    assert client.tokens_to_usd(50_000) == 2.5


def test_usd_to_cents_rounds() -> None:
    # Python's banker's rounding: 0.005*100=0.5 -> 0, 0.006*100=0.6 -> 1.
    assert AllScaleClient.usd_to_cents(0.0) == 0
    assert AllScaleClient.usd_to_cents(0.006) == 1
    assert AllScaleClient.usd_to_cents(1.234) == 123
    assert AllScaleClient.usd_to_cents(1.236) == 124


def test_stablecoin_name_and_enum_round_trip() -> None:
    assert stablecoin_name(1) == "USDT"
    assert stablecoin_enum("USDT") == 1
    assert stablecoin_enum("usdt") == 1
    assert stablecoin_enum(None, default=1) == 1


@pytest.mark.asyncio
async def test_create_invoice_signs_request_and_posts_to_checkout_intents() -> None:
    api_key = "pub-123"
    api_secret = "super-secret-value"
    client = AllScaleClient(
        api_key=api_key,
        api_secret=api_secret,
        base_url="https://api.example",
        usd_per_1k_tokens=0.04,
        stablecoin=1,
    )
    captured: dict[str, object] = {}

    def _record(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        captured["url"] = str(request.url)
        captured["body"] = bytes(request.content)
        return httpx.Response(
            200,
            json={
                "data": {
                    "intent_id": "ci_42",
                    "payment_url": "https://pay.example/ci_42",
                }
            },
        )

    with respx.mock:
        respx.post("https://api.example/v1/checkout_intents/").mock(side_effect=_record)
        invoice = await client.create_invoice(
            session_id="sess1",
            tokens_billed=2_500,
            customer_ref="user_x",
        )

    headers = captured["headers"]  # type: ignore[assignment]
    body = captured["body"]  # type: ignore[assignment]
    assert isinstance(headers, dict)
    assert isinstance(body, (bytes, bytearray))
    assert headers["x-api-key"] == api_key
    assert headers["x-timestamp"].isdigit()
    assert headers["x-nonce"]
    assert headers["x-signature"].startswith("v1=")

    body_obj = json.loads(body)
    # 2500 tokens at $0.04/1k = $0.10 = 10 cents (matches AllScale's floor).
    assert body_obj["amount_cents"] == 10
    assert body_obj["stable_coin"] == 1
    assert body_obj["user_id"] == "user_x"
    assert body_obj["extra"]["session_id"] == "sess1"

    canonical = _build_canonical(
        method="POST",
        path="/v1/checkout_intents/",
        query="",
        timestamp=headers["x-timestamp"],
        nonce=headers["x-nonce"],
        body_sha256_hex=_sha256_hex(bytes(body)),
    )
    expected_sig = base64.b64encode(
        hmac.new(api_secret.encode(), canonical.encode(), hashlib.sha256).digest()
    ).decode("ascii")
    assert headers["x-signature"] == f"v1={expected_sig}"

    assert invoice.invoice_ref == "ci_42"
    assert invoice.payment_url == "https://pay.example/ci_42"
    assert invoice.stablecoin == 1
    assert invoice.usd_amount == pytest.approx(0.10, rel=1e-3)


@pytest.mark.asyncio
async def test_create_invoice_enforces_min_payment_floor() -> None:
    """Tiny amounts (< 0.1 USDT) must be rounded up to the AllScale floor."""
    client = AllScaleClient(
        api_key="pub",
        api_secret="sec",
        base_url="https://api.example",
        usd_per_1k_tokens=0.001,  # makes 100 tokens worth $0.0001
        stablecoin=1,
    )
    captured_body: dict[str, object] = {}

    def _record(request: httpx.Request) -> httpx.Response:
        captured_body["body"] = json.loads(request.content)
        return httpx.Response(200, json={"data": {"intent_id": "ci_1"}})

    with respx.mock:
        respx.post("https://api.example/v1/checkout_intents/").mock(side_effect=_record)
        await client.create_invoice(
            session_id="s", tokens_billed=100, customer_ref="u"
        )
    assert captured_body["body"]["amount_cents"] == 10  # 0.1 USDT minimum
