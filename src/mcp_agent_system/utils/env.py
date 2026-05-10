"""Typed environment configuration loader using Pydantic Settings."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    # ── CLōD ──
    clod_api_key: str = Field(default="")
    clod_base_url: str = Field(default="https://api.clod.io/v1")
    clod_active_model: str = Field(default="anthropic/claude-opus-4")
    clod_summary_model: str = Field(default="meta-llama/llama-3.1-8b-instruct")
    clod_max_output_tokens: int = Field(default=8096)

    # ── Context window management ──
    context_transfer_threshold: float = Field(default=0.80, ge=0.1, le=0.99)
    context_limit_tokens: int = Field(default=200_000, ge=1024)

    # ── Database ──
    database_url: str = Field(default="sqlite:///./dev.db")

    # ── MCP server ──
    mcp_transport: Literal["stdio", "sse"] = Field(default="stdio")
    sse_host: str = Field(default="0.0.0.0")
    sse_port: int = Field(default=3001)
    log_level: str = Field(default="info")

    # ── Greptile ──
    greptile_api_key: str = Field(default="")
    greptile_github_token: str = Field(default="")
    greptile_base_url: str = Field(default="https://api.greptile.com/v2")

    # ── AllScale Checkout ──
    allscale_api_key: str = Field(default="")
    allscale_api_secret: str = Field(default="")
    allscale_base_url: str = Field(default="https://openapi.allscale.io")
    # 1 = USDT (currently the only stablecoin AllScale supports).
    allscale_stablecoin: int = Field(default=1)
    allscale_usd_per_1k_tokens: float = Field(default=0.02, ge=0.0)
    allscale_redirect_url: str = Field(default="")

    # ── Blockchain for Good Alliance ──
    # Default uses Alchemy's BNB testnet RPC; ``{ALCHEMY_API_KEY}`` in the URL
    # is substituted with ``alchemy_api_key`` at resolve time so users can
    # swap providers by simply overriding ``BGA_RPC_URL``.
    alchemy_api_key: str = Field(default="")
    bga_rpc_url: str = Field(default="https://bnb-testnet.g.alchemy.com/v2/{ALCHEMY_API_KEY}")
    bga_chain_id: int = Field(default=97)
    bga_private_key: str = Field(default="")
    bga_contract_address: str = Field(default="")
    bga_explorer_base: str = Field(default="https://testnet.bscscan.com/tx")
    bga_enabled: bool = Field(default=True)

    @property
    def resolved_bga_rpc_url(self) -> str:
        """RPC URL with ``{ALCHEMY_API_KEY}`` substituted from ``alchemy_api_key``."""
        return self.bga_rpc_url.replace("{ALCHEMY_API_KEY}", self.alchemy_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reload_settings() -> Settings:
    """Force reload of settings (useful in tests)."""
    get_settings.cache_clear()
    return get_settings()
