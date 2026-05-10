"""Blockchain for Good Alliance (BGA) on-chain registry client.

Each correction logged via ``log_correction`` is hashed and recorded on a
public ``CorrectionRegistry`` contract on BNB Chain testnet (chain id 97).
The resulting transaction hash is persisted on the correction row so the
training-data commons is publicly auditable.

This supports the BGA's mission via:
    SDG 4  - transparent, auditable training data for educational AI
    SDG 16 - tamper-proof institutional accountability of AI corrections

If ``BGA_ENABLED=false`` or the contract address is missing, this client
becomes a graceful no-op so tools never fail because of chain issues.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from typing import Any

from eth_account import Account
from web3 import Web3

from ..utils.env import get_settings
from ..utils.logger import get_logger

log = get_logger(__name__)


CORRECTION_REGISTRY_ABI: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "recordCorrection",
        "stateMutability": "nonpayable",
        "inputs": [
            {"name": "sessionId", "type": "bytes16"},
            {"name": "originalHash", "type": "bytes32"},
            {"name": "correctionHash", "type": "bytes32"},
        ],
        "outputs": [],
    },
    {
        "type": "function",
        "name": "recordContextBlock",
        "stateMutability": "nonpayable",
        "inputs": [
            {"name": "sessionId", "type": "bytes16"},
            {"name": "blockId", "type": "bytes16"},
            {"name": "summaryHash", "type": "bytes32"},
        ],
        "outputs": [],
    },
    {
        "type": "event",
        "name": "CorrectionRecorded",
        "anonymous": False,
        "inputs": [
            {"indexed": True, "name": "sessionId", "type": "bytes16"},
            {"indexed": False, "name": "originalHash", "type": "bytes32"},
            {"indexed": False, "name": "correctionHash", "type": "bytes32"},
            {"indexed": False, "name": "timestamp", "type": "uint256"},
        ],
    },
    {
        "type": "event",
        "name": "ContextBlockRecorded",
        "anonymous": False,
        "inputs": [
            {"indexed": True, "name": "sessionId", "type": "bytes16"},
            {"indexed": False, "name": "blockId", "type": "bytes16"},
            {"indexed": False, "name": "summaryHash", "type": "bytes32"},
            {"indexed": False, "name": "timestamp", "type": "uint256"},
        ],
    },
]


def sha256_bytes32(text: str) -> bytes:
    return hashlib.sha256(text.encode("utf-8")).digest()


def uuid_to_bytes16(value: str) -> bytes:
    """Convert a UUID-string to a 16-byte payload."""
    try:
        return uuid.UUID(value).bytes
    except (ValueError, AttributeError):
        # Fall back to hashing if the input is not a valid UUID.
        return hashlib.sha256(value.encode("utf-8")).digest()[:16]


@dataclass
class ChainTx:
    tx_hash: str
    chain_id: int
    explorer_url: str


class ChainClient:
    """web3.py wrapper for the BGA CorrectionRegistry contract."""

    def __init__(
        self,
        *,
        rpc_url: str | None = None,
        chain_id: int | None = None,
        private_key: str | None = None,
        contract_address: str | None = None,
        explorer_base: str | None = None,
        enabled: bool | None = None,
    ) -> None:
        s = get_settings()
        self.rpc_url = rpc_url or s.resolved_bga_rpc_url
        self.chain_id = chain_id if chain_id is not None else s.bga_chain_id
        self.private_key = private_key if private_key is not None else s.bga_private_key
        self.contract_address = contract_address or s.bga_contract_address
        self.explorer_base = (explorer_base or s.bga_explorer_base).rstrip("/")
        env_enabled = s.bga_enabled if enabled is None else enabled
        # An unsubstituted ``{ALCHEMY_API_KEY}`` placeholder means the user
        # has not configured a provider key yet; treat it like "disabled" so
        # we degrade gracefully rather than 401-ing on every call.
        rpc_configured = bool(self.rpc_url) and "{ALCHEMY_API_KEY}" not in self.rpc_url
        self.enabled = bool(
            env_enabled
            and self.private_key
            and self.contract_address
            and rpc_configured
        )

    def _explorer(self, tx_hash: str) -> str:
        return f"{self.explorer_base}/{tx_hash}"

    def _account(self, w3: Web3) -> Any:
        return Account.from_key(self.private_key)

    def _contract(self, w3: Web3) -> Any:
        return w3.eth.contract(
            address=Web3.to_checksum_address(self.contract_address),
            abi=CORRECTION_REGISTRY_ABI,
        )

    def _send(self, w3: Web3, fn_call: Any) -> str:
        acct = self._account(w3)
        nonce = w3.eth.get_transaction_count(acct.address)
        gas_price = w3.eth.gas_price
        tx = fn_call.build_transaction(
            {
                "from": acct.address,
                "nonce": nonce,
                "chainId": self.chain_id,
                "gasPrice": gas_price,
                "gas": 200_000,
            }
        )
        signed = acct.sign_transaction(tx)
        raw = getattr(signed, "rawTransaction", None) or signed.raw_transaction
        tx_hash = w3.eth.send_raw_transaction(raw)
        return tx_hash.hex() if isinstance(tx_hash, (bytes, bytearray)) else str(tx_hash)

    def record_correction(
        self,
        *,
        session_id: str,
        original: str,
        correction: str,
    ) -> ChainTx | None:
        if not self.enabled:
            log.info("chain.disabled", op="record_correction")
            return None
        try:
            w3 = Web3(Web3.HTTPProvider(self.rpc_url))
            contract = self._contract(w3)
            tx_hash = self._send(
                w3,
                contract.functions.recordCorrection(
                    uuid_to_bytes16(session_id),
                    sha256_bytes32(original),
                    sha256_bytes32(correction),
                ),
            )
            normalized = tx_hash if tx_hash.startswith("0x") else f"0x{tx_hash}"
            log.info(
                "chain.correction_recorded",
                session_id=session_id,
                tx_hash=normalized,
                chain_id=self.chain_id,
            )
            return ChainTx(
                tx_hash=normalized,
                chain_id=self.chain_id,
                explorer_url=self._explorer(normalized),
            )
        except Exception as exc:
            log.warning("chain.record_correction_failed", error=str(exc))
            return None

    def record_context_block(
        self,
        *,
        session_id: str,
        block_id: str,
        summary: str,
    ) -> ChainTx | None:
        if not self.enabled:
            log.info("chain.disabled", op="record_context_block")
            return None
        try:
            w3 = Web3(Web3.HTTPProvider(self.rpc_url))
            contract = self._contract(w3)
            tx_hash = self._send(
                w3,
                contract.functions.recordContextBlock(
                    uuid_to_bytes16(session_id),
                    uuid_to_bytes16(block_id),
                    sha256_bytes32(summary),
                ),
            )
            normalized = tx_hash if tx_hash.startswith("0x") else f"0x{tx_hash}"
            log.info(
                "chain.context_block_recorded",
                session_id=session_id,
                block_id=block_id,
                tx_hash=normalized,
            )
            return ChainTx(
                tx_hash=normalized,
                chain_id=self.chain_id,
                explorer_url=self._explorer(normalized),
            )
        except Exception as exc:
            log.warning("chain.record_context_block_failed", error=str(exc))
            return None
