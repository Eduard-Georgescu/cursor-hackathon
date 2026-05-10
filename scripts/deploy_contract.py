"""Compile + deploy ``CorrectionRegistry.sol`` to BNB Chain testnet.

Usage::

    python scripts/deploy_contract.py

Reads ``BGA_RPC_URL``, ``BGA_PRIVATE_KEY``, and ``BGA_CHAIN_ID`` from the
environment. On success, prints the deployed contract address and writes
it to a marker file (``contracts/.deployed_address``) so the operator can
copy it into ``.env`` as ``BGA_CONTRACT_ADDRESS``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from eth_account import Account
from solcx import compile_source, install_solc
from web3 import Web3

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "contracts" / "CorrectionRegistry.sol"
ADDRESS_OUTPUT = ROOT / "contracts" / ".deployed_address"


def main() -> int:
    rpc = os.environ.get(
        "BGA_RPC_URL",
        "https://bnb-testnet.g.alchemy.com/v2/{ALCHEMY_API_KEY}",
    )
    alchemy_key = os.environ.get("ALCHEMY_API_KEY", "")
    rpc = rpc.replace("{ALCHEMY_API_KEY}", alchemy_key)
    if "{ALCHEMY_API_KEY}" in rpc or (
        "alchemy.com" in rpc and rpc.rstrip("/").endswith("/v2")
    ):
        print(
            "ERROR: ALCHEMY_API_KEY is not set (or BGA_RPC_URL still contains "
            "the {ALCHEMY_API_KEY} placeholder). Set ALCHEMY_API_KEY, or override "
            "BGA_RPC_URL with a fully-formed provider URL.",
            file=sys.stderr,
        )
        return 1
    chain_id = int(os.environ.get("BGA_CHAIN_ID", "97"))
    pk = os.environ.get("BGA_PRIVATE_KEY", "")
    if not pk:
        print("ERROR: BGA_PRIVATE_KEY is not set", file=sys.stderr)
        return 1

    install_solc("0.8.24")
    source = CONTRACT_PATH.read_text(encoding="utf-8")
    compiled = compile_source(source, output_values=["abi", "bin"], solc_version="0.8.24")
    _, artifact = next(iter(compiled.items()))

    w3 = Web3(Web3.HTTPProvider(rpc))
    account = Account.from_key(pk)
    contract = w3.eth.contract(abi=artifact["abi"], bytecode=artifact["bin"])

    nonce = w3.eth.get_transaction_count(account.address)
    tx = contract.constructor().build_transaction(
        {
            "from": account.address,
            "nonce": nonce,
            "chainId": chain_id,
            "gasPrice": w3.eth.gas_price,
            "gas": 1_500_000,
        }
    )
    signed = account.sign_transaction(tx)
    raw = getattr(signed, "rawTransaction", None) or signed.raw_transaction
    tx_hash = w3.eth.send_raw_transaction(raw)
    print(f"Deploying... tx={tx_hash.hex()}")
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    address = receipt.contractAddress
    print(f"Deployed CorrectionRegistry at: {address}")
    print(f"Add to .env: BGA_CONTRACT_ADDRESS={address}")
    ADDRESS_OUTPUT.write_text(address + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
