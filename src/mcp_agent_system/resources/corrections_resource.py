"""``agent://corrections/{id}`` resource.

Returns all correction records for a session in two shapes:

    * ``corrections``    - the raw rows (with on-chain proofs)
    * ``training_pairs`` - flattened (input, output) pairs ready for fine-tuning

Each row also includes a ``chain_explorer_url`` so reviewers can verify the
on-chain hash via the BGA registry.
"""

from __future__ import annotations

from ..db.repositories.correction_repo import CorrectionRepo
from ..db.session import session_scope
from ..utils.env import get_settings


def render_corrections_payload(session_id: str) -> dict:
    settings = get_settings()
    explorer_base = settings.bga_explorer_base.rstrip("/")
    with session_scope() as db:
        rows = CorrectionRepo.find_by_session(db, session_id)
        return {
            "session_id": session_id,
            "total": len(rows),
            "chain_explorer_base": explorer_base,
            "corrections": [
                {
                    "correction_id": r.correction_id,
                    "original": r.original,
                    "correction": r.correction,
                    "response": r.response,
                    "severity": r.severity,
                    "category": r.category,
                    "timestamp": r.timestamp,
                    "tx_hash": r.tx_hash,
                    "chain_id": r.chain_id,
                    "chain_explorer_url": (
                        f"{explorer_base}/{r.tx_hash}" if r.tx_hash else None
                    ),
                }
                for r in rows
            ],
            "training_pairs": [
                {
                    "input": r.original,
                    "output": r.correction,
                    "meta": {
                        "severity": r.severity,
                        "category": r.category,
                        "tx_hash": r.tx_hash,
                    },
                }
                for r in rows
            ],
        }
