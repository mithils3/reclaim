from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Paper:
    arxiv_id: str
    # The agent's reproduction run directory the auditor explores.
    run_dir: str = ""
