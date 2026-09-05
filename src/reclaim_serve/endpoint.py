"""The published endpoint contract.

A single small JSON file is the entire cross-repo seam. The server writes it once
it is healthy; a consumer (the reclaim runner, or anything else) reads ``base_url``
from it.

This module is the format's one definition: the writer and the reader sit together
here, and the consumer side imports ``read_base_url`` instead of re-parsing the
keys. The dependency only ever points this way — the serving half imports nothing
from the agent packages (see the package docstring).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def build_payload(
    *,
    base_url: str,
    host: str,
    port: int,
    served_model_name: str,
    model_path: str,
    node: str,
    job_id: str,
) -> dict[str, Any]:
    return {
        "base_url": base_url,
        "host": host,
        "port": port,
        "served_model_name": served_model_name,
        "model_path": model_path,
        "node": node,
        "job_id": job_id,
        "health": "ready",
    }


def write_endpoint(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write the endpoint file (write-tmp-then-rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def remove_endpoint(path: Path) -> None:
    """Remove a published endpoint file if present (best effort, on shutdown)."""
    try:
        path.unlink()
    except OSError:  # includes FileNotFoundError
        pass


def read_base_url(path: Path) -> str | None:
    """Read ``base_url`` from an endpoint file, or None if missing/invalid."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    url = data.get("base_url")
    return url if isinstance(url, str) and url else None
