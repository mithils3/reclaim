"""Resolve one sbatch sweep's runs to the exact bundles a grader should read.

A sweep is a ``batch_id`` (``slurm-<jobid>``) shared by every ``reclaim_repro``
process the sbatch launched; each run leaves its bundle at
``<runs-dir>/<arxiv_id>/<budget>h/<run_id>/``. Naming a paper is therefore not
enough to name a run: a paper attempted more than once has several bundles, and
"the sweep's attempt" is only identifiable through the ``run_id`` its row carries.

This module reads the sweep's rows from the run-viewer telemetry database and
resolves each ``run_id`` to the bundle it actually wrote, so a grader points at one attempt and
its verdict patches the matching ``repro_runs`` row. Read-only; needs
``RECLAIM_TELEMETRY_URL`` + ``RECLAIM_TELEMETRY_KEY``.
"""

from __future__ import annotations

import json
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reclaim_repro import postgrest

HTTP_TIMEOUT = 20.0
SELECT = (
    "run_id,arxiv_id,status,model,budget,started_at,updated_at,"
    "batch_id,batch_label,audit_score,audit_verdict"
)


@dataclass
class Run:
    """One reproduction run of the sweep, and the bundle it wrote (once resolved)."""

    run_id: str
    arxiv_id: str
    status: str
    budget: float | None
    audit_score: int | None
    bundle: Path | None = None


def fetch_runs(base_url: str, key: str, batch_id: str) -> list[dict[str, Any]]:
    """Every ``repro_runs`` row of one batch, oldest first."""
    query = urllib.parse.urlencode(
        {"batch_id": f"eq.{batch_id}", "select": SELECT, "order": "started_at.asc"}
    )
    code, text = postgrest.request(
        f"{base_url.rstrip('/')}/rest/v1/repro_runs?{query}",
        service_key=key,
        method="GET",
        timeout=HTTP_TIMEOUT,
        max_attempts=4,
    )
    if not code or code >= 300:
        raise SystemExit(f"batch_runs: repro_runs query failed (HTTP {code}): {text[:300]}")
    rows = json.loads(text)
    if not isinstance(rows, list):
        raise SystemExit(f"batch_runs: unexpected response for {batch_id}: {text[:200]}")
    return rows


def select_runs(
    rows: list[dict[str, Any]],
    *,
    include_running: bool = False,
    skip_audited: bool = False,
) -> list[Run]:
    """One :class:`Run` per paper: the sweep's newest attempt, filtered.

    Running rows are dropped by default (their bundle is still being written, and
    grading a half-finished run scores the harness, not the model). ``skip_audited``
    drops papers that already carry a verdict, so a re-run only grades what is
    missing instead of paying for the whole sweep again.
    """
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        arxiv_id, run_id = row.get("arxiv_id"), row.get("run_id")
        if not arxiv_id or not run_id:
            continue
        if not include_running and row.get("status") == "running":
            continue
        if skip_audited and row.get("audit_score") is not None:
            continue
        previous = latest.get(arxiv_id)
        if previous is None or _started(row) >= _started(previous):
            latest[arxiv_id] = row
    return [
        Run(
            run_id=str(row["run_id"]),
            arxiv_id=str(row["arxiv_id"]),
            status=str(row.get("status") or "?"),
            budget=row.get("budget"),
            audit_score=row.get("audit_score"),
        )
        for row in sorted(latest.values(), key=lambda r: str(r.get("arxiv_id")))
    ]


def _started(row: dict[str, Any]) -> str:
    return str(row.get("started_at") or row.get("updated_at") or "")


def bundle_for(runs_dir: Path, arxiv_id: str, run_id: str) -> Path | None:
    """The bundle directory this run wrote: ``<runs-dir>/<arxiv_id>/<budget>h/<run_id>``.

    The budget dir is the only level between the paper and the run, so this globs
    that one level rather than walking the bundle (a finished workspace is a full
    clone plus checkpoints, and recursing it to find a path we already know costs
    minutes).
    """
    paper_dir = Path(runs_dir) / arxiv_id
    if not paper_dir.is_dir():
        return None
    for candidate in sorted(paper_dir.glob(f"*/{run_id}")):
        if candidate.is_dir():
            return candidate
    return None


def newest_bundle(runs_dir: Path, arxiv_id: str) -> Path | None:
    """The most recent bundle for a paper, for grading outside a sweep.

    Used when the caller names papers rather than a batch: with no ``run_id`` to
    match, the newest bundle that actually holds a run record (``report.json`` or
    ``stats.json``) is the only defensible choice. Only bundle roots are looked at
    (``<budget>h/<run_id>/``), so a report.json the agent happened to clone into its
    workspace is not mistaken for a run record.
    """
    paper_dir = Path(runs_dir) / arxiv_id
    if not paper_dir.is_dir():
        return None
    marked = [
        path.parent
        for name in ("report.json", "stats.json")
        for path in paper_dir.glob(f"*/*/{name}")
        if path.is_file()
    ]
    if not marked:
        return None
    return max(marked, key=lambda path: path.stat().st_mtime)
