"""One lockfile row -> one fully-rendered reproduction episode.

Turns the audited lockfile (the selected-paper audit pool) into the inputs the
forked tool loop needs: the opening prompt and the per-episode run directory.

Lockfile loading and the per-row field accessors live in ``dataset``; the prompt
renderer lives in ``prompt_render``. This module is the orchestration seam: it
selects the episode row, resolves the run directory, and assembles the
``EpisodeInput`` / ``ExecutionContext``.

The run directory is resolved to ``<runs-dir>/<arxiv_id>/<budget>h/<run_id>/`` —
the layout contract the existing ``reclaim_vllm`` auditor reads (it walks
``<runs-dir>/<arxiv_id>`` recursively).
"""

from __future__ import annotations

import argparse
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reclaim_repro.context import Budget, ExecutionContext
from reclaim_repro.dataset import (
    DEFAULT_LOCKFILE_SPLIT,
    arxiv_id_of,
    band_max_hours,
    format_hours,
    load_lockfile_rows,
)
from reclaim_repro.prompt_render import render_reproduce_prompt
from reclaim_repro.reference import safe_component

# Compute ceiling for a row whose selection_band is missing/unparseable, used when
# the budget is derived per-paper from the band (the default) rather than pinned flat.
DEFAULT_UNBANDED_BUDGET_H100_HOURS = 8.0


@dataclass
class RunPaths:
    """The resolved bundle layout for one episode."""

    run_dir: Path
    workspace: Path
    reference: Path
    evidence: Path


@dataclass
class EpisodeInput:
    """One lockfile row turned into a ready-to-run episode."""

    arxiv_id: str
    row: dict[str, Any]
    prompt: str
    run_paths: RunPaths
    budget: float


def select_episode_rows(rows: dict[str, dict], *, paper_id: str | None) -> list[dict]:
    if not paper_id:
        raise SystemExit("Pass --paper-id <arxiv_id> to select the episode to reproduce.")
    row = rows.get(paper_id) or rows.get(paper_id.split("v")[0])
    if row is None:
        sample = ", ".join(sorted(rows)[:8])
        raise SystemExit(f"paper-id {paper_id!r} not in lockfile (e.g. {sample} ...).")
    return [row]


def new_run_id() -> str:
    """Time-stamped, randomly-suffixed id so re-runs never collide in one dir."""
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return f"{stamp}-{secrets.token_hex(3)}"


def resolve_run_paths(
    runs_dir: Path, arxiv_id: str, budget: float, run_id: str | None = None
) -> RunPaths:
    run_dir = (
        Path(runs_dir)
        / safe_component(arxiv_id)
        / f"{format_hours(budget)}h"
        / (run_id or new_run_id())
    )
    return RunPaths(
        run_dir=run_dir,
        workspace=run_dir / "workspace",
        reference=run_dir / "reference",
        evidence=run_dir / "evidence",
    )


def prepare_episodes(args: argparse.Namespace) -> list[EpisodeInput]:
    template = Path(args.prompt_file).read_text(encoding="utf-8")
    rows = load_lockfile_rows(
        getattr(args, "lockfile", None),
        split=getattr(args, "split", DEFAULT_LOCKFILE_SPLIT),
    )
    selected = select_episode_rows(rows, paper_id=args.paper_id)
    # Default: derive each paper's ceiling from its selection_band. A flat
    # --budget-h100-hours, when given, overrides the band for every paper.
    flat_override = getattr(args, "budget_h100_hours", None)
    pinned_run_id = getattr(args, "run_id", None)
    episodes: list[EpisodeInput] = []
    for row in selected:
        arxiv_id = arxiv_id_of(row)
        if flat_override is not None:
            budget = float(flat_override)
        else:
            band_budget = band_max_hours(row)
            budget = band_budget if band_budget is not None else DEFAULT_UNBANDED_BUDGET_H100_HOURS
        run_paths = resolve_run_paths(args.runs_dir, arxiv_id, budget, pinned_run_id)
        prompt = render_reproduce_prompt(template, row, budget=budget)
        episodes.append(
            EpisodeInput(arxiv_id=arxiv_id, row=row, prompt=prompt, run_paths=run_paths, budget=budget)
        )
    return episodes


def build_context(ep: EpisodeInput) -> ExecutionContext:
    """Construct the per-episode loop state from a prepared episode."""
    return ExecutionContext(
        arxiv_id=ep.arxiv_id,
        lockfile_row=ep.row,
        workspace=ep.run_paths.workspace,
        reference=ep.run_paths.reference,
        evidence=ep.run_paths.evidence,
        run_dir=ep.run_paths.run_dir,
        budget=Budget(total_h100_hours=ep.budget),
    )
