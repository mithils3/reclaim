"""Durable evidence store for one reproduction episode.

The auditor trusts only what it can re-trace from disk, so every reproduction
episode writes a durable record under ``evidence/`` as it works:

  - ``commands.log``    -- one human-readable line per shell command (cmd, rc, cwd),
  - ``trajectory.jsonl``-- one JSON row per tool call / step (machine-readable),
  - ``env.lock``        -- the resolved environment (``uv pip freeze``),
  - ``patches/``        -- every diff the agent applied, saved verbatim.

These are *primitives*: ``init_evidence`` lays the files down at setup; the tools
(``workspace_bash``, ``edit_file``) and the loop append to them as the episode
runs. Microcompact can safely elide stale tool output from the conversation
precisely because the ground truth lives here, not in the model's context.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from reclaim_vllm.vllm.io import append_jsonl_row


@dataclass
class EvidencePaths:
    """Resolved locations of the four evidence sinks for one episode."""

    root: Path
    commands_log: Path
    trajectory: Path
    env_lock: Path
    patches_dir: Path


def evidence_paths(evidence_dir: Path) -> EvidencePaths:
    root = Path(evidence_dir)
    return EvidencePaths(
        root=root,
        commands_log=root / "commands.log",
        trajectory=root / "trajectory.jsonl",
        env_lock=root / "env.lock",
        patches_dir=root / "patches",
    )


def init_evidence(evidence_dir: Path) -> EvidencePaths:
    """Create the evidence dir + ``patches/`` and touch the append-only sinks."""
    paths = evidence_paths(evidence_dir)
    paths.patches_dir.mkdir(parents=True, exist_ok=True)
    for sink in (paths.commands_log, paths.trajectory, paths.env_lock):
        if not sink.exists():
            sink.write_text("", encoding="utf-8")
    return paths


def log_command(
    evidence_dir: Path,
    command: str,
    *,
    returncode: int | None = None,
    cwd: str | Path | None = None,
    duration_s: float | None = None,
) -> None:
    """Append one line to ``commands.log`` recording a shell command's outcome."""
    paths = evidence_paths(evidence_dir)
    paths.commands_log.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rc = "?" if returncode is None else str(returncode)
    parts = [stamp, f"rc={rc}"]
    if duration_s is not None:
        parts.append(f"{duration_s:.1f}s")
    if cwd is not None:
        parts.append(f"cwd={cwd}")
    header = " ".join(parts)
    with paths.commands_log.open("a", encoding="utf-8") as handle:
        handle.write(f"[{header}] {command}\n")


def append_trajectory(evidence_dir: Path, row: dict) -> None:
    """Append one structured step to ``trajectory.jsonl``."""
    append_jsonl_row(evidence_paths(evidence_dir).trajectory, row, truncate=False)


# Next sequence number for each numbered sink, keyed by (resolved dir, glob). Counting
# the directory again on every call is quadratic in the episode's steps, and a long run
# writes hundreds of patches and GPU step logs. The first call for a directory seeds
# from the glob — so a resumed episode continues its own numbering instead of
# overwriting — and later calls just increment. Keyed by directory, so a fresh episode
# in the same process starts from its own count.
_NEXT_SEQ: dict[tuple[str, str], int] = {}


def _next_seq(directory: Path, pattern: str) -> int:
    """The sequence number this call owns; the count advances whether or not it lands."""
    key = (str(directory), pattern)
    seq = _NEXT_SEQ.get(key)
    if seq is None:
        seq = sum(1 for _ in directory.glob(pattern))
    _NEXT_SEQ[key] = seq + 1
    return seq


def save_patch(evidence_dir: Path, diff: str, *, name: str | None = None) -> Path:
    """Persist a diff under ``patches/`` with a zero-padded sequence prefix."""
    paths = evidence_paths(evidence_dir)
    paths.patches_dir.mkdir(parents=True, exist_ok=True)
    seq = _next_seq(paths.patches_dir.resolve(), "*.diff")
    slug = _slug(name) if name else "patch"
    target = paths.patches_dir / f"{seq:04d}-{slug}.diff"
    target.write_text(diff if diff.endswith("\n") else diff + "\n", encoding="utf-8")
    return target


def next_gpu_log(evidence_dir: Path) -> Path:
    """Allocate the next ``gpu_step_<n>.log`` path (the streamed output of one GPU step).

    ``run_gpu`` tees each step's stdout/stderr here as it arrives, so a step SLURM
    kills at the --time wall still leaves its output on disk — for the agent (the
    file is visible in the container at ``/repro/evidence``) and for the auditor.
    """
    paths = evidence_paths(evidence_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    seq = _next_seq(paths.root.resolve(), "gpu_step_*.log")
    return paths.root / f"gpu_step_{seq:04d}.log"


def save_plan(evidence_dir: Path, rendered: str) -> Path:
    """Snapshot the agent's current plan to ``plan.md`` (overwritten each update)."""
    paths = evidence_paths(evidence_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    target = paths.root / "plan.md"
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    target.write_text(f"# Plan (updated {stamp})\n\n{rendered}\n", encoding="utf-8")
    return target


def _slug(name: str) -> str:
    keep = "".join(c if c.isalnum() or c in "-_." else "-" for c in name.strip())
    return keep.strip("-.")[:48] or "patch"
