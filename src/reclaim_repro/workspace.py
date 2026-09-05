"""Per-paper workspace setup for one reproduction episode.

Integrates the run-directory layout resolved in ``inputs.resolve_run_paths`` (the
shared "directory maker": ``<runs-dir>/<arxiv_id>/<budget>h/<run_id>/`` with a
fresh time+random ``run_id`` so re-runs of the same paper never collide) and
materializes everything the agent needs *before* the loop starts:

  - the directory layout (``workspace/`` rw, ``reference/`` ro, ``evidence/``),
  - the read-only ``reference/`` copy (paper LaTeX + every supplement file),
  - the durable ``evidence/`` sinks.

It deliberately does **not** clone the paper's code, install its dependencies, or
build a venv: that is the agent's job. The agent builds its venv as its first
setup step, inside the Apptainer container (see ``prompts/prompt_reproduce.txt``),
and installs a GH200 (aarch64) CUDA ``torch`` into it from the PyTorch wheel index
— the raw CUDA image has no prebuilt torch to inherit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from reclaim_repro import evidence as evidence_mod
from reclaim_repro import reference as reference_mod
from reclaim_repro.evidence import EvidencePaths
from reclaim_repro.inputs import RunPaths


@dataclass
class WorkspaceResult:
    """What ``prepare_workspace`` laid down, for reporting and the loop."""

    run_paths: RunPaths
    evidence: EvidencePaths
    reference: dict | None = None


def create_layout(run_paths: RunPaths) -> None:
    """Create the run-dir layout (run_dir, workspace, reference, evidence)."""
    for path in (run_paths.run_dir, run_paths.workspace, run_paths.reference, run_paths.evidence):
        Path(path).mkdir(parents=True, exist_ok=True)


def prepare_workspace(
    run_paths: RunPaths,
    *,
    arxiv_id: str,
    bundle_dataset: str = reference_mod.DEFAULT_DATASET,
    materialize_ref: bool = True,
    overwrite_reference: bool = False,
    reference_row: dict | None = None,
) -> WorkspaceResult:
    """Lay down one episode's bundle: dirs, evidence sinks, and the reference copy."""
    create_layout(run_paths)
    evidence = evidence_mod.init_evidence(run_paths.evidence)
    reference = None
    if materialize_ref:
        reference = reference_mod.materialize_reference(
            arxiv_id,
            run_paths.reference,
            dataset=bundle_dataset,
            overwrite=overwrite_reference,
            row=reference_row,
        )
    return WorkspaceResult(run_paths=run_paths, evidence=evidence, reference=reference)
