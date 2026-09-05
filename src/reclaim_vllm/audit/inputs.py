"""Audit-mode inputs: render the central claim and the agent run-directory manifest.

The auditor grades one agent reproduction attempt per paper. Its inputs are:
  - the paper's ``central_claim`` (from the audit pool), and
  - a manifest of the agent's run directory (its *.log files, output artifacts,
    and any code it wrote), which the auditor then explores with the
    path-confined run-dir tools (list_run_files / read_run_file / write_run_file
    / bash — the last two can write a re-scoring script into the run dir).
"""

from __future__ import annotations

import json
from pathlib import Path

from reclaim_vllm.config.config import BUNDLE_PLACEHOLDER, CLAIM_PLACEHOLDER, RUBRIC_PLACEHOLDER
from reclaim_vllm.tools.run_dir_tools import run_dir_manifest

RUN_BUNDLE_NO_DIR_TEXT = (
    "(No --runs-dir configured, so no agent reproduction run directory is bound "
    "for this paper. With no run to inspect, the only defensible verdict is "
    "`unverifiable` with score 1.)"
)


def load_audit_rubric(path) -> str:
    return Path(str(path)).read_text(encoding="utf-8")


def claim_block(record: dict | None) -> str:
    if not record:
        return "(no central claim found for this paper)"
    parts: list[str] = []
    claim = record.get("central_claim")
    if isinstance(claim, str) and claim.strip():
        parts.append(claim.strip())
    evidence = {key: record[key] for key in ("claim_evidence", "mre_config") if record.get(key)}
    if evidence:
        parts.append(
            "\nReported numbers / experiment context:\n"
            + json.dumps(evidence, ensure_ascii=False, indent=2)
        )
    # The success bar is now PINNED in the lockfile as a coherent
    # (config, metric, value, scope, match_bar_kind) tuple. The auditor ADOPTS it
    # verbatim and sets only op / tolerance — it does not re-derive the bar.
    target = record.get("match_target")
    if isinstance(target, dict) and any(target.get(k) for k in target):
        parts.append(
            "\nPinned success bar (adopt verbatim — do NOT re-derive):\n"
            + json.dumps(target, ensure_ascii=False, indent=2)
            + "\nUse this tuple's config / metric / value (reference_value) / scope "
            "and match_bar_kind as given; set only `op` and `tolerance` to match the "
            "pinned match_bar_kind, per rubric C1. The one exception: if the paper in "
            "reference/ directly contradicts this tuple (wrong metric, reference "
            "value, table cell, or scope), trust the paper, correct the bar to it, "
            "and record the discrepancy in methodology_notes."
        )
    else:
        # Legacy rows with no pinned tuple fall back to deriving the bar.
        parts.append(
            "\nNo success bar is pinned for this paper. Derive the C1 match bar "
            "(classify match_bar_kind first, then set op / reference_value / "
            "tolerance) from the claim and reported numbers above, per the rubric."
        )
    return "\n".join(parts) if parts else "(no central claim found for this paper)"


def load_run_bundle(paper_id: str, runs_dir) -> str:
    # The prompt is seeded with a manifest of the paper's run directory; the
    # auditor reads the file contents on demand through the run-dir tools.
    if not runs_dir:
        return RUN_BUNDLE_NO_DIR_TEXT
    return run_dir_manifest(Path(str(runs_dir)) / paper_id)


def build_audit_prompt(
    template: str, rubric: str, record: dict | None, paper_id: str, runs_dir
) -> str:
    """Render the audit prompt for the bundle at ``<runs-dir>/<paper_id>``."""
    return render_audit_prompt(template, rubric, record, load_run_bundle(paper_id, runs_dir))


def build_audit_prompt_for_dir(template: str, rubric: str, record: dict | None, run_dir) -> str:
    """Same prompt, for a bundle addressed directly instead of by naming convention.

    A grader that already knows which run it is grading (e.g. one run of a sweep,
    resolved by ``run_id``) has the bundle path in hand and should not have to
    fake the ``<runs-dir>/<paper_id>`` layout to get a prompt rendered.
    """
    bundle = run_dir_manifest(Path(str(run_dir))) if run_dir else RUN_BUNDLE_NO_DIR_TEXT
    return render_audit_prompt(template, rubric, record, bundle)


def render_audit_prompt(template: str, rubric: str, record: dict | None, bundle: str) -> str:
    return (
        template.replace(CLAIM_PLACEHOLDER, claim_block(record))
        .replace(RUBRIC_PLACEHOLDER, rubric)
        .replace(BUNDLE_PLACEHOLDER, bundle)
    )
