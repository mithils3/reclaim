"""The existing auditor grades the repro bundle unchanged.

The reproduction agent writes its bundle to
``<runs-dir>/<arxiv_id>/<budget>h/<run_id>/`` -- ``report.json`` + ``evidence/``
alongside ``workspace/`` + ``reference/``. The auditor
(``reclaim_vllm --mode audit --runs-dir <root>``) maps ``paper_id`` ->
``<runs-dir>/<arxiv_id>`` and walks it recursively. This gate proves that the
**unmodified** ``reclaim_vllm`` audit pipeline -- the same calls
``run_arxiv_prompt_vllm.main`` makes in audit mode -- seeds its prompt from the
bundle, reaches every cited number through its read-only run-dir tools, and
renders a verdict, with **no changes to the auditor** and **no** ``result.json`` /
``repro.yaml`` (the agent reports; the auditor authors the verdict).

It builds the bundle through the real reproduction-agent code paths
(``resolve_run_paths`` + ``init_evidence`` + ``write_episode_report``) so the test
breaks if the layout contract drifts.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from reclaim_repro.context import ExecutionContext
from reclaim_repro.evidence import append_trajectory, init_evidence, log_command
from reclaim_repro.inputs import resolve_run_paths
from reclaim_repro.report import AGENT_ASSESSMENTS, REPORT_FILENAME, write_episode_report
from reclaim_repro.workspace import create_layout

# The auditor side, imported AS-IS -- nothing here is repro-aware.
from reclaim_vllm.audit.audit import finalize_audit_row
from reclaim_vllm.audit.inputs import build_audit_prompt
from reclaim_vllm.tools.run_dir_tools import list_run_files, read_run_file, run_bash
from run_arxiv_prompt_vllm import run_dir_for

ARXIV = "2510.21323"
BUDGET = 8.0
RUN_ID = "20260630T120000Z-m2gate"

# The three placeholders the audit prompt template carries (config.config).
AUDIT_TEMPLATE = "CLAIM:\n{CENTRAL_CLAIM}\n\nRUBRIC:\n{RUBRIC}\n\nBUNDLE:\n{RUN_BUNDLE}\n"


def claim_record() -> dict:
    """An audit-pool row as ``--mode audit`` loads it (central claim + pinned bar)."""
    return {
        "paper_id": ARXIV,
        "central_claim": "Method X reaches 76.5% top-1 accuracy on ImageNet-1k.",
        "match_target": {
            "config": "ViT-B/16, 300 epochs",
            "metric": "top-1 accuracy (%)",
            "value": "76.5%",
            "scope": "ImageNet-1k val",
            "match_bar_kind": "point_estimate",
        },
    }


def good_report() -> dict:
    """The agent's terminal account -- numbers cited into the bundle on disk."""
    return {
        "paper_id": ARXIV,
        "claim": "Method X reaches 76.5% top-1 on ImageNet-1k.",
        "what_ran": "Built a uv venv at workspace/.venv, cloned the repo, ran eval on val.",
        "scoring_command": "python eval.py --ckpt out/best.pt --split val",
        "measurements": [
            {
                "metric": "top-1 accuracy (%)",
                "observed_value": "76.3%",
                "reference_value": "76.5%",
                "scope": "ImageNet-1k val",
                # Citations are run-dir-relative (how the agent saw them); from the
                # auditor's paper-dir root they live under <budget>h/<run_id>/.
                "evidence": ["evidence/commands.log", "workspace/out/metrics.json"],
            }
        ],
        "agent_assessment": "partial",
        "changes_made": "Pinned torch 2.3 to match the image CUDA.",
        "blockers": "",
        "evidence_files": ["evidence/commands.log", "workspace/out/metrics.json"],
    }


def sample_auditor_verdict() -> dict:
    """A verdict the LLM auditor would emit after reading THIS bundle (audit schema)."""
    return {
        "paper_id": ARXIV,
        "central_claim": "76.5% top-1 on ImageNet-1k",
        "execution_verified": True,
        "execution_evidence": "ran eval.py; metrics.json present under workspace/out/",
        "measured_value": 76.3,
        "measured_citation": "workspace/out/metrics.json:1",
        "cheat_flags": [],
        "score": 10,
        "rationale": "76.3 within 5% of 76.5; traced to metrics.json the report cites.",
    }


def build_bundle(runs_dir: Path, *, n_supp: int = 4, n_ws: int = 6) -> Path:
    """Lay one paper's bundle down through the real code paths; return its run dir."""
    rp = resolve_run_paths(runs_dir, ARXIV, BUDGET, run_id=RUN_ID)
    create_layout(rp)

    # evidence/ -- the durable, auditor-trusted store (sorts first in the walk).
    init_evidence(rp.evidence)
    log_command(rp.evidence, "python eval.py --ckpt out/best.pt --split val",
                returncode=0, cwd="workspace")
    append_trajectory(rp.evidence, {"type": "run_gpu", "gpus": 1, "minutes": 12})

    # reference/ -- read-only paper copy (latex/ + supplement/), shaped like
    # reference.write_paper but built offline so the gate needs no HF network.
    (rp.reference / "latex").mkdir(parents=True, exist_ok=True)
    (rp.reference / "latex" / "main.tex").write_text("\\documentclass{article}", encoding="utf-8")
    supp = rp.reference / "supplement"
    supp.mkdir(parents=True, exist_ok=True)
    for i in range(n_supp):
        (supp / f"supp_{i:04d}.py").write_text(f"# supplement file {i}\n", encoding="utf-8")
    (rp.reference / "MANIFEST.txt").write_text("reference manifest\n", encoding="utf-8")

    # workspace/ -- the agent's editable clone. The per-paper uv venv lives at
    # workspace/.venv, which the auditor's SKIP_DIRS excludes; the scoring artifact
    # the report cites lives under workspace/out/.
    venv = rp.workspace / ".venv" / "lib" / "python3.11" / "site-packages"
    venv.mkdir(parents=True, exist_ok=True)
    for i in range(40):
        (venv / f"pkg_{i:03d}.py").write_text("x = 1\n", encoding="utf-8")
    (rp.workspace / "out").mkdir(exist_ok=True)
    (rp.workspace / "out" / "metrics.json").write_text(
        json.dumps({"top1": 76.3, "split": "val"}), encoding="utf-8"
    )
    for i in range(n_ws):
        (rp.workspace / f"mod_{i:02d}.py").write_text("pass\n", encoding="utf-8")

    # report.json -- the Phase-5 forced-final-pass writer (the real thing).
    ctx = ExecutionContext(arxiv_id=ARXIV, run_dir=rp.run_dir, evidence=rp.evidence)
    written = write_episode_report(ctx, json.dumps(good_report()), "natural")
    assert written == rp.run_dir / REPORT_FILENAME
    return rp.run_dir


def rel_in_paper_dir(run_dir: Path, paper_dir: Path, *parts: str) -> str:
    """A bundle path as the auditor addresses it: relative to ``<runs-dir>/<arxiv_id>``."""
    return str(run_dir.relative_to(paper_dir).joinpath(*parts))


class AuditEntrySeamTests(unittest.TestCase):
    def test_paper_id_binds_to_runs_dir_arxiv_id(self) -> None:
        # The layout contract: one paper -> one directory the auditor walks.
        with tempfile.TemporaryDirectory() as tmp:
            runs_dir = Path(tmp)
            self.assertEqual(run_dir_for(runs_dir, ARXIV), str(runs_dir / ARXIV))

    def test_existing_audit_prompt_seeds_from_the_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runs_dir = Path(tmp)
            build_bundle(runs_dir)
            prompt = build_audit_prompt(
                AUDIT_TEMPLATE, "RUBRIC_TEXT", claim_record(), ARXIV, runs_dir
            )
            # The pinned bar is adopted verbatim (claim_block), the rubric is filled,
            # and the bundle manifest names the agent's account + its evidence.
            self.assertIn("Pinned success bar", prompt)
            self.assertIn("RUBRIC_TEXT", prompt)
            self.assertIn("AGENT RUN DIRECTORY", prompt)
            self.assertIn(REPORT_FILENAME, prompt)
            self.assertIn("evidence/commands.log", prompt)
            self.assertNotIn("{RUN_BUNDLE}", prompt)


class AuditToolsReachEveryNumberTests(unittest.TestCase):
    def test_tools_open_report_evidence_and_the_cited_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runs_dir = Path(tmp)
            run_dir = build_bundle(runs_dir)
            paper_dir = runs_dir / ARXIV  # the auditor's run-dir root

            report_rel = rel_in_paper_dir(run_dir, paper_dir, REPORT_FILENAME)
            report = read_run_file({"path": report_rel}, paper_dir)
            self.assertTrue(report["ok"])
            parsed = json.loads(report["text"])
            self.assertEqual(parsed["report_status"], "complete")

            # The number the report cites is traceable from the auditor's root.
            artifact_rel = rel_in_paper_dir(run_dir, paper_dir, "workspace", "out", "metrics.json")
            artifact = read_run_file({"path": artifact_rel}, paper_dir)
            self.assertTrue(artifact["ok"])
            self.assertIn("76.3", artifact["text"])

            commands_rel = rel_in_paper_dir(run_dir, paper_dir, "evidence", "commands.log")
            self.assertTrue(read_run_file({"path": commands_rel}, paper_dir)["ok"])

            # And a recursive listing (the auditor's discovery path) surfaces them all.
            listing = list_run_files({"recursive": True}, paper_dir)
            paths = {e["path"] for e in listing["entries"]}
            self.assertIn(report_rel, paths)
            self.assertIn(artifact_rel, paths)

    def test_venv_is_excluded_from_the_auditor_view(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runs_dir = Path(tmp)
            build_bundle(runs_dir)
            paper_dir = runs_dir / ARXIV
            listing = list_run_files({"recursive": True}, paper_dir)
            self.assertFalse(
                any(".venv" in e["path"] for e in listing["entries"]),
                "the per-paper uv venv must not pollute the auditor's manifest",
            )


class VerdictIsTheAuditorsTests(unittest.TestCase):
    def test_report_is_an_account_not_a_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = build_bundle(Path(tmp))
            report = json.loads((run_dir / REPORT_FILENAME).read_text(encoding="utf-8"))
            # The agent states an honest self-assessment...
            self.assertIn(report["agent_assessment"], AGENT_ASSESSMENTS)
            # ...but writes NO graded score / verdict: those are the auditor's alone.
            self.assertNotIn("score", report)
            self.assertNotIn("verdict", report)
            # The two roles even use distinct vocabularies (no shared self-grading).
            self.assertIn("could_not_run", AGENT_ASSESSMENTS)  # agent-only
            self.assertNotIn("unverifiable", AGENT_ASSESSMENTS)  # auditor-only

    def test_no_result_json_or_repro_yaml_in_the_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = build_bundle(Path(tmp))
            names = {p.name for p in run_dir.rglob("*") if p.is_file()}
            self.assertIn(REPORT_FILENAME, names)
            self.assertNotIn("result.json", names)
            self.assertNotIn("repro.yaml", names)

    def test_existing_finalizer_grades_a_verdict_over_the_bundle(self) -> None:
        # The deterministic grade pipeline runs UNCHANGED over a verdict an auditor
        # would emit from this bundle: score 10 -> reproduced, status verified.
        with tempfile.TemporaryDirectory() as tmp:
            build_bundle(Path(tmp))
            row = finalize_audit_row(sample_auditor_verdict(), {"exit_reason": "natural"})
            self.assertEqual(row["verdict"], "reproduced")
            self.assertTrue(row["reproduced"])
            self.assertEqual(row["score"], 10)
            self.assertEqual(row["verification_status"], "verified")
            self.assertFalse(row["has_high_cheat_flag"])


class BulkyBundleStillGradeableTests(unittest.TestCase):
    """A bulky reference/ can push report.json past the 200-entry SEED manifest, but
    the auditor's tools still reach it -- so grading stays possible with zero changes.
    Documents the seed-truncation behavior as a tested property, not a surprise."""

    def test_report_truncated_from_seed_but_reachable_by_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runs_dir = Path(tmp)
            run_dir = build_bundle(runs_dir, n_supp=400)
            paper_dir = runs_dir / ARXIV

            prompt = build_audit_prompt(
                AUDIT_TEMPLATE, "R", claim_record(), ARXIV, runs_dir
            )
            # The seed lists the first 200 sorted paths; reference/ sorts before
            # report.json, so the agent's account is crowded out of the seed text...
            self.assertNotIn(REPORT_FILENAME, prompt)
            self.assertIn("more", prompt)  # "... and N more" tells the auditor to look

            # ...yet every auditor tool still reaches it (find / read / recursive list).
            found = run_bash({"command": "find . -name report.json"}, paper_dir)
            self.assertTrue(found["ok"])
            self.assertIn(REPORT_FILENAME, found["stdout"])
            report_rel = rel_in_paper_dir(run_dir, paper_dir, REPORT_FILENAME)
            self.assertTrue(read_run_file({"path": report_rel}, paper_dir)["ok"])


if __name__ == "__main__":
    unittest.main()
