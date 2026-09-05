from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from reclaim_repro import batch_runs


def _row(arxiv_id: str, run_id: str, **over) -> dict:
    row = {
        "arxiv_id": arxiv_id,
        "run_id": run_id,
        "status": "finished",
        "started_at": "2026-07-21T10:00:00Z",
        "audit_score": None,
    }
    row.update(over)
    return row


class SelectRunsTests(unittest.TestCase):
    def test_keeps_the_newest_attempt_per_paper(self) -> None:
        rows = [
            _row("2505.1", "old", started_at="2026-07-21T09:00:00Z"),
            _row("2505.1", "new", started_at="2026-07-21T11:00:00Z"),
        ]
        runs = batch_runs.select_runs(rows)
        self.assertEqual([(r.arxiv_id, r.run_id) for r in runs], [("2505.1", "new")])

    def test_drops_running_rows_by_default(self) -> None:
        rows = [_row("2505.1", "r1", status="running"), _row("2505.2", "r2")]
        self.assertEqual([r.arxiv_id for r in batch_runs.select_runs(rows)], ["2505.2"])
        self.assertEqual(
            [r.arxiv_id for r in batch_runs.select_runs(rows, include_running=True)],
            ["2505.1", "2505.2"],
        )

    def test_skip_audited_leaves_only_ungraded_papers(self) -> None:
        rows = [_row("2505.1", "r1", audit_score=7), _row("2505.2", "r2")]
        runs = batch_runs.select_runs(rows, skip_audited=True)
        self.assertEqual([r.arxiv_id for r in runs], ["2505.2"])

    def test_ignores_rows_without_ids(self) -> None:
        self.assertEqual(batch_runs.select_runs([{"status": "finished"}]), [])


class BundleResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.runs_dir = Path(self.tmp.name) / "agent_runs"
        # Two attempts at the same paper, as a re-run leaves behind.
        for run_id in ("run-old", "run-new"):
            (self.runs_dir / "2505.1" / "8h" / run_id).mkdir(parents=True)
        self.addCleanup(self.tmp.cleanup)

    def test_finds_the_bundle_of_one_specific_run(self) -> None:
        found = batch_runs.bundle_for(self.runs_dir, "2505.1", "run-new")
        self.assertEqual(found, self.runs_dir / "2505.1" / "8h" / "run-new")

    def test_missing_run_or_paper_is_none(self) -> None:
        self.assertIsNone(batch_runs.bundle_for(self.runs_dir, "2505.1", "run-gone"))
        self.assertIsNone(batch_runs.bundle_for(self.runs_dir, "9999.9", "run-new"))

    def test_newest_bundle_picks_the_latest_run_record(self) -> None:
        # No run_id to match (grading by paper id), so recency decides.
        old = self.runs_dir / "2505.1" / "8h" / "run-old" / "report.json"
        new = self.runs_dir / "2505.1" / "8h" / "run-new" / "report.json"
        old.write_text("{}", encoding="utf-8")
        new.write_text("{}", encoding="utf-8")
        os.utime(old, (1_000_000, 1_000_000))
        os.utime(new, (2_000_000, 2_000_000))
        self.assertEqual(
            batch_runs.newest_bundle(self.runs_dir, "2505.1"),
            self.runs_dir / "2505.1" / "8h" / "run-new",
        )

    def test_newest_bundle_ignores_dirs_with_no_run_record(self) -> None:
        self.assertIsNone(batch_runs.newest_bundle(self.runs_dir, "2505.1"))
        self.assertIsNone(batch_runs.newest_bundle(self.runs_dir, "9999.9"))


if __name__ == "__main__":
    unittest.main()
