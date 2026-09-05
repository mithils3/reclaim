from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from reclaim_repro.inputs import resolve_run_paths
from reclaim_repro.workspace import create_layout, prepare_workspace

ROW = {
    "arxiv_id": "2505.11483",
    "paper_tex_files": [{"relative_path": "main.tex", "text": "x"}],
    "supplement_files": [{"relative_path": "code/run.py", "content": b"print(1)\n"}],
}


class CreateLayoutTests(unittest.TestCase):
    def test_creates_all_four_dirs(self):
        with tempfile.TemporaryDirectory() as d:
            paths = resolve_run_paths(Path(d) / "runs", "2505.11483", 8.0, run_id="RID")
            create_layout(paths)
            for sub in (paths.run_dir, paths.workspace, paths.reference, paths.evidence):
                self.assertTrue(sub.is_dir(), sub)


class PrepareWorkspaceTests(unittest.TestCase):
    def test_full_offline_setup(self):
        with tempfile.TemporaryDirectory() as d:
            paths = resolve_run_paths(Path(d) / "runs", "2505.11483", 8.0, run_id="RID")
            # The agent builds its own venv in the container; setup only lays down the
            # dir layout, evidence sinks, and the read-only reference copy.
            result = prepare_workspace(paths, arxiv_id="2505.11483", reference_row=ROW)
            # layout + evidence sinks
            self.assertTrue(paths.workspace.is_dir())
            self.assertTrue((paths.evidence / "commands.log").is_file())
            self.assertTrue((paths.evidence / "patches").is_dir())
            # reference materialized from the row (no network)
            self.assertTrue(result.reference["ok"])
            self.assertTrue((paths.reference / "supplement" / "code" / "run.py").is_file())
            self.assertTrue((paths.reference / "MANIFEST.txt").is_file())

    def test_can_skip_reference(self):
        with tempfile.TemporaryDirectory() as d:
            paths = resolve_run_paths(Path(d) / "runs", "x", 8.0, run_id="RID")
            result = prepare_workspace(paths, arxiv_id="x", materialize_ref=False)
            self.assertIsNone(result.reference)
            self.assertTrue(paths.workspace.is_dir())


if __name__ == "__main__":
    unittest.main()
