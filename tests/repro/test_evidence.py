from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from reclaim_repro import evidence


class EvidenceTests(unittest.TestCase):
    def test_init_lays_down_all_sinks(self):
        with tempfile.TemporaryDirectory() as d:
            paths = evidence.init_evidence(Path(d) / "evidence")
            self.assertTrue(paths.commands_log.is_file())
            self.assertTrue(paths.trajectory.is_file())
            self.assertTrue(paths.env_lock.is_file())
            self.assertTrue(paths.patches_dir.is_dir())

    def test_log_command_appends_line_with_rc(self):
        with tempfile.TemporaryDirectory() as d:
            ev = Path(d) / "evidence"
            evidence.init_evidence(ev)
            evidence.log_command(ev, "git clone repo code", returncode=0, cwd="/ws")
            evidence.log_command(ev, "false", returncode=1)
            lines = (ev / "commands.log").read_text().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertIn("git clone repo code", lines[0])
            self.assertIn("rc=0", lines[0])
            self.assertIn("rc=1", lines[1])

    def test_append_trajectory_is_jsonl(self):
        with tempfile.TemporaryDirectory() as d:
            ev = Path(d) / "evidence"
            evidence.init_evidence(ev)
            evidence.append_trajectory(ev, {"step": 1, "tool": "workspace_bash"})
            evidence.append_trajectory(ev, {"step": 2, "tool": "read_file"})
            import json

            rows = [json.loads(line) for line in (ev / "trajectory.jsonl").read_text().splitlines() if line]
            self.assertEqual([r["step"] for r in rows], [1, 2])

    def test_save_patch_sequences_and_slugs(self):
        with tempfile.TemporaryDirectory() as d:
            ev = Path(d) / "evidence"
            evidence.init_evidence(ev)
            first = evidence.save_patch(ev, "--- a\n+++ b\n", name="train.py")
            second = evidence.save_patch(ev, "diff two", name="model.py")
            self.assertEqual(first.name, "0000-train.py.diff")
            self.assertEqual(second.name, "0001-model.py.diff")
            self.assertTrue(first.read_text().endswith("\n"))


if __name__ == "__main__":
    unittest.main()
