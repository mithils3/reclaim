"""The streaming ``agent.log`` view written alongside the evidence sinks."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from reclaim_repro import live_log
from reclaim_repro.context import ExecutionContext


def _ctx(root: Path) -> ExecutionContext:
    (root / "evidence").mkdir(parents=True, exist_ok=True)
    return ExecutionContext(arxiv_id="2506.09045", evidence=root / "evidence")


class LiveLogTests(unittest.TestCase):
    def test_writes_sibling_of_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = _ctx(Path(d))
            live_log.log_round_open(ctx, {"reasoning": "plan"}, round_index=0)
            log = Path(d) / "agent.log"
            self.assertTrue(log.is_file())  # <run_dir>/agent.log, not inside evidence/
            self.assertFalse((Path(d) / "evidence" / "agent.log").exists())

    def test_streams_incrementally_call_before_result(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = _ctx(Path(d))
            live_log.log_round_open(ctx, {"reasoning": "I will clone the repo."}, round_index=1)
            call = {"function": {"name": "workspace_bash", "arguments": '{"command": "git clone x"}'}}
            live_log.log_call_start(ctx, call)
            mid = (Path(d) / "agent.log").read_text()
            # The command line is on disk before the result exists (streamed).
            self.assertIn("$ git clone x", mid)
            self.assertNotIn("rc=0", mid)
            live_log.log_call_result(ctx, {"ok": True, "returncode": 0, "duration_s": 2.4,
                                           "stdout": "Cloning into x"})
            full = (Path(d) / "agent.log").read_text()
            self.assertIn("I will clone the repo.", full)
            self.assertIn("rc=0", full)
            self.assertIn("Cloning into x", full)

    def test_final_records_exit_reason(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = _ctx(Path(d))
            live_log.log_final(ctx, {"content": "done"}, round_index=4, exit_reason="natural")
            text = (Path(d) / "agent.log").read_text()
            self.assertIn("FINAL", text)
            self.assertIn("exit=natural", text)
            self.assertIn("done", text)

    def test_full_log_keeps_everything_agent_log_truncates(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = _ctx(Path(d))
            n = live_log.HEAD_LINES + 30
            live_log.log_round_open(
                ctx, {"reasoning": "\n".join(f"think {i}" for i in range(n))}, round_index=0
            )
            live_log.log_call_result(
                ctx, {"ok": True, "stdout": "\n".join(f"out {i}" for i in range(n))}
            )

            skim = (Path(d) / "agent.log").read_text()
            full = (Path(d) / "agent.full.log").read_text()

            # agent.log head-truncates each block and marks the elision...
            self.assertIn("more line", skim)
            self.assertNotIn(f"think {n - 1}", skim)
            self.assertNotIn(f"out {n - 1}", skim)
            # ...agent.full.log keeps every line and never elides.
            self.assertNotIn("more line", full)
            self.assertIn("think 0", full)
            self.assertIn(f"think {n - 1}", full)
            self.assertIn(f"out {n - 1}", full)

    def test_no_evidence_is_a_noop(self):
        ctx = ExecutionContext(arxiv_id="x", evidence=None)
        live_log.log_round_open(ctx, {"reasoning": "x"})  # must not raise


if __name__ == "__main__":
    unittest.main()
