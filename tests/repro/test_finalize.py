from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from reclaim_repro import live_log
from reclaim_repro.context import ExecutionContext
from reclaim_repro.evidence import init_evidence
from reclaim_repro.loop import finalize_episode


def make_args(tmp: Path) -> argparse.Namespace:
    return argparse.Namespace(
        tool_rounds=300,
        max_input_tokens=100_000,
        final_no_tools_message="Return your final report as a single JSON object.",
        output=tmp / "reproduce.jsonl",
        save_round_jsonl=False,
        trace_output=tmp / "trace.jsonl",
    )


def make_ctx(tmp: Path, arxiv_id: str = "2510.21323") -> ExecutionContext:
    run_dir = tmp / arxiv_id / "8h" / "run1"
    evidence = run_dir / "evidence"
    init_evidence(evidence)
    return ExecutionContext(arxiv_id=arxiv_id, run_dir=run_dir, evidence=evidence)


class FinalizeEpisodeTests(unittest.TestCase):
    def tearDown(self) -> None:
        # Never leak a registered sink into sibling tests.
        live_log.register_sink(None)

    def test_failed_model_call_finalizes_with_degraded_report(self) -> None:
        # The error path (loop.handle_request_done on a raising model call) hands an empty
        # message + exit_reason="error". The run must still land a terminal report.json and
        # a final row, rather than stranding as a permanently "running" row with no account.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ctx = make_ctx(tmp_path)
            conversations = {ctx.arxiv_id: []}
            final_rows: dict[str, dict] = {}
            finalize_episode(
                ctx.arxiv_id,
                {"custom_id": ctx.arxiv_id, "response": {"status_code": 0, "body": None}},
                {},
                [],
                7,
                "error",
                conversations,
                final_rows,
                {ctx.arxiv_id: 8},
                {ctx.arxiv_id: ctx},
                make_args(tmp_path),
            )
            # The episode is recorded as final so the loop's missing-response check passes.
            self.assertIn(ctx.arxiv_id, final_rows)
            self.assertEqual(final_rows[ctx.arxiv_id]["tool_loop"]["exit_reason"], "error")
            # report.json exists and is a degraded account carrying the exit reason.
            report = json.loads((ctx.run_dir / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["report_status"], "degraded")
            self.assertEqual(report["exit_reason"], "error")
            self.assertEqual(report["paper_id"], ctx.arxiv_id)

    def test_report_json_exists_before_the_final_event(self) -> None:
        # Ordering guard: the "final" event flips the run row terminal in the telemetry sink,
        # and the auditor's manifest expects report.json to already exist. finalize_episode
        # must persist report.json BEFORE emitting the final event.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ctx = make_ctx(tmp_path)
            seen: dict[str, bool] = {}

            def sink(kind: str, sink_ctx: ExecutionContext, payload: dict) -> None:
                if kind == "final":
                    seen["report_present_at_final"] = (sink_ctx.run_dir / "report.json").exists()

            live_log.register_sink(sink)
            good = {
                "paper_id": ctx.arxiv_id,
                "claim": "c",
                "what_ran": "w",
                "scoring_command": "python eval.py",
                "measurements": [],
                "agent_assessment": "reproduced",
                "changes_made": "",
                "blockers": "",
                "evidence_files": [],
            }
            finalize_episode(
                ctx.arxiv_id,
                {"custom_id": ctx.arxiv_id, "response": {"status_code": 200, "body": {}}},
                {"content": json.dumps(good)},
                [],
                12,
                "natural",
                {ctx.arxiv_id: []},
                {},
                {ctx.arxiv_id: 13},
                {ctx.arxiv_id: ctx},
                make_args(tmp_path),
            )
            self.assertTrue(seen.get("report_present_at_final"))
            report = json.loads((ctx.run_dir / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["report_status"], "complete")


if __name__ == "__main__":
    unittest.main()
