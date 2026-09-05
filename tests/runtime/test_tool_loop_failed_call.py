from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from concurrent.futures import Future
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from reclaim_vllm.runtime.tool_loop import (
    finalize_failed_request,
    handle_request_done,
    prepare_incremental_outputs,
)


def audit_args(root: Path) -> argparse.Namespace:
    return argparse.Namespace(
        output=root / "responses.jsonl",
        extracted_output=root / "extracted.jsonl",
        save_round_jsonl=False,
        trace_output=root / "trace.jsonl",
        mode="audit",
        tool_rounds=40,
        max_input_tokens=None,
        max_repeated_tool_calls=5,
        final_no_tools_message="wrap up",
    )


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class FailedModelCallTests(unittest.TestCase):
    """A dead model call must terminate one paper, never the whole batch."""

    def test_finalize_writes_a_degraded_verdict_and_a_final_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            args = audit_args(root)
            prepare_incremental_outputs(args)
            final_rows: dict[str, dict] = {}
            exit_reasons: dict[str, str] = {}
            conversations = {"2501.00001": [{"role": "user", "content": "grade it"}]}

            finalize_failed_request(
                "2501.00001",
                11,
                RuntimeError("HTTP Error 400: unknown parameter `provider`"),
                conversations,
                final_rows,
                {"2501.00001": 11},
                exit_reasons,
                args,
            )

            # Landing in final_rows is what stops run_tool_loop's `missing` check
            # from turning one paper's failure into a SystemExit for the batch.
            self.assertIn("2501.00001", final_rows)
            self.assertEqual(exit_reasons["2501.00001"], "error")
            row = final_rows["2501.00001"]
            self.assertEqual(row["response"]["status_code"], 0)
            self.assertIn("provider", row["response"]["error"])
            self.assertEqual(row["tool_loop"]["exit_reason"], "error")
            self.assertEqual(row["tool_loop"]["tool_rounds_used"], 11)

            verdicts = read_jsonl(args.extracted_output)
            self.assertEqual(len(verdicts), 1)
            self.assertEqual(verdicts[0]["custom_id"], "2501.00001")
            self.assertEqual(verdicts[0]["verification_status"], "degraded")
            self.assertIsNone(verdicts[0]["score"])
            self.assertEqual(len(read_jsonl(args.output)), 1)

    def test_handle_request_done_catches_a_raising_future(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            args = audit_args(root)
            prepare_incremental_outputs(args)
            future: Future = Future()
            future.set_exception(RuntimeError("HTTP Error 400: Bad Request"))
            request_futures = {future: {"custom_id": "2501.00002", "round_index": 3,
                                        "include_tools": False}}
            final_rows: dict[str, dict] = {}
            conversations = {"2501.00002": [{"role": "user", "content": "grade it"}]}

            handle_request_done(
                future, request_futures, {}, None, conversations, final_rows,
                {"2501.00002": 3}, {}, {}, {}, args,
            )

            self.assertEqual(request_futures, {})
            self.assertEqual(final_rows["2501.00002"]["tool_loop"]["exit_reason"], "error")
            self.assertEqual(read_jsonl(args.extracted_output)[0]["verification_status"],
                             "degraded")


if __name__ == "__main__":
    unittest.main()
