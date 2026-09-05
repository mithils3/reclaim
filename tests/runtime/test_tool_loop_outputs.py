from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from reclaim_vllm.runtime.tool_loop import append_completed_outputs, prepare_incremental_outputs


class ToolLoopOutputTests(unittest.TestCase):
    def test_completed_outputs_are_appended_incrementally(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            args = argparse.Namespace(
                output=root / "responses.jsonl",
                extracted_output=root / "extracted.jsonl",
                save_round_jsonl=True,
                trace_output=root / "trace.jsonl",
                mode="audit",
            )
            prepare_incremental_outputs(args)
            row = final_row(
                json.dumps(
                    {
                        "paper_id": "2501.00001",
                        "execution_verified": True,
                        "score": 9,
                    }
                )
            )

            append_completed_outputs("2501.00001", row, [{"role": "user"}], args)

            self.assertEqual(len(read_jsonl(args.output)), 1)
            extracted = read_jsonl(args.extracted_output)
            self.assertEqual(extracted[0]["score"], 9)
            self.assertEqual(extracted[0]["verdict"], "reproduced")
            self.assertTrue(extracted[0]["reproduced"])
            self.assertEqual(extracted[0]["verification_status"], "verified")
            trace = read_jsonl(args.trace_output)
            self.assertEqual(trace[0]["custom_id"], "2501.00001")


def final_row(content: str) -> dict:
    return {
        "custom_id": "2501.00001",
        "response": {
            "body": {
                "choices": [
                    {
                        "message": {
                            "content": content,
                        }
                    }
                ]
            }
        },
        "tool_loop": {"tool_rounds_used": 1},
    }


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


if __name__ == "__main__":
    unittest.main()
