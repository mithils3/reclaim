from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from reclaim_vllm.runtime.run_health import degraded_row, loop_telemetry


class FinalizeRowTests(unittest.TestCase):
    def test_degraded_row_helper_keeps_raw_content(self) -> None:
        row = degraded_row("2501.00001", "garbage", None, {"exit_reason": "natural"})
        self.assertEqual(row["verification_status"], "degraded")
        self.assertIsNone(row["score"])
        self.assertEqual(row["raw_content"], "garbage")


class TelemetryTests(unittest.TestCase):
    def test_counts_tool_calls_and_errors(self) -> None:
        messages = [
            {"role": "tool", "content": '{"ok": true}'},
            {"role": "tool", "content": '{"ok": false, "error": "boom"}'},
            {"role": "assistant", "content": "x" * 400},
        ]
        telemetry = loop_telemetry(messages, max_input_tokens=100)
        self.assertEqual(telemetry["tool_calls"], 2)
        self.assertEqual(telemetry["tool_errors"], 1)
        self.assertTrue(telemetry["input_overflow"])

    def test_no_overflow_within_budget(self) -> None:
        telemetry = loop_telemetry([{"role": "user", "content": "hi"}], max_input_tokens=100)
        self.assertFalse(telemetry["input_overflow"])


if __name__ == "__main__":
    unittest.main()
