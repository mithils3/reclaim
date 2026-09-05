from __future__ import annotations

import json
import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from reclaim_vllm.runtime.loop_guards import (
    context_budget_exceeded,
    record_tool_call,
    repeated_tool_call,
)
from reclaim_vllm.tools.result_limits import is_transient_error, truncate_tool_result


def call(name: str = "fetch_url", **arguments) -> dict:
    return {"function": {"name": name, "arguments": json.dumps(arguments)}}


class RepeatPolicyTests(unittest.TestCase):
    def test_failed_calls_are_not_counted_as_repeats(self) -> None:
        counts: Counter = Counter()
        failed = call(url="https://x.test")
        record_tool_call(counts, failed, {"ok": False, "error": "timeout"})
        self.assertEqual(sum(counts.values()), 0)

        record_tool_call(counts, failed, {"ok": True})
        self.assertEqual(sum(counts.values()), 1)

    def test_repeated_successful_call_is_cut_off(self) -> None:
        counts = {"id": Counter()}
        the_call = call(url="https://x.test")
        record_tool_call(counts["id"], the_call, {"ok": True})
        self.assertIsNone(repeated_tool_call("id", [the_call], counts, max_repeats=2))
        record_tool_call(counts["id"], the_call, {"ok": True})
        self.assertIsNotNone(repeated_tool_call("id", [the_call], counts, max_repeats=2))


class ContextBudgetTests(unittest.TestCase):
    def test_budget_trips_before_token_limit(self) -> None:
        messages = [{"role": "tool", "content": "x" * 350}]
        self.assertTrue(context_budget_exceeded(messages, max_input_tokens=100))
        self.assertFalse(context_budget_exceeded(messages, max_input_tokens=200))
        self.assertFalse(context_budget_exceeded(messages, max_input_tokens=None))


class TruncationTests(unittest.TestCase):
    def test_small_results_pass_through_unchanged(self) -> None:
        result = {"ok": True, "text": "short"}
        self.assertEqual(truncate_tool_result(result, 1000), result)

    def test_huge_results_are_clamped_with_note(self) -> None:
        result = {"ok": True, "result": {"text": "x" * 500_000}, "other": "y" * 200_000}
        clamped = truncate_tool_result(result, 40_000)
        self.assertLessEqual(len(json.dumps(clamped, ensure_ascii=False)), 41_000)
        self.assertTrue(clamped["truncated"])
        self.assertIn("truncation_note", clamped)
        self.assertTrue(clamped["result"]["text"].endswith("…[truncated]"))

    def test_transient_errors_are_detected(self) -> None:
        self.assertTrue(is_transient_error({"ok": False, "error": "TimeoutError: timed out"}))
        self.assertTrue(is_transient_error({"ok": False, "error": "429 rate limit"}))
        self.assertFalse(is_transient_error({"ok": False, "error": "Unknown tool: bash"}))
        self.assertFalse(is_transient_error({"ok": True, "error": "timed out"}))


if __name__ == "__main__":
    unittest.main()
