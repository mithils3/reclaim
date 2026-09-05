from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from reclaim_repro.compact import ELIDE_MIN_CHARS, choose_cut, elide_compact


def _bash_call() -> dict:
    return {"id": "c2", "function": {"name": "workspace_bash", "arguments": json.dumps({"command": "ls"})}}


def _tool(content: str, call_id: str = "c1") -> dict:
    return {"role": "tool", "tool_call_id": call_id, "name": "t", "content": content}


def _conversation() -> list[dict]:
    # system, pinned task prompt, a bulky-output turn, a small-output turn, final assistant.
    return [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "TASK: reproduce FID=90.68 ±3 anchor"},
        {"role": "assistant", "content": "installing", "tool_calls": [_bash_call()]},
        _tool("LOG" * 5000, "c2"),  # ~15k chars of stdout — the bulk to elide
        {"role": "assistant", "content": "checking", "tool_calls": [_bash_call()]},
        _tool("ok", "c2"),  # tiny result — stays verbatim
        {"role": "assistant", "content": "done"},
    ]


class ChooseCutTests(unittest.TestCase):
    def test_noop_when_everything_fits_in_keep_window(self):
        messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "hi"}]
        self.assertLessEqual(choose_cut(messages, keep_recent_tokens=10_000), 1)

    def test_walks_back_from_newest(self):
        messages = _conversation()
        # Tiny keep window → the cut lands well before the end so there is an old span.
        cut = choose_cut(messages, keep_recent_tokens=10)
        self.assertGreater(cut, 1)
        self.assertLess(cut, len(messages))


class ElideCompactTests(unittest.TestCase):
    def test_elides_bulky_tool_output_and_pins_head(self):
        messages = _conversation()
        original = [dict(m) for m in messages]
        stats = elide_compact(messages, keep_recent_tokens=10, full_log_path="/run/agent.full.log")
        self.assertTrue(stats["compacted"])
        self.assertEqual(stats["elided_messages"], 1)
        # System + pinned task prompt untouched.
        self.assertEqual(messages[0], original[0])
        self.assertEqual(messages[1], original[1])
        # The bulky tool result is now a pointer, not the 15k-char blob.
        self.assertTrue(messages[3]["content"].startswith("[elided "))
        self.assertIn("/run/agent.full.log", messages[3]["content"])
        self.assertNotIn("LOG" * 5000, messages[3]["content"])
        # Structure preserved: still a tool message with its id, just shorter content.
        self.assertEqual(messages[3]["role"], "tool")
        self.assertEqual(messages[3]["tool_call_id"], "c2")
        self.assertLess(stats["chars_after"], stats["chars_before"])

    def test_keeps_assistant_reasoning_and_tool_calls_verbatim(self):
        messages = _conversation()
        elide_compact(messages, keep_recent_tokens=10)
        # Every assistant turn (content + tool_calls) survives the compaction unchanged —
        # this is what keeps the agent's stated intent alive across the boundary.
        self.assertEqual(messages[2]["content"], "installing")
        self.assertEqual(messages[2]["tool_calls"][0]["id"], "c2")

    def test_small_tool_results_stay_verbatim(self):
        messages = _conversation()
        elide_compact(messages, keep_recent_tokens=10)
        # The "ok" result is under ELIDE_MIN_CHARS — reading it beats a disk round-trip.
        small = [m for m in messages if m.get("role") == "tool" and m.get("content") == "ok"]
        self.assertEqual(len(small), 1)

    def test_default_pointer_when_no_log_path(self):
        messages = _conversation()
        elide_compact(messages, keep_recent_tokens=10)
        self.assertIn("evidence/", messages[3]["content"])

    def test_noop_when_nothing_old_enough(self):
        messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "hi"}]
        stats = elide_compact(messages, keep_recent_tokens=10_000)
        self.assertFalse(stats["compacted"])
        self.assertEqual(stats["reason"], "nothing-old-enough")

    def test_noop_when_no_bulky_output(self):
        # Old span exists but every tool result is small — nothing worth eliding.
        messages = [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "TASK"},
            {"role": "assistant", "content": "a", "tool_calls": [_bash_call()]},
            _tool("ok", "c2"),
            {"role": "assistant", "content": "b", "tool_calls": [_bash_call()]},
            _tool("also fine", "c2"),
            {"role": "assistant", "content": "c"},
        ]
        snapshot = [dict(m) for m in messages]
        stats = elide_compact(messages, keep_recent_tokens=10)
        self.assertFalse(stats["compacted"])
        self.assertEqual(stats["reason"], "no-bulk-tool-output-or-reasoning")
        self.assertEqual(messages, snapshot)

    def test_idempotent_across_repeated_passes(self):
        messages = _conversation()
        elide_compact(messages, keep_recent_tokens=10)
        after_first = [dict(m) for m in messages]
        stats = elide_compact(messages, keep_recent_tokens=10)
        # Nothing bulky left to elide (the placeholder is under ELIDE_MIN_CHARS).
        self.assertFalse(stats["compacted"])
        self.assertEqual(messages, after_first)
        self.assertLess(len(messages[3]["content"]), ELIDE_MIN_CHARS)


class DropOldReasoningTests(unittest.TestCase):
    """Old CoT is the un-elidable floor that killed the majority of one sweep's runs."""

    def _convo(self) -> list[dict]:
        return [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "TASK"},
            {"role": "assistant", "content": "old", "reasoning": "R" * 5000,
             "tool_calls": [_bash_call()]},
            _tool("x" * 4000, "c2"),
            {"role": "assistant", "content": "recent", "reasoning": "K" * 200},
        ]

    def test_drops_reasoning_from_the_old_span(self) -> None:
        messages = self._convo()
        stats = elide_compact(messages, keep_recent_tokens=10)
        self.assertTrue(stats["compacted"])
        self.assertEqual(stats["dropped_reasoning"], 1)
        self.assertNotIn("reasoning", messages[2])

    def test_keeps_content_and_tool_calls_of_the_stripped_turn(self) -> None:
        # Losing *how* it reasoned must not lose *what* it concluded or did.
        messages = self._convo()
        elide_compact(messages, keep_recent_tokens=10)
        self.assertEqual(messages[2]["content"], "old")
        self.assertEqual(len(messages[2]["tool_calls"]), 1)

    def test_never_touches_the_recent_tail(self) -> None:
        messages = self._convo()
        elide_compact(messages, keep_recent_tokens=10)
        self.assertEqual(messages[-1]["reasoning"], "K" * 200)

    def test_never_touches_the_pinned_head(self) -> None:
        messages = self._convo()
        messages[1]["reasoning"] = "PINNED"
        elide_compact(messages, keep_recent_tokens=10)
        self.assertEqual(messages[1]["reasoning"], "PINNED")

    def test_compacts_on_reasoning_alone_when_no_bulky_stdout(self) -> None:
        # A reasoning-heavy run with small tool output still has space to reclaim.
        # This is the DeepSeek-V4 shape: the CoT is the bulk, the stdout is not.
        messages = [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "TASK"},
            {"role": "assistant", "content": "a", "reasoning": "R" * 9000},
            _tool("ok", "c2"),
            {"role": "assistant", "content": "b", "reasoning": "R" * 9000},
            _tool("ok", "c2"),
            {"role": "assistant", "content": "c"},
        ]
        stats = elide_compact(messages, keep_recent_tokens=10)
        self.assertTrue(stats["compacted"])
        self.assertEqual(stats["elided_messages"], 0)
        self.assertGreaterEqual(stats["dropped_reasoning"], 1)
        self.assertLess(stats["chars_after"], stats["chars_before"])


if __name__ == "__main__":
    unittest.main()
