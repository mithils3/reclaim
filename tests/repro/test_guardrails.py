from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from reclaim_repro.context import ExecutionContext
from reclaim_repro.loop import _over


def _args(max_input_tokens: int = 100_000) -> argparse.Namespace:
    return argparse.Namespace(max_input_tokens=max_input_tokens)


class OverThresholdTests(unittest.TestCase):
    def test_fires_on_real_prompt_tokens(self):
        ctx = ExecutionContext(arxiv_id="x")
        ctx.last_prompt_tokens = 61_000
        # 61k of a 100k budget: over 0.6, under 0.7 — exact, no chars-per-token estimate.
        self.assertTrue(_over(ctx, _args(), 0.6))
        self.assertFalse(_over(ctx, _args(), 0.7))

    def test_ceiling_check(self):
        ctx = ExecutionContext(arxiv_id="x")
        ctx.last_prompt_tokens = 100_001
        self.assertTrue(_over(ctx, _args(), 1.0))

    def test_no_usage_yet_never_fires(self):
        ctx = ExecutionContext(arxiv_id="x")  # last_prompt_tokens defaults to None
        # No measurement yet → skip compaction, never estimate from chars.
        self.assertFalse(_over(ctx, _args(), 0.0))
        self.assertFalse(_over(ctx, _args(), 0.6))


if __name__ == "__main__":
    unittest.main()
