from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from reclaim_repro.context import Budget, ExecutionContext


class BudgetTests(unittest.TestCase):
    def test_remaining_and_exhausted(self):
        budget = Budget(total_h100_hours=2.0)
        self.assertEqual(budget.remaining(), 2.0)
        self.assertFalse(budget.exhausted())
        budget.consume(1.5)
        self.assertAlmostEqual(budget.remaining(), 0.5)
        self.assertFalse(budget.exhausted())
        budget.consume(0.5)
        self.assertTrue(budget.exhausted())

    def test_consume_ignores_negative(self):
        budget = Budget(total_h100_hours=1.0)
        budget.consume(-5.0)
        self.assertEqual(budget.remaining(), 1.0)

    def test_context_defaults(self):
        ctx = ExecutionContext(arxiv_id="2401.00001")
        self.assertEqual(ctx.lockfile_row, {})
        self.assertIsNone(ctx.budget)
        self.assertIsNone(ctx.workspace)
        self.assertIsNone(ctx.evidence)


if __name__ == "__main__":
    unittest.main()
