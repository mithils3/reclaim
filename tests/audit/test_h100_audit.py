from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from reclaim_vllm.audit.h100 import arithmetic_mismatch, h100_band, recomputed_hours


class BandTests(unittest.TestCase):
    def test_band_edges(self) -> None:
        self.assertEqual(h100_band(0), "0-8")
        self.assertEqual(h100_band(8), "0-8")
        self.assertEqual(h100_band(8.1), "8-32")
        self.assertEqual(h100_band(96), "32-96")
        self.assertEqual(h100_band(192), "96-192")
        self.assertEqual(h100_band(2_200_000), ">192")
        self.assertIsNone(h100_band(None))
        self.assertIsNone(h100_band("four"))


class ArithmeticTests(unittest.TestCase):
    def test_consistent_arithmetic_passes(self) -> None:
        estimate = {"gpu_count": 8, "wallclock_hours": 7, "h100_equivalent_multiplier": 0.32}
        recomputed = recomputed_hours(estimate)
        self.assertAlmostEqual(recomputed, 17.92)
        self.assertFalse(arithmetic_mismatch(17.92, recomputed))

    def test_mismatched_arithmetic_is_flagged(self) -> None:
        self.assertTrue(arithmetic_mismatch(100, 17.92))

    def test_missing_arithmetic_is_none(self) -> None:
        self.assertIsNone(recomputed_hours({"gpu_count": None, "wallclock_hours": None, "h100_equivalent_multiplier": 0.32}))
        self.assertIsNone(arithmetic_mismatch(17.92, None))


if __name__ == "__main__":
    unittest.main()
