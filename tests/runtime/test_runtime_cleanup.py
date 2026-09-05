from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from reclaim_vllm.config.config import DEFAULT_MODEL, MINIMAX_M2_MODEL
from run_arxiv_prompt_vllm import parse_args


class RuntimeCleanupTests(unittest.TestCase):
    def test_audit_client_sampling_stays_unset(self) -> None:
        """The auditor is a URL-only client: parse_args leaves the request-time
        sampling fields unset so requests omit them and the served model's
        generation_config defaults apply (serve-launch flags live in
        reclaim_serve/profiles.py, not here)."""
        with patch.object(sys, "argv", ["run_arxiv_prompt_vllm.py"]):
            args = parse_args()

        self.assertEqual(DEFAULT_MODEL, MINIMAX_M2_MODEL)
        self.assertEqual(args.model, MINIMAX_M2_MODEL)
        self.assertEqual(args.max_model_len, 196608)
        self.assertIsNone(args.temperature)
        self.assertIsNone(args.top_p)
        self.assertIsNone(args.top_k)


if __name__ == "__main__":
    unittest.main()
