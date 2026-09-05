"""Argparse smoke tests for the three CLI surfaces the live sweep drives directly.

Mirrors the real invocations in:
  - scripts/reproduce/repro_audit_one.sh                (reproduce stage + audit stage)
  - scripts/reproduce/minimax_m2/easy_minimax_m2.sbatch   (same two, sweep-parameterized)
  - scripts/serve/serve_gh200.sbatch                     (reclaim_serve, plus its own
    inlined `python -m reclaim_serve` call in the sweep sbatch)

This file is the parse-level safety net: a change that accidentally breaks one of
the real invocations shows up as a test failure here instead of a multi-day SLURM
job dying at argv[0]. Every argv list below is lifted from a real script line
above, with only inline dummy values substituted for paths, URLs and ids.

Offline: no network, no HF downloads. The only seam needed is patching
sys.argv for reclaim_vllm's parse_args(), which reads it directly instead of
accepting an argv parameter.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from reclaim_repro.cli_args import parse_args as parse_repro_args
from reclaim_repro.cluster import DEFAULT_CLUSTER
from reclaim_serve.args import parse_args as parse_serve_args
from reclaim_vllm.config.cli_args import parse_args as parse_runner_args
from reclaim_vllm.schema.audit import AUDIT_RESPONSE_FORMAT
from reclaim_vllm.tools.run_dir_tools import AUDIT_TOOLS


class ReclaimCliSmokeTests(unittest.TestCase):
    """argv mirrors the reproduce stage of repro_audit_one.sh / the easy sbatch."""

    def test_repro_audit_one_reproduce_stage_parses(self):
        argv = [
            "--paper-id", "2505.18513",
            "--split", "dev",
            "--run-id", "20260701T000000Z-abc123",
            "--vllm-server-url", "http://localhost:8000/v1",
            "--served-model-name", "deepseek/deepseek-v4-pro",
        ]
        args = parse_repro_args(argv)
        self.assertEqual(args.paper_id, "2505.18513")
        self.assertEqual(args.split, "dev")
        self.assertEqual(args.cluster_profile.name, DEFAULT_CLUSTER)
        self.assertTrue(args.tools)  # build_repro_tools advertised toolset
        self.assertIsNotNone(args.response_format)

    def test_easy_sweep_reproduce_stage_parses(self):
        argv = [
            "--paper-id", "2505.18513",
            "--split", "eval",
            "--lockfile", "data/splits/eval_100.jsonl",
            "--run-id", "12345-2505.18513-abc123",
            "--tool-rounds", "300",
            "--served-model-name", "MiniMaxAI/MiniMax-M2.7",
            "--output", "/tmp/reproduce_2505.18513.jsonl",
            "--save-round-jsonl",
        ]
        args = parse_repro_args(argv)
        self.assertEqual(args.lockfile, "data/splits/eval_100.jsonl")
        self.assertEqual(args.tool_rounds, 300)
        self.assertEqual(args.cluster_profile.gpus_per_node, 4)
        self.assertTrue(args.save_round_jsonl)


class AuditRunnerCliSmokeTests(unittest.TestCase):
    """argv mirrors run_arxiv_prompt_vllm.py --mode audit in both drivers.

    parse_args() reads sys.argv directly (no argv parameter), so sys.argv is
    patched for the duration of the call -- the only seam these two scripts need.
    """

    def _parse(self, argv):
        with mock.patch.object(sys, "argv", ["run_arxiv_prompt_vllm.py", *argv]):
            return parse_runner_args()

    def test_repro_audit_one_audit_stage_parses(self):
        argv = [
            "--mode", "audit",
            "--vllm-server-url", "http://localhost:8000/v1",
            "--served-model-name", "deepseek/deepseek-v4-pro",
            "--model", "deepseek/deepseek-v4-pro",
            "--runs-dir", "/tmp/grade_root",
            "--claims", "data/splits/dev_14.jsonl",
            "--paper-ids-file", "/tmp/audit_ids.txt",
            "--tool-rounds", "25",
            "--output", "/tmp/audit_2505.18513.jsonl",
            "--extracted-output", "/tmp/audit_2505.18513_extracted.jsonl",
            "--trace-output", "/tmp/audit_2505.18513_trace.jsonl",
            "--save-round-jsonl",
        ]
        args = self._parse(argv)
        self.assertEqual(args.mode, "audit")
        self.assertEqual(args.response_format, AUDIT_RESPONSE_FORMAT)
        self.assertEqual(args.tools, AUDIT_TOOLS)
        self.assertTrue(args.use_tools)

    def test_easy_sweep_audit_stage_parses(self):
        argv = [
            "--mode", "audit",
            "--vllm-server-url", "http://localhost:8000/v1",
            "--served-model-name", "MiniMaxAI/MiniMax-M2.7",
            "--model", "MiniMaxAI/MiniMax-M2.7",
            "--runs-dir", "/tmp/grade_2505.18513",
            "--claims", "data/splits/eval_100.jsonl",
            "--paper-ids-file", "/tmp/ids.txt",
            "--tool-rounds", "40",
            "--max-input-tokens", "128000",
            "--max-tokens", "8192",
            "--max-model-len", "196608",
            "--output", "/tmp/audit_2505.18513.jsonl",
            "--extracted-output", "/tmp/audit_2505.18513_extracted.jsonl",
            "--trace-output", "/tmp/audit_2505.18513_trace.jsonl",
            "--save-round-jsonl",
        ]
        args = self._parse(argv)
        self.assertEqual(args.max_input_tokens, 128000)
        self.assertEqual(args.max_model_len, 196608)
        self.assertEqual(args.runs_dir, Path("/tmp/grade_2505.18513"))


class ServeCliSmokeTests(unittest.TestCase):
    """argv mirrors scripts/serve/serve_gh200.sbatch and the easy sweep's serve step."""

    def test_serve_gh200_invocation_parses(self):
        argv = [
            "--model", "MiniMaxAI/MiniMax-M2.7",
            "--served-model-name", "MiniMaxAI/MiniMax-M2.7",
            "--port", "8000",
            "--tensor-parallel-size", "4",
            "--endpoint-file", "/tmp/endpoints/minimax_m2.json",
        ]
        args = parse_serve_args(argv)
        self.assertEqual(args.served_model_name, "MiniMaxAI/MiniMax-M2.7")
        self.assertEqual(args.port, 8000)
        self.assertEqual(args.tensor_parallel_size, 4)
        self.assertEqual(args.endpoint_file, Path("/tmp/endpoints/minimax_m2.json"))

    def test_easy_sweep_serve_invocation_parses(self):
        argv = [
            "--model", "MiniMaxAI/MiniMax-M2.7",
            "--served-model-name", "MiniMaxAI/MiniMax-M2.7",
            "--port", "8000",
            "--tensor-parallel-size", "4",
            "--max-model-len", "196608",
            "--distributed-executor-backend", "mp",
            "--compilation-config", '{"mode":3,"pass_config":{"fuse_minimax_qk_norm":true}}',
            "--endpoint-file", "/tmp/sweeps/1/endpoint.json",
        ]
        args = parse_serve_args(argv)
        self.assertEqual(args.max_model_len, 196608)
        self.assertEqual(args.distributed_executor_backend, "mp")
        self.assertIn("fuse_minimax_qk_norm", args.compilation_config)


if __name__ == "__main__":
    unittest.main()
