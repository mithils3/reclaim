from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from reclaim_serve import __main__ as serve_main
from reclaim_serve.args import parse_args


def env_for(argv: list[str], environ: dict[str, str], fabric_ip: str | None):
    args = parse_args(argv)
    with mock.patch.dict(serve_main.os.environ, environ, clear=True), mock.patch.object(
        serve_main.network, "fabric_ipv4", return_value=fabric_ip
    ):
        return serve_main._serve_env(args)


SINGLE = ["--model", "MiniMaxAI/MiniMax-M2.7"]
MULTI = ["--model", "m", "--nnodes", "2", "--node-rank", "1", "--master-addr", "1.2.3.4"]


class ServeEnvTests(unittest.TestCase):
    def test_single_node_disables_flashinfer_sampler(self) -> None:
        env = env_for(SINGLE, {}, "192.0.2.20")
        self.assertEqual(env["VLLM_USE_FLASHINFER_SAMPLER"], "0")
        self.assertNotIn("VLLM_HOST_IP", env)

    def test_explicit_flashinfer_sampler_wins(self) -> None:
        env = env_for(SINGLE, {"VLLM_USE_FLASHINFER_SAMPLER": "1"}, None)
        self.assertEqual(env["VLLM_USE_FLASHINFER_SAMPLER"], "1")

    def test_multinode_pins_fabric_ip_and_async_handling(self) -> None:
        env = env_for(MULTI, {}, "192.0.2.20")
        self.assertEqual(env["VLLM_HOST_IP"], "192.0.2.20")
        self.assertEqual(env["TORCH_NCCL_ASYNC_ERROR_HANDLING"], "1")
        self.assertEqual(env["VLLM_USE_FLASHINFER_SAMPLER"], "0")

    def test_explicit_host_ip_wins(self) -> None:
        env = env_for(MULTI, {"VLLM_HOST_IP": "10.0.0.9"}, "192.0.2.20")
        self.assertEqual(env["VLLM_HOST_IP"], "10.0.0.9")

    def test_explicit_async_handling_preserved(self) -> None:
        env = env_for(MULTI, {"TORCH_NCCL_ASYNC_ERROR_HANDLING": "0"}, "192.0.2.20")
        self.assertEqual(env["TORCH_NCCL_ASYNC_ERROR_HANDLING"], "0")

    def test_no_fabric_ip_still_sets_async_handling(self) -> None:
        env = env_for(MULTI, {}, None)
        self.assertNotIn("VLLM_HOST_IP", env)
        self.assertEqual(env["TORCH_NCCL_ASYNC_ERROR_HANDLING"], "1")


if __name__ == "__main__":
    unittest.main()
