from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from reclaim_serve import launch
from reclaim_serve.args import parse_args
from reclaim_serve.launch import build_serve_command
from reclaim_serve.profiles import Profile, resolve_profile


def command_for(argv: list[str]) -> list[str]:
    args = parse_args(argv)
    return build_serve_command(args, resolve_profile(args.model))


def value_after(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


class MinimaxServeCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cmd = command_for(["--model", "MiniMaxAI/MiniMax-M2.7"])

    def test_serves_the_model_on_all_interfaces(self) -> None:
        self.assertEqual(self.cmd[:3], ["vllm", "serve", "MiniMaxAI/MiniMax-M2.7"])
        self.assertEqual(value_after(self.cmd, "--host"), "0.0.0.0")
        self.assertEqual(value_after(self.cmd, "--port"), "8000")

    def test_uses_minimax_profile_flags(self) -> None:
        self.assertEqual(value_after(self.cmd, "--tensor-parallel-size"), "4")
        self.assertEqual(value_after(self.cmd, "--tool-call-parser"), "minimax_m2")
        self.assertEqual(value_after(self.cmd, "--reasoning-parser"), "minimax_m2")
        self.assertIn("--enable-auto-tool-choice", self.cmd)
        self.assertIn("--trust-remote-code", self.cmd)
        self.assertIn("cudagraph_mode", value_after(self.cmd, "--compilation-config"))

    def test_served_model_name_defaults_to_model(self) -> None:
        self.assertEqual(value_after(self.cmd, "--served-model-name"), "MiniMaxAI/MiniMax-M2.7")

    def test_served_model_name_override(self) -> None:
        cmd = command_for(["--model", "/local/path", "--served-model-name", "MiniMaxAI/MiniMax-M2.7"])
        self.assertEqual(value_after(cmd, "--served-model-name"), "MiniMaxAI/MiniMax-M2.7")


class Qwen3ServeCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cmd = command_for(["--model", "Qwen/Qwen3.6-27B-FP8"])

    def test_swap_space_defaults_to_vllm(self) -> None:
        self.assertNotIn("--swap-space", self.cmd)

    def test_swap_space_cli_override(self) -> None:
        cmd = command_for(["--model", "Qwen/Qwen3.6-27B-FP8", "--swap-space", "0"])
        self.assertEqual(value_after(cmd, "--swap-space"), "0.0")


class DeepseekV4ServeCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cmd = command_for(["--model", "deepseek-ai/DeepSeek-V4-Flash"])

    def test_uses_deepseek_profile_flags(self) -> None:
        self.assertEqual(value_after(self.cmd, "--tensor-parallel-size"), "2")
        self.assertEqual(value_after(self.cmd, "--tool-call-parser"), "deepseek_v4")
        self.assertEqual(value_after(self.cmd, "--reasoning-parser"), "deepseek_v4")
        self.assertEqual(value_after(self.cmd, "--kv-cache-dtype"), "fp8")
        # 400*1024, over the 393216 Think Max floor but well under the checkpoint's
        # 1M: the repro harness takes its input ceiling from whatever this serves, so
        # the floor is not the target, and 1M does not fit at TP=2.
        self.assertEqual(value_after(self.cmd, "--max-model-len"), "409600")
        self.assertIn("--enable-auto-tool-choice", self.cmd)

    def test_parser_override_is_the_sbatch_escape_hatch(self) -> None:
        # If a vLLM build names the V4 tool parser deepseek_v3, the sbatch overrides it.
        cmd = command_for(
            [
                "--model",
                "deepseek-ai/DeepSeek-V4-Flash",
                "--tool-call-parser",
                "deepseek_v3",
                "--reasoning-parser",
                "deepseek_v3",
            ]
        )
        self.assertEqual(value_after(cmd, "--tool-call-parser"), "deepseek_v3")
        self.assertEqual(value_after(cmd, "--reasoning-parser"), "deepseek_v3")


class ChatTemplateKwargsTests(unittest.TestCase):
    def test_template_kwargs_cli_override(self) -> None:
        cmd = command_for(
            [
                "--model",
                "MiniMaxAI/MiniMax-M2.7",
                "--default-chat-template-kwargs",
                '{"enable_thinking":false}',
            ]
        )
        self.assertEqual(
            value_after(cmd, "--default-chat-template-kwargs"), '{"enable_thinking":false}'
        )

    def test_profiles_without_the_pin_omit_the_flag(self) -> None:
        cmd = command_for(["--model", "Qwen/Qwen3.6-27B-FP8"])
        self.assertNotIn("--default-chat-template-kwargs", cmd)


class MultinodeServeCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cmd = command_for(
            [
                "--model",
                "MiniMaxAI/MiniMax-M2.7",
                "--tensor-parallel-size",
                "4",
                "--pipeline-parallel-size",
                "2",
                "--nnodes",
                "2",
                "--node-rank",
                "1",
                "--master-addr",
                "1.2.3.4",
                "--headless",
            ]
        )

    def test_wires_multinode_rendezvous(self) -> None:
        self.assertEqual(value_after(self.cmd, "--tensor-parallel-size"), "4")
        self.assertEqual(value_after(self.cmd, "--pipeline-parallel-size"), "2")
        self.assertEqual(value_after(self.cmd, "--nnodes"), "2")
        self.assertEqual(value_after(self.cmd, "--node-rank"), "1")
        self.assertEqual(value_after(self.cmd, "--master-addr"), "1.2.3.4")
        self.assertIn("--headless", self.cmd)


class CompilationConfigSanitizerTests(unittest.TestCase):
    """The minimax sbatch config must survive a vLLM that dropped the pass flag."""

    CONFIG = '{"mode":3,"pass_config":{"fuse_minimax_qk_norm":true}}'

    def config_given_fields(self, fields: set[str] | None) -> str:
        with mock.patch.object(launch, "_known_pass_config_fields", return_value=fields):
            cmd = command_for(["--model", "m", "--compilation-config", self.CONFIG])
        return value_after(cmd, "--compilation-config")

    def test_unknown_pass_key_is_dropped(self) -> None:
        value = self.config_given_fields({"enable_fusion", "enable_noop"})
        self.assertEqual(json.loads(value), {"mode": 3})

    def test_known_pass_key_is_kept(self) -> None:
        value = self.config_given_fields({"fuse_minimax_qk_norm"})
        self.assertEqual(json.loads(value), {"mode": 3, "pass_config": {"fuse_minimax_qk_norm": True}})

    def test_passes_through_when_vllm_is_unavailable(self) -> None:
        self.assertEqual(self.config_given_fields(None), self.CONFIG)


class AllreduceFusionDefaultOffTests(unittest.TestCase):
    """vLLM 0.25 default-enables fuse_allreduce_rms; its mnnvl backend IMAs on GH200."""

    def config_given(self, config: str, fields: set[str] | None) -> str:
        with mock.patch.object(launch, "_known_pass_config_fields", return_value=fields):
            cmd = command_for(["--model", "m", "--compilation-config", config])
        return value_after(cmd, "--compilation-config")

    def test_job_2667723_config_disables_the_fusion(self) -> None:
        # The repair sbatch's config on the upgraded venv: the stale QK-norm key
        # is dropped AND the crashing allreduce fusion is forced off.
        value = self.config_given(
            '{"mode":3,"pass_config":{"fuse_minimax_qk_norm":true}}',
            {"fuse_allreduce_rms", "fuse_norm_quant"},
        )
        self.assertEqual(
            json.loads(value), {"mode": 3, "pass_config": {"fuse_allreduce_rms": False}}
        )

    def test_injects_pass_config_when_absent(self) -> None:
        value = self.config_given('{"mode":3}', {"fuse_allreduce_rms"})
        self.assertEqual(
            json.loads(value), {"mode": 3, "pass_config": {"fuse_allreduce_rms": False}}
        )

    def test_explicit_opt_in_is_preserved(self) -> None:
        config = '{"mode":3,"pass_config":{"fuse_allreduce_rms":true}}'
        value = self.config_given(config, {"fuse_allreduce_rms"})
        self.assertEqual(value, config)

    def test_untouched_on_a_vllm_without_the_field(self) -> None:
        config = '{"mode":3,"pass_config":{"fuse_minimax_qk_norm":true}}'
        value = self.config_given(config, {"fuse_minimax_qk_norm"})
        self.assertEqual(value, config)


class MegaMoeExtraArgGuardTests(unittest.TestCase):
    """--moe-backend deep_gemm_mega_moe dies on GH200 after 48 shards have loaded."""

    def command_given_below_sm100(self, below: bool, extra: list[str]) -> list[str]:
        with mock.patch.object(launch, "_below_sm100", return_value=below):
            return command_for(["--model", "m", "--"] + extra)

    def test_mega_moe_is_dropped_below_sm100(self) -> None:
        cmd = self.command_given_below_sm100(
            True, ["--moe-backend", "deep_gemm_mega_moe", "--enable-expert-parallel"]
        )
        self.assertNotIn("--moe-backend", cmd)
        self.assertNotIn("deep_gemm_mega_moe", cmd)
        self.assertIn("--enable-expert-parallel", cmd)

    def test_equals_form_is_dropped(self) -> None:
        cmd = self.command_given_below_sm100(
            True, ["--moe-backend=deep_gemm_mega_moe", "--enable-expert-parallel"]
        )
        self.assertNotIn("--moe-backend=deep_gemm_mega_moe", cmd)
        self.assertIn("--enable-expert-parallel", cmd)

    def test_kept_on_sm100(self) -> None:
        cmd = self.command_given_below_sm100(
            False, ["--moe-backend", "deep_gemm_mega_moe", "--enable-expert-parallel"]
        )
        self.assertEqual(value_after(cmd, "--moe-backend"), "deep_gemm_mega_moe")
        self.assertIn("--enable-expert-parallel", cmd)

    def test_other_backends_pass_through_below_sm100(self) -> None:
        cmd = self.command_given_below_sm100(True, ["--moe-backend", "triton"])
        self.assertEqual(value_after(cmd, "--moe-backend"), "triton")

    def test_no_gpu_probe_without_moe_flag(self) -> None:
        # The capability probe imports torch, so it must not run for unrelated args.
        with mock.patch.object(launch, "_below_sm100", side_effect=AssertionError("probed")):
            cmd = command_for(["--model", "m", "--", "--swap-space", "16"])
        self.assertEqual(cmd[-2:], ["--swap-space", "16"])


class PassthroughTests(unittest.TestCase):
    def test_single_node_omits_multinode_flags(self) -> None:
        cmd = command_for(["--model", "MiniMaxAI/MiniMax-M2.7"])
        for flag in ("--pipeline-parallel-size", "--nnodes", "--node-rank", "--headless"):
            self.assertNotIn(flag, cmd)


class CoalescedFlagTests(unittest.TestCase):
    """The CLI-over-profile flag table: fallback, the 0 rule, and emission order."""

    def test_profile_swap_space_of_zero_still_emits_the_flag(self) -> None:
        # 0 GiB disables the host offload tier; it must not be read as "unset".
        profile = Profile(
            name="t", tensor_parallel_size=1, tool_call_parser="tc",
            reasoning_parser="rp", swap_space_gb=0,
        )
        cmd = build_serve_command(parse_args(["--model", "m"]), profile)
        self.assertEqual(value_after(cmd, "--swap-space"), "0")

    def test_cli_swap_space_of_zero_overrides_a_nonzero_profile(self) -> None:
        profile = Profile(
            name="t", tensor_parallel_size=1, tool_call_parser="tc",
            reasoning_parser="rp", swap_space_gb=16,
        )
        cmd = build_serve_command(parse_args(["--model", "m", "--swap-space", "0"]), profile)
        self.assertEqual(value_after(cmd, "--swap-space"), "0.0")

    def test_cli_value_wins_over_the_profile_default(self) -> None:
        profile = Profile(
            name="t", tensor_parallel_size=1, tool_call_parser="tc",
            reasoning_parser="rp", kv_cache_dtype="fp8", block_size=128,
            mm_encoder_tp_mode="data",
        )
        cmd = build_serve_command(
            parse_args(["--model", "m", "--kv-cache-dtype", "auto", "--block-size", "32"]),
            profile,
        )
        self.assertEqual(value_after(cmd, "--kv-cache-dtype"), "auto")
        self.assertEqual(value_after(cmd, "--block-size"), "32")
        self.assertEqual(value_after(cmd, "--mm-encoder-tp-mode"), "data")

    def test_max_num_seqs_comes_only_from_the_profile(self) -> None:
        profile = Profile(
            name="t", tensor_parallel_size=1, tool_call_parser="tc",
            reasoning_parser="rp", max_num_seqs=32,
        )
        cmd = build_serve_command(parse_args(["--model", "m"]), profile)
        self.assertEqual(value_after(cmd, "--max-num-seqs"), "32")

    def test_unset_options_emit_no_flag(self) -> None:
        profile = Profile(
            name="t", tensor_parallel_size=1, tool_call_parser="tc", reasoning_parser="rp"
        )
        cmd = build_serve_command(parse_args(["--model", "m"]), profile)
        for flag in (
            "--mm-encoder-tp-mode", "--compilation-config", "--default-chat-template-kwargs",
            "--distributed-executor-backend", "--kv-cache-dtype", "--block-size",
            "--swap-space", "--max-num-seqs",
        ):
            self.assertNotIn(flag, cmd)

    def test_flags_are_emitted_in_table_order(self) -> None:
        # vLLM does not care, but pinning the order keeps the table honest.
        profile = Profile(
            name="t", tensor_parallel_size=1, tool_call_parser="tc", reasoning_parser="rp",
            mm_encoder_tp_mode="data", compilation_config='{"cudagraph_mode":"PIECEWISE"}',
            default_chat_template_kwargs='{"enable_thinking":true}', kv_cache_dtype="fp8",
            block_size=128, max_num_seqs=32, swap_space_gb=8,
        )
        cmd = build_serve_command(
            parse_args(["--model", "m", "--distributed-executor-backend", "mp"]), profile
        )
        expected = [
            "--mm-encoder-tp-mode", "--compilation-config", "--default-chat-template-kwargs",
            "--distributed-executor-backend", "--kv-cache-dtype", "--block-size",
            "--swap-space", "--max-num-seqs",
        ]
        self.assertEqual([token for token in cmd if token in expected], expected)


if __name__ == "__main__":
    unittest.main()
