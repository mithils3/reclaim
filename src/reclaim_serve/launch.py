"""Build and start the ``vllm serve`` subprocess.

The command is assembled from the model's serve profile and binds a routable
host/port (and, for multi-node, wires the rendezvous flags).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
from collections.abc import Callable
from typing import Any

from reclaim_serve.profiles import Profile


def build_serve_command(args: argparse.Namespace, profile: Profile) -> list[str]:
    tp = args.tensor_parallel_size or profile.tensor_parallel_size
    command = [
        args.vllm_bin,
        "serve",
        args.model,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--served-model-name",
        args.served_model_name,
        "--tensor-parallel-size",
        str(tp),
        "--max-model-len",
        str(args.max_model_len or profile.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization or profile.gpu_memory_utilization),
        "--tool-call-parser",
        args.tool_call_parser or profile.tool_call_parser,
        "--reasoning-parser",
        args.reasoning_parser or profile.reasoning_parser,
        "--enable-auto-tool-choice",
        # Without this, vLLM omits usage.prompt_tokens_details entirely, so every
        # response reports cached_tokens=0 even when prefix caching is hitting. The
        # KV reuse happens regardless (enable_prefix_caching defaults to True); this
        # flag is purely what makes the cache-hit count observable downstream.
        "--enable-prompt-tokens-details",
    ]
    if args.trust_remote_code or profile.trust_remote_code:
        command.append("--trust-remote-code")
    command.extend(_coalesced_flags(args, profile))
    if args.enable_expert_parallel or profile.enable_expert_parallel:
        command.append("--enable-expert-parallel")
    command.extend(_multinode_flags(args))
    command.extend(_supported_extra_args(args.extra_vllm_args))
    return command


def _supported_extra_args(extra: list[str]) -> list[str]:
    """Drop pass-through flags this GPU cannot run.

    One launch-time rewrite so far, the extra-args sibling of the ``pass_config``
    rewrites below:

    - ``--moe-backend deep_gemm_mega_moe`` is dropped below SM100. It is the
      vLLM recipe's Blackwell FP8 MoE kernel, and on a Hopper card vLLM raises
      ``NotImplementedError("DeepGEMM MegaMoE requires SM100 GPUs.")`` only
      AFTER loading all 48 checkpoint shards, so the serve dies ~3 minutes in
      and the allocation is spent. The Hopper port is
      an unmerged RFC (vllm#42284), blocked on an upstream DeepGEMM PR.
      Dropping the flag lands on vLLM's auto MoE-backend selection, which is
      what the official recipe does on Hopper and what every verified GH200
      serve of this model has run. Pass a non-mega backend explicitly to
      override; other ``--moe-backend`` values are left alone.
    """
    kept: list[str] = []
    index = 0
    while index < len(extra):
        token = extra[index]
        value = None
        width = 1
        if token == "--moe-backend" and index + 1 < len(extra):
            value = extra[index + 1]
            width = 2
        elif token.startswith("--moe-backend="):
            value = token.split("=", 1)[1]
        if value == "deep_gemm_mega_moe" and _below_sm100():
            print(
                "reclaim_serve: dropping --moe-backend deep_gemm_mega_moe "
                "(SM100-only; vLLM auto-selects the MoE backend on this GPU — "
                "pass a non-mega backend explicitly to override)",
                file=sys.stderr,
                flush=True,
            )
            index += width
            continue
        kept.append(token)
        index += 1
    return kept


def _below_sm100() -> bool:
    """True only when the GPU is determinably pre-Blackwell (compute major < 10).

    An undeterminable capability (no torch, no CUDA, a probe that raises) returns
    False: leave the operator's flag alone and let vLLM be the authority. torch is
    imported lazily so a command without a mega-MoE flag never pays for it.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        return torch.cuda.get_device_capability(0)[0] < 10
    except Exception:
        return False


def _supported_compilation_config(compilation: str) -> str:
    """Normalize ``pass_config`` against the installed vLLM.

    Two launch-time rewrites, both keyed off the installed ``PassConfig`` so
    they track the venv instead of being edited into every sbatch script:

    - Unknown keys are dropped. vLLM deletes a pass flag once the fusion
      becomes automatic (e.g. the MiniMax QK-norm pass), and an unknown key
      makes ``vllm serve`` exit 2 on a pydantic validation error before ready.
      Older builds still need the explicit flag for throughput.
    - ``fuse_allreduce_rms`` defaults to OFF. vLLM 0.25 enables the FlashInfer
      allreduce+RMSNorm fusion by default for TP compiles, and the mnnvl
      backend it auto-selects on GH200 hits a CUDA illegal memory access in the
      profile run, which dies before ready on a 2-GPU share. Set
      the key explicitly in the config to opt back in on a build that fixed it.
    """
    try:
        config = json.loads(compilation)
    except json.JSONDecodeError:
        return compilation
    if not isinstance(config, dict):
        return compilation
    known = _known_pass_config_fields()
    if known is None:
        return compilation
    raw_passes = config.get("pass_config")
    passes = dict(raw_passes) if isinstance(raw_passes, dict) else {}
    changed = False
    dropped = sorted(set(passes) - known)
    if dropped:
        passes = {key: value for key, value in passes.items() if key in known}
        changed = True
        print(
            "reclaim_serve: dropping pass_config keys this vLLM does not support: "
            + " ".join(dropped),
            file=sys.stderr,
            flush=True,
        )
    if "fuse_allreduce_rms" in known and "fuse_allreduce_rms" not in passes:
        passes["fuse_allreduce_rms"] = False
        changed = True
        print(
            "reclaim_serve: disabling fuse_allreduce_rms (FlashInfer mnnvl "
            "allreduce fusion IMAs on GH200; set it in --compilation-config to "
            "opt back in)",
            file=sys.stderr,
            flush=True,
        )
    if not changed:
        return compilation
    if passes:
        config["pass_config"] = passes
    else:
        config.pop("pass_config", None)
    return json.dumps(config, separators=(",", ":"))


def _known_pass_config_fields() -> set[str] | None:
    """Field names of the installed vLLM's PassConfig, or None if undeterminable."""
    try:
        from vllm.config import PassConfig
    except Exception:
        return None
    fields = getattr(PassConfig, "model_fields", None)
    if fields:
        return set(fields)
    try:
        return {field.name for field in dataclasses.fields(PassConfig)}
    except Exception:
        return None


@dataclasses.dataclass(frozen=True)
class _Flag:
    """A ``vllm serve`` flag whose value is the CLI arg falling back to the profile.

    ``arg`` and ``profile`` name the attribute on the argparse namespace and on the
    Profile; either is None for a flag only one side carries. ``render`` turns the
    coalesced value into its command-line token. ``keep_zero`` switches both the
    fallback and the emit test from truthiness to ``is not None``, so an explicit 0
    is passed through instead of being read as "unset".
    """

    flag: str
    arg: str | None = None
    profile: str | None = None
    render: Callable[[Any], str] = str
    keep_zero: bool = False


# ORDER IS LOAD-BEARING: this is the order the flags appear in the launched command,
# and the tests assert on it.
_COALESCED_FLAGS: tuple[_Flag, ...] = (
    _Flag("--mm-encoder-tp-mode", arg="mm_encoder_tp_mode", profile="mm_encoder_tp_mode"),
    _Flag(
        "--compilation-config",
        arg="compilation_config",
        profile="compilation_config",
        # Rewritten against the installed vLLM's PassConfig on the way out.
        render=_supported_compilation_config,
    ),
    # Server-side default for the chat template's render-time switches (thinking on/off,
    # effort ladders). Per-request RECLAIM_CHAT_TEMPLATE_KWARGS still wins where it is
    # set; this is the floor for the requests that omit them.
    _Flag(
        "--default-chat-template-kwargs",
        arg="default_chat_template_kwargs",
        profile="default_chat_template_kwargs",
    ),
    _Flag("--distributed-executor-backend", arg="distributed_executor_backend"),
    _Flag("--kv-cache-dtype", arg="kv_cache_dtype", profile="kv_cache_dtype"),
    _Flag("--block-size", arg="block_size", profile="block_size"),
    # 0 GiB is a real setting here (it disables the host offload tier), not "unset".
    _Flag("--swap-space", arg="swap_space", profile="swap_space_gb", keep_zero=True),
    _Flag("--max-num-seqs", profile="max_num_seqs"),
)


def _coalesced_flags(args: argparse.Namespace, profile: Profile) -> list[str]:
    """Emit every table flag the CLI or the profile sets, in table order."""
    tokens: list[str] = []
    for spec in _COALESCED_FLAGS:
        from_args = getattr(args, spec.arg) if spec.arg else None
        from_profile = getattr(profile, spec.profile) if spec.profile else None
        if spec.keep_zero:
            value = from_args if from_args is not None else from_profile
            present = value is not None
        else:
            value = from_args or from_profile
            present = bool(value)
        if present:
            tokens.extend([spec.flag, spec.render(value)])
    return tokens


def _multinode_flags(args: argparse.Namespace) -> list[str]:
    """Pipeline-parallel rendezvous flags, set only for a multi-node serve."""
    flags: list[str] = []
    if args.pipeline_parallel_size and args.pipeline_parallel_size > 1:
        flags.extend(["--pipeline-parallel-size", str(args.pipeline_parallel_size)])
    if args.nnodes and args.nnodes > 1:
        flags.extend(["--nnodes", str(args.nnodes)])
    if args.node_rank is not None:
        flags.extend(["--node-rank", str(args.node_rank)])
    if args.master_addr:
        flags.extend(["--master-addr", args.master_addr])
    if args.headless:
        flags.append("--headless")
    return flags


def start_process(
    command: list[str], env: dict[str, str] | None = None
) -> subprocess.Popen:
    """Launch vLLM, inheriting stdout/stderr so logs land in the Slurm log.

    ``env`` of None inherits this process's environment unchanged; a dict
    replaces it wholesale (callers pass a copy of ``os.environ`` plus overrides).
    """
    print("Starting vLLM server: " + " ".join(command), file=sys.stderr, flush=True)
    return subprocess.Popen(command, env=env)
