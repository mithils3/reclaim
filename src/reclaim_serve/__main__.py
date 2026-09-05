"""Entry point: launch vLLM, wait for health, publish the URL, then block.

    python -m reclaim_serve --model /path/to/MiniMax-M2.7 --port 8000

On a head rank this prints the routable base URL and writes the endpoint file
once ``/health`` is green; it then waits on the server process and removes the
endpoint file on exit. Headless (non-head) ranks just exec vLLM and wait.
"""

from __future__ import annotations

import os
import signal
import sys
import types

from reclaim_serve import network
from reclaim_serve.args import parse_args
from reclaim_serve.endpoint import build_payload, remove_endpoint, write_endpoint
from reclaim_serve.launch import build_serve_command, start_process
from reclaim_serve.profiles import resolve_profile
from reclaim_serve.wait import wait_until_ready


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    profile = resolve_profile(args.model)
    command = build_serve_command(args, profile)
    process = start_process(command, _serve_env(args))

    def _forward(signum: int, _frame: types.FrameType | None) -> None:
        process.send_signal(signum)

    signal.signal(signal.SIGINT, _forward)
    signal.signal(signal.SIGTERM, _forward)

    if args.headless:
        # Non-head rank: no API to publish or probe; just shadow the process.
        return process.wait()

    published = False
    try:
        wait_until_ready(f"http://127.0.0.1:{args.port}/health", process)
        host = network.advertised_host(args.advertise_ip, args.iface)
        url = network.base_url(host, args.port)
        payload = build_payload(
            base_url=url,
            host=host,
            port=args.port,
            served_model_name=args.served_model_name,
            model_path=args.model,
            node=os.environ.get("SLURMD_NODENAME", ""),
            job_id=os.environ.get("SLURM_JOB_ID", ""),
        )
        write_endpoint(args.endpoint_file, payload)
        published = True
        _announce(url, args)
        return process.wait()
    finally:
        if published:
            remove_endpoint(args.endpoint_file)


def _serve_env(args) -> dict[str, str] | None:
    """Extra env for the vLLM subprocess, or None to inherit ours unchanged.

    Every serve: ``VLLM_USE_FLASHINFER_SAMPLER=0``. vLLM defaults that env to
    True on CUDA, and ``flashinfer_sampler_supported()`` imports the FlashInfer
    attention backend before it checks anything else, so a venv without
    ``flashinfer`` kills every worker in ``Sampler.__init__`` with
    ``ModuleNotFoundError`` and the engine never reaches ready. Installed, the
    kernel is the other half of the problem: it also crashes the profile run here
    (``TopKMaskLogits: invalid resource handle``). The native torch sampler is what
    every verified GH200 serve has run, so it is the launcher's default rather than
    an export each sbatch script has to remember. Export
    ``VLLM_USE_FLASHINFER_SAMPLER=1`` to opt back in on a box that fixed both.

    Multi-node only: pin ``VLLM_HOST_IP`` to this node's own fabric IP so vLLM's
    internal RPC / multiproc message queue lands on the same fabric as NCCL.
    Otherwise vLLM's ``get_ip()`` guesses the public VLAN, and remote workers
    hang dialing the engine message queue *after* NCCL init has completed. We
    also force ``TORCH_NCCL_ASYNC_ERROR_HANDLING`` so a future fabric mismatch
    fails with a stack instead of an indefinite hang. An explicit value set in
    the environment (e.g. by the launch runbook) always wins.
    """
    overrides: dict[str, str] = {
        "VLLM_USE_FLASHINFER_SAMPLER": os.environ.get("VLLM_USE_FLASHINFER_SAMPLER", "0")
    }
    if args.nnodes and args.nnodes > 1:
        if not os.environ.get("VLLM_HOST_IP"):
            ip = network.fabric_ipv4(args.iface)
            if ip:
                overrides["VLLM_HOST_IP"] = ip
        overrides.setdefault(
            "TORCH_NCCL_ASYNC_ERROR_HANDLING",
            os.environ.get("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1"),
        )
    env = dict(os.environ)
    env.update(overrides)
    print(
        "reclaim_serve: pinning vLLM subprocess env "
        + " ".join(f"{k}={v}" for k, v in overrides.items()),
        file=sys.stderr,
        flush=True,
    )
    return env


def _announce(url: str, args) -> None:
    line = "=" * 72
    print(
        f"\n{line}\n"
        f"vLLM server READY\n"
        f"  base url        : {url}\n"
        f"  served model    : {args.served_model_name}\n"
        f"  endpoint file   : {args.endpoint_file}\n"
        f"  attach a runner : --vllm-server-url {url} --model {args.served_model_name}\n"
        f"  or export       : export RECLAIM_SERVER_URL={url}\n"
        f"{line}\n",
        file=sys.stderr,
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
