"""The cluster profile: the operator-set substrate for the held GPU allocation.

The reproduction agent holds **one** SLURM allocation for the episode: the first
``run_gpu`` call acquires it and every later step runs into the same held node (see
``slurm.py`` / ``gpu_session.py``). A ``Cluster`` carries the facts an operator sets
*once* and the agent is merely *entitled to* (account, partition, node type ``hw``)
plus the Apptainer image a GPU step runs inside. The model never picks these; on the
call that starts the session it picks only ``gpus``/``minutes`` (the hold's size and
hard lifetime, billed as wall clock in ``budget.py``).

There is one built-in profile, describing a 4-GPU-per-node GH200 partition behind
SLURM. Its site values are read from the environment so the code carries no site's
spelling:

===========================  ==============================  =========================
Variable                     Meaning                         Default
===========================  ==============================  =========================
``RECLAIM_SLURM_ACCOUNT``    ``salloc -A``                    ``gpu-account``
``RECLAIM_SLURM_PARTITION``  ``salloc -p`` (batch pool)       ``gpu``
``RECLAIM_GPUS_PER_NODE``    cap on one step's ``--gpus``     ``4``
``RECLAIM_APPTAINER_IMAGE``  sandbox ``.sif`` every step runs ``/opt/images/cuda-agent.sif``
===========================  ==============================  =========================

The two per-run overrides are ``--partition`` (pick a different node pool) and
``--apptainer-image`` (swap the sandbox .sif); everything else is pinned here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Any

DEFAULT_CLUSTER = "default"

ENV_ACCOUNT = "RECLAIM_SLURM_ACCOUNT"
ENV_PARTITION = "RECLAIM_SLURM_PARTITION"
ENV_GPUS_PER_NODE = "RECLAIM_GPUS_PER_NODE"
ENV_APPTAINER_IMAGE = "RECLAIM_APPTAINER_IMAGE"

# Sites name their pools differently, so these are roles rather than spellings.
GPU_ACCOUNT = os.environ.get(ENV_ACCOUNT) or "gpu-account"
GPU_PARTITION = os.environ.get(ENV_PARTITION) or "gpu"
GPUS_PER_NODE = int(os.environ.get(ENV_GPUS_PER_NODE) or 4)

# The sandbox image. A raw NVIDIA CUDA image (12.9 + cuDNN, the ``devel`` flavor so
# ``nvcc`` and the host compilers are present) works better here than an NGC PyTorch
# image: torch is NOT prebuilt, so the agent installs the matched torch family itself
# from the aarch64 CUDA wheel index (see prompts/prompt_reproduce.txt). That sidesteps
# the NGC ``+nv`` torch ABI wall, since no stock ``torchaudio``/``torchvision`` wheel
# matches an NGC ``+nv`` torch and NGC PyTorch images ship no torchaudio at all. The
# image is the CUDA base with the agent's CLI tools (git/curl/build tools/ffmpeg)
# layered in, because a bare CUDA image ships none of them and a host ``module load``
# cannot reach inside the ``--cleanenv`` sandbox. Every agent step runs inside this
# read-only container (see sandbox.py); swap it per run with --apptainer-image or
# $REPRO_APPTAINER_SIF.
DEFAULT_APPTAINER_SIF = os.environ.get(ENV_APPTAINER_IMAGE) or "/opt/images/cuda-agent.sif"


@dataclass(frozen=True)
class Cluster:
    """Allocation-time facts plus the GPU-step environment."""

    name: str
    hw: str                              # budget multiplier key (see budget.HW_MULTIPLIER)
    gpus_per_node: int                   # upper bound on a single step's --gpus
    account: str | None = None           # salloc -A
    partition: str | None = None         # salloc -p
    apptainer_image: str | None = None   # MANDATORY sandbox .sif every step runs inside (sandbox.py)
    sandbox_cpus: int | None = None      # cores per agent CPU step on the shared orchestrator node; None = uncapped
    sandbox_mem_gb: int | None = None    # per-process address-space cap (GiB) for agent CPU steps; None = uncapped


# The single built-in profile. GPU steps always run through a JIT salloc, so
# account/partition are mandatory. Every step runs inside the mandatory Apptainer
# sandbox (sandbox.py): the CUDA .sif is the read-only root, so the agent's CUDA
# toolchain (``nvcc``, cuDNN, the CUDA libraries) comes from the image. There is no
# host ``module load``, which would not exist inside the --cleanenv container. torch
# is not in the image; the agent installs a GH200 (aarch64) CUDA torch from the
# PyTorch wheel index as its first setup step.
_PROFILES: dict[str, Cluster] = {
    DEFAULT_CLUSTER: Cluster(
        name=DEFAULT_CLUSTER,
        hw="gh200",
        gpus_per_node=GPUS_PER_NODE,
        account=GPU_ACCOUNT,
        # The batch pool is the default. A site usually also runs a faster-queueing
        # short/interactive pool; the agent discovers those through the
        # ``list_partitions`` tool and selects one per step by passing ``partition``
        # to ``run_gpu``. The profile pins only the *default*.
        partition=GPU_PARTITION,
        apptainer_image=DEFAULT_APPTAINER_SIF,
        # Per-paper share of the serving node's request. On the deployment we ran
        # (72 cores and 110 GB per GPU), 12 cores and 16 GB per agent CPU step let
        # six agents plus the vLLM server share the cgroup. Enforcement is in
        # sandbox.py, where the build fan-out is capped.
        sandbox_cpus=12,
        sandbox_mem_gb=16,
    ),
}


def cluster_defaults() -> dict[str, dict[str, Any]]:
    """The built-in default substrate per known cluster (source for ``list_partitions``).

    One source of truth for "the default choice per known cluster": each entry
    is the account / default partition / node size / hw the profile pins, which the
    ``list_partitions`` tool surfaces alongside the live ``sinfo`` partition list.
    """
    return {
        name: {
            "account": c.account,
            "default_partition": c.partition,
            "gpus_per_node": c.gpus_per_node,
            "hw": c.hw,
        }
        for name, c in _PROFILES.items()
    }


def resolve_cluster(
    name: str = DEFAULT_CLUSTER,
    *,
    partition: str | None = None,
    apptainer_image: str | None = None,
) -> Cluster:
    """The profile with the two per-run overrides (partition / image) applied."""
    try:
        base = _PROFILES[name]
    except KeyError:
        raise SystemExit(f"unknown cluster {name!r}; choose from {', '.join(_PROFILES)}")
    return replace(
        base,
        partition=partition if partition is not None else base.partition,
        apptainer_image=apptainer_image if apptainer_image is not None else base.apptainer_image,
    )
