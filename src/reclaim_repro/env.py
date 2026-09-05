"""Tool-enforced execution environment for the reproduction agent.

The agent issues plain shell commands; the **tools** (never the model) decide where
each one runs. ``exec_argv`` is the single seam both sides use: the orchestrator tools
(``workspace_bash``, the venv build) pass it straight to ``subprocess.run``; the GPU
substrate (``slurm.py``) splices it in after ``srun``.

Every step body is just ``cd <ws> && <cmd>`` — the agent's cwd pinned to the episode's
workspace. There is no host ``module load`` and no CPU/GPU split in the body any more:
the CUDA toolkit and ``nvcc`` come from the **Apptainer image** the step runs inside
(``sandbox.py``), not from the bare host; ``torch`` is not prebuilt — the agent installs
an aarch64 CUDA build into its venv (see the prompt). ``on_gpu`` no longer
changes the body; it only tells the sandbox to add ``--nv`` so a GPU step sees the
device, while a CPU-setup step on the login node does not.

The sandbox is **mandatory** for every agent step (``sandbox`` is passed on each call);
the pure builders keep it optional so argv-shape tests can assert the bare body without
the container wrap.
"""

from __future__ import annotations

import shlex
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reclaim_repro.sandbox import Sandbox


def exec_argv(
    workspace,
    command: str,
    *,
    on_gpu: bool = False,
    sandbox: "Sandbox | None" = None,
) -> list[str]:
    """Argv that runs ``command`` in the episode's environment.

    Pass directly to ``subprocess.run`` for orchestrator-side tools, or splice in after
    ``srun ...`` for a GPU step. When ``sandbox`` is supplied (every agent step — the
    sandbox is mandatory) the body ``cd``s to the sandbox's short container ``workdir``
    and is wrapped in the episode's Apptainer container, so the step runs inside the
    read-only image and can only *write* the episode's own dirs. ``on_gpu`` adds ``--nv``
    for GPU steps. With no sandbox the body ``cd``s to the given host ``workspace``.
    """
    # The body carries no ``module load``: the container supplies the toolchain, and
    # inside the ``--cleanenv`` container ``module`` does not exist, so loading would
    # fail and short-circuit the real command. The ``cd`` is belt-and-suspenders
    # alongside the sandbox's ``--pwd``.
    workdir = sandbox.workdir if sandbox is not None else workspace
    body = f"cd {shlex.quote(str(workdir))} && {command}"
    if sandbox is not None:
        return sandbox.wrap_argv(body, nv=on_gpu)
    return ["bash", "-lc", body]
