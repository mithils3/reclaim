"""Filesystem sandbox for the reproduction agent's shell steps (Apptainer).

The agent runs *arbitrary* shell — it ``git clone``s an unknown repo and executes
its code (``workspace_bash`` on the orchestrator, ``run_gpu`` on a GPU node). The
file tools (``tools/files.py``) confine their *path arguments*, but a shell command
can write anywhere the user can, so confinement has to happen at the OS level rather
than by inspecting command strings. This module wraps each step in an **Apptainer**
container, the runtime HPC sites usually support where bubblewrap and Docker are
unavailable, so the step runs inside a read-only image and can only *write* the
episode's own dirs.

Apptainer confines differently from bubblewrap: the container **image** is the
read-only root (``/``), so the agent's CUDA toolchain (``nvcc``; ``torch`` is NOT
prebuilt, the agent installs it) comes from the image, a raw NVIDIA CUDA ``.sif``, and
host ``module load`` is gone. Only bound paths are visible at all:

* **read-write** binds over the episode's ``workspace``/``evidence``, the node-local
  ``/tmp`` (the agent's bulk scratch — a real bind, **not** a tmpfs, so weights and
  datasets land on the node's fast local disk), and the shared package caches
  (``~/.cache`` incl. HF/uv/pip) so a download or wheel build is reused across papers;
* a **read-only** bind over the episode's ``reference`` (the paper copy);
* everything else on the host — ``$HOME`` and the secrets ``~/.bashrc`` exports, other
  ``/projects``/``/work`` paths — is simply *not mounted*, so it can be neither read
  nor written.

``--cleanenv`` stops the orchestrator's ``LD_LIBRARY_PATH``/``PYTHONPATH`` leaking into
the container (host libs would shadow the image's own and crash ``git``/``torch``), and
``--no-home`` keeps the host home — and any keys ``~/.bashrc`` would export into a
``bash -lc`` login shell — out of the sandbox. Env the agent legitimately needs (HF
auth/caches, proxies) is forwarded via ``APPTAINERENV_*`` (:func:`forward_env`), never
on the command line. ``--nv`` on GPU steps passes the device + CUDA driver through.

The sandbox is **mandatory**: every agent shell step is wrapped, with no opt-out.
:func:`require_apptainer` **hard-fails the run** if Apptainer cannot execute the image on
this host (or no image is configured) rather than letting a step run unconfined.
"""

from __future__ import annotations

import functools
import os
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from reclaim_repro.inputs import RunPaths

# Host env vars forwarded into the ``--cleanenv`` container (mirrored to ``APPTAINERENV_*``
# in :func:`forward_env`) so HF auth, cache locations, and proxies survive the env scrub
# without ever appearing on the command line.
FORWARD_ENV = (
    "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN",
    "HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HUB_ENABLE_HF_TRANSFER",
    "UV_CACHE_DIR", "PIP_CACHE_DIR",
    "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy",
)

# One-time, cached result of the runtime Apptainer probe, keyed by image (so tests can
# reset it and distinct images probe independently).
_PROBE: dict[str, bool] = {}


def apptainer_usable(image: str) -> bool:
    """True iff ``apptainer`` can actually ``exec`` *this image* on this host.

    ``shutil.which`` alone is not enough — the image must exist and be runnable here
    (a bad path or a node without unprivileged container support both fail). We run a
    trivial ``apptainer exec <image> true`` once per image and cache the verdict.
    """
    if image not in _PROBE:
        _PROBE[image] = _probe_apptainer(image)
    return _PROBE[image]


def _probe_apptainer(image: str) -> bool:
    if not image or not shutil.which("apptainer") or not Path(image).exists():
        return False
    try:
        proc = subprocess.run(
            ["apptainer", "exec", "--cleanenv", image, "true"],
            capture_output=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


def require_apptainer(image: str | None) -> None:
    """Hard-fail the run unless Apptainer can sandbox here — the sandbox is mandatory.

    Called once before any agent step runs. Refusing here is the point: a step must
    never run unconfined, so a missing image or an unusable Apptainer aborts the run.
    """
    if not image:
        raise SystemExit(
            "the Apptainer sandbox is mandatory but no image is configured: pass "
            "--apptainer-image (or set $REPRO_APPTAINER_SIF), or pick a cluster profile "
            "that pins one. The reproduction agent refuses to run shell steps "
            "unconfined."
        )
    if apptainer_usable(image):
        return
    raise SystemExit(
        f"the Apptainer sandbox is mandatory but unusable here: `apptainer exec {image} "
        "true` failed — check that `apptainer` is on PATH and the image exists and is "
        "readable on this node (set $APPTAINER_CACHEDIR to /scratch or NVMe if you hit a "
        "~/.apptainer quota error)."
    )


def forward_env() -> None:
    """Mirror the :data:`FORWARD_ENV` allowlist into ``APPTAINERENV_*`` for ``--cleanenv``.

    Apptainer imports ``APPTAINERENV_FOO`` as ``FOO`` inside the container even under
    ``--cleanenv``. Setting it in *our* process environment (never on argv) forwards HF
    auth and cache/proxy settings without exposing a token via ``ps``. Idempotent — an
    explicit ``APPTAINERENV_*`` already in the environment wins.

    It also pins ``UV_PYTHON_PREFERENCE=system`` so ``uv venv --python 3.12`` resolves to
    the image's bundled ``/usr/bin/python3.12`` (whose dev headers sit at the conventional
    ``/usr/include/python3.12/``) instead of downloading a managed standalone CPython —
    that download lands outside ``/usr/include`` and agents reliably fail to point gcc at
    it, so C/Cython extension builds die with ``Python.h: No such file``. ``system``
    prefers the bundled interpreter but still falls back to a managed download if 3.12 is
    somehow absent. The ``UV_PYTHON_INSTALL_DIR`` + XDG redirects below keep that fallback
    (and any other tool dir that defaults to the read-only ``--no-home`` ``$HOME``) on the
    rw-bound ``~/.cache`` so it works *and* persists across papers.
    """
    for var in FORWARD_ENV:
        value = os.environ.get(var)
        if value is not None:
            os.environ.setdefault(f"APPTAINERENV_{var}", value)
    os.environ.setdefault("APPTAINERENV_UV_PYTHON_PREFERENCE", "system")
    # A clone of a private or deleted repo must fail, not ask. git prompts on /dev/tty
    # rather than stdin, so closing the step's stdin is not enough on its own; this makes
    # the failure immediate and legible ("terminal prompts disabled") in the tool result,
    # which the agent can act on. Paired with the tty-free session in workspace_bash.
    os.environ.setdefault("APPTAINERENV_GIT_TERMINAL_PROMPT", "0")
    cache = Path.home() / ".cache"
    home_write_defaults = {
        "UV_PYTHON_INSTALL_DIR": cache / "uv" / "python",
        "XDG_CACHE_HOME": cache,
        "XDG_DATA_HOME": cache / "xdg-data",
        "XDG_CONFIG_HOME": cache / "xdg-config",
    }
    for var, path in home_write_defaults.items():
        os.environ.setdefault(f"APPTAINERENV_{var}", str(path))


# Fixed, SHORT container mountpoints for the episode dirs. The host paths carry a deep,
# per-run, random-id path (``…/agent_runs/<arxiv>/<budget>/<run_id>/workspace``); we bind
# each onto a stable short name *inside* the container so the agent uses ``/repro/workspace``
# etc. — the same across every episode, never has to ``cd`` (``--pwd`` lands each step in
# the workspace), and never has to retype the long path. The file tools (``files.py``) run
# host-side, so they translate these ``/repro/...`` paths back to the host run dir. The
# layout mirrors ``inputs.RunPaths``: ``<run_dir>/{workspace,reference,evidence}`` ↔
# ``/repro/{workspace,reference,evidence}``.
CONTAINER_RUN = "/repro"
CONTAINER_WORKSPACE = "/repro/workspace"
CONTAINER_REFERENCE = "/repro/reference"
CONTAINER_EVIDENCE = "/repro/evidence"


@functools.lru_cache(maxsize=None)
def _which_cached(name: str, _lookup: Callable[[str], str | None]) -> str | None:
    return _lookup(name)


def which_tool(name: str) -> str | None:
    """``shutil.which(name)`` for a host binary, resolved once per process.

    Every sandboxed step wraps itself with a ``uv`` lookup and (when capped) a
    ``taskset`` lookup — roughly a thousand tool calls per episode, each paying a full
    PATH scan of tens of ``stat`` calls — and neither binary can appear, move, or vanish
    while a run is in flight, so the scan happens once and the answer is reused. The
    lookup function is part of the cache key, so a caller that swaps ``shutil.which``
    out is never served an earlier lookup's answer.
    """
    return _which_cached(name, shutil.which)


@dataclass(frozen=True)
class Bind:
    """One ``apptainer --bind`` spec: ``src`` on the host → ``dst`` in the container."""

    src: str
    dst: str
    ro: bool = False

    def arg(self) -> str:
        spec = self.src if self.dst == self.src else f"{self.src}:{self.dst}"
        return f"{spec}:ro" if self.ro else spec


@dataclass(frozen=True)
class Sandbox:
    """How to wrap one step's ``bash -lc`` body in the episode's Apptainer container.

    ``image`` is the read-only ``.sif`` that becomes ``/``; ``binds`` are the mounts
    (the episode's workspace/evidence remapped to short ``/repro`` paths rw, reference ro,
    plus the node-local ``/tmp`` and the shared caches at their own paths); ``workdir`` is
    the container cwd every step lands in (``--pwd``), so the agent never has to ``cd``.
    The wrap is unconditional — there is no opt-out (see module docstring).
    """

    image: str
    binds: tuple[Bind, ...] = ()
    workdir: str = CONTAINER_WORKSPACE
    cpus: int | None = None  # CPU-step core cap; None = uncapped (see CPU-cap helpers below)
    mem_gb: int | None = None  # CPU-step per-process address-space cap; None = uncapped

    def wrap_argv(self, body: str, *, nv: bool = False) -> list[str]:
        """``["apptainer", "exec", <flags...>, <image>, "bash", "-lc", <body>]``.

        ``nv`` adds ``--nv`` for a GPU step; a capped CPU step is prefixed with
        ``taskset`` instead (CPU-cap helpers below), so its default build parallelism
        follows the shrunk core count, and its body gets the ``ulimit -v`` RAM cap.
        """
        prefix = taskset_argv(self.cpus) if not nv and self.cpus else []
        if not nv and self.mem_gb:
            body = ulimit_body(self.mem_gb, body)
        return [*prefix, *self._apptainer_prefix(nv=nv), "bash", "-lc", body]

    def status(self) -> str:
        """Human-readable effective state for the setup summary / evidence."""
        rw = ", ".join(f"{b.src}->{b.dst}" for b in self.binds if not b.ro) or "(none)"
        ro = ", ".join(f"{b.src}->{b.dst}" for b in self.binds if b.ro) or "(none)"
        cap = (f"; cpus<={self.cpus}" if self.cpus else "") + (f"; mem<={self.mem_gb}G" if self.mem_gb else "")
        return f"apptainer (mandatory) {self.image}; pwd {self.workdir}; rw: {rw}; ro: {ro}{cap}"

    def _apptainer_prefix(self, *, nv: bool) -> list[str]:
        # --pwd lands each step in the workspace mount, so the body needs no `cd` to find
        # it and the agent never types the long host path.
        argv = ["apptainer", "exec", "--cleanenv", "--no-home", "--pwd", self.workdir]
        if nv:
            # GPU steps stream into a teed evidence log (slurm.run_in_session); a
            # block-buffered python would hold its last ~8KB in the pipe and lose it
            # when SLURM kills the step at the --time wall, so force write-through.
            # (`stdbuf` can't do this here: its LD_PRELOAD does not survive into the
            # container.) --nv passes the device + CUDA driver through.
            argv += ["--nv", "--env", "PYTHONUNBUFFERED=1"]
        elif self.cpus:
            # cap build fan-out for tools that read env instead of nproc
            argv += build_env_args(self.cpus)
        for bind in self.binds:
            argv += ["--bind", bind.arg()]
        # The agent image bundles its own `uv` + `python3.12`, so this bind is just an
        # override: when the host has a (newer) `uv`, mount it over the image's copy
        # (aarch64, runs fine in the container); otherwise the baked-in `uv` is used.
        uv = which_tool("uv")
        if uv:
            argv += ["--bind", f"{uv}:/usr/local/bin/uv"]
        argv.append(self.image)
        return argv


def default_cache_dirs() -> list[Path]:
    """Shared caches that stay writable so HF/uv/pip/torch don't re-fetch per paper.

    ``~/.cache`` covers the default HF hub cache, uv, pip, torch hub, and triton; a
    custom ``HF_HOME``/``UV_CACHE_DIR`` (and the legacy ``~/.huggingface`` token dir)
    are added when set/present so the HF token stays both readable and refreshable.
    """
    home = Path.home()
    dirs = [home / ".cache"]
    for var in ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "UV_CACHE_DIR"):
        value = os.environ.get(var)
        if value:
            dirs.append(Path(value).expanduser())
    legacy = home / ".huggingface"
    if legacy.exists():
        dirs.append(legacy)
    return dirs


def from_run_paths(
    run_paths: "RunPaths",
    *,
    image: str,
    caches: Iterable[Path] | None = None,
    cpus: int | None = None,
    mem_gb: int | None = None,
) -> Sandbox:
    """Build the episode's sandbox binds: workspace/evidence remapped to short ``/repro``
    paths (rw), reference ro, the node-local ``/tmp`` and the shared caches at their own
    paths (rw).

    Every bind source must exist before ``apptainer`` runs, so we create the episode dirs
    and cache roots up front (idempotent) and drop any that can't be created. Cache roots
    default to :func:`default_cache_dirs`. ``cpus`` sets the CPU-step core cap (CPU-cap helpers below).
    """
    binds: list[Bind] = [Bind("/tmp", "/tmp")]  # node-local scratch (a real bind, not a tmpfs)
    workspace = _ensure_dir(run_paths.workspace)
    if workspace is not None:
        binds.append(Bind(str(workspace), CONTAINER_WORKSPACE))
    evidence = _ensure_dir(run_paths.evidence)
    if evidence is not None:
        binds.append(Bind(str(evidence), CONTAINER_EVIDENCE))
    reference = _ensure_dir(run_paths.reference)
    if reference is not None:
        binds.append(Bind(str(reference), CONTAINER_REFERENCE, ro=True))
    seen: set[str] = set()
    for cache in (caches if caches is not None else default_cache_dirs()):
        resolved = _ensure_dir(cache)
        if resolved is not None and str(resolved) not in seen:
            seen.add(str(resolved))
            binds.append(Bind(str(resolved), str(resolved)))
    return Sandbox(image=image, binds=tuple(binds), workdir=CONTAINER_WORKSPACE, cpus=cpus, mem_gb=mem_gb)


def _ensure_dir(path: Path | None) -> Path | None:
    if path is None:
        return None
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        return Path(path).resolve()
    except OSError:
        return None


# ---- CPU-cap helpers -------------------------------------------------------
# Per-agent CPU and build-parallelism caps for CPU-step Apptainer sandboxing.
#
# The failure they prevent: the sweep's vLLM server shares one --mem cgroup with the
# parallel reproduction agents' CPU shell steps on the same node. An agent that runs
# something like ``uv pip install flash-attn --no-build-isolation`` gets a ninja that
# defaults -j to nproc, and the parallel nvcc jobs can exceed the node's memory cgroup,
# letting the kernel OOM-kill the vLLM engine.
#
# Apptainer's native ``--cpus``/``--memory`` cgroup flags are unusable without cgroup
# delegation, which most shared clusters do not grant: they fail with "while applying
# cgroups config ... memory.max: no such file or directory". The portable fix is
# two-part instead:
#
# 1. ``taskset`` sched-affinity pinning. Affinity survives into the container and
#    ``nproc`` honors it, so every build tool's default ``-j`` follows without the
#    container runtime needing cgroup permissions at all.
# 2. Explicit build-parallelism env vars for tools that read ``MAX_JOBS`` etc. directly
#    instead of asking ``nproc``.
#
# Caps apply ONLY to CPU steps (``workspace_bash``, ``nv=False``), which run on the
# shared serving node. GPU steps (``nv=True``) run on dedicated JIT-allocated nodes where
# capping would hurt legitimate work, so they are never touched here.


def taskset_argv(cpus: int) -> list[str]:
    """``["taskset", "-c", "<core ids>"]`` pinning the step to *cpus* cores, or ``[]``.

    Returns ``[]`` when there is nothing to do: no ``taskset`` binary, a non-positive
    cap, or the process's current allowed set is already at or below *cpus* cores (no
    shrinking needed).

    Core selection rotates ``sorted(os.sched_getaffinity(0))`` by
    ``(os.getpid() * cpus) % len(allowed)`` and takes the first *cpus* entries. Each
    paper's episode runs in its own orchestrator process, so a pid-derived offset
    spreads concurrent agents across the job's cores without any coordination between
    them; overlap between two agents' picks is harmless (the OS scheduler balances
    runnable threads across the machine regardless), the only thing that matters is
    that ``nproc`` inside the taskset slice reports *cpus*.
    """
    if cpus <= 0 or not which_tool("taskset"):
        return []
    try:
        allowed = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return []  # non-Linux, or affinity unsupported here
    if len(allowed) <= cpus:
        return []
    offset = (os.getpid() * cpus) % len(allowed)
    rotated = allowed[offset:] + allowed[:offset]
    cores = ",".join(str(c) for c in rotated[:cpus])
    return ["taskset", "-c", cores]


def build_env_args(cpus: int) -> list[str]:
    """Apptainer ``--env`` args capping build fan-out for tools that ignore affinity.

    Compile RAM scales with job count, not core count, so these are set independent of
    whether :func:`taskset_argv` fired. ``MAX_JOBS`` is what flash-attn/torch extension
    builds honor directly; ``NVCC_THREADS``, ``CMAKE_BUILD_PARALLEL_LEVEL``, and
    ``MAKEFLAGS`` cover cmake/make/nvcc invocations that read their job count from the
    environment instead of calling ``nproc``.

    The job count stays at 4 even when the core slice is wider: one nvcc job peaks at
    3-5 GB, so 4 jobs is what fits the per-paper RAM share (see :func:`ulimit_body`),
    while the extra cores still speed up everything that is not a compile.
    """
    jobs = min(cpus, 4)
    return [
        "--env", f"MAX_JOBS={jobs}",
        "--env", "NVCC_THREADS=2",
        "--env", f"CMAKE_BUILD_PARALLEL_LEVEL={jobs}",
        "--env", f"MAKEFLAGS=-j{jobs}",
    ]


def ulimit_body(mem_gb: int, body: str) -> str:
    """Prefix *body* with a per-process address-space cap of *mem_gb* GiB.

    Without cgroup delegation there is no true per-tree RAM ceiling; ``ulimit -v``
    (RLIMIT_AS) is the portable approximation. It is per *process*, so a build's N
    jobs can jointly exceed it (bounded by ``build_env_args``'s fan-out cap), and it
    counts virtual address space: a step that mmaps a dataset larger than the cap
    fails even though the resident footprint is small. That failure mode is accepted;
    bulk data work belongs on the paper's own ``run_gpu`` node rather than the shared serving node.
    """
    return f"ulimit -v {mem_gb * 1024 * 1024} 2>/dev/null; {body}"
