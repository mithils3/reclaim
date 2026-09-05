"""SLURM substrate — a held GPU allocation the agent runs successive steps into.

The agent's GPU access is **one allocation held for the episode**, not a fresh
``salloc`` per command. The first ``run_gpu`` call acquires it and every later call
runs into the same node, so the agent stops re-queueing between install → verify →
run:

    # acquire once (returns the instant the alloc is granted, then stays up):
    salloc --no-shell -A <account> -p <partition> --nodes=1 --gpus=<k> --time=<min>
    # run each step into the held jobid (no new queue wait), inside the sandbox image:
    srun --jobid=<jobid> --ntasks=1 apptainer exec --nv ... <sif> bash -lc 'cd <ws> && <cmd>'
    # release when the agent is done (or at teardown):
    scancel <jobid>

``<account>/<partition>`` come from the cluster profile (``cluster.py``);
``<k>``/``<min>`` come from the model's ``run_gpu`` arguments on the call that
*starts* the session. ``--time`` is the hold's hard lifetime (SLURM kills it at the
wall limit) and the budget pre-authorization. Queue wait is naturally excluded from
billing: ``acquire_session`` blocks during the wait and the meter only starts the
clock once the allocation is granted (see ``gpu_session``).

The ``srun`` payload (the Apptainer wrap + ``cd`` + ``<cmd>``) is built by
``env.exec_argv(..., on_gpu=True, sandbox=...)`` — ``on_gpu`` adds ``--nv`` so the GPU
step sees the device + CUDA driver; the CUDA toolkit itself comes from the image, not a
host ``module load``. ``build_acquire``/``build_srun`` are pure (the tests assert the
exact argv); the ``*_session`` helpers execute them. ``salloc --no-shell`` is the one
cluster-specific seam — some sites need a held ``salloc ... sleep`` or
``SallocDefaultCommand`` instead.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from typing import TYPE_CHECKING, IO

from reclaim_repro import env
from reclaim_repro.cluster import Cluster

if TYPE_CHECKING:
    from reclaim_repro.sandbox import Sandbox

# Host RAM (GB) and CPU cores requested per GPU on the salloc. Each per-GPU request is
# sized to stay inside one GPU's share of a node, so the scheduler does not round the
# allocation up to a whole node, while still giving each GPU enough host memory and
# cores for dataloading and preprocessing that a per-core partition default would
# otherwise starve. Set $RECLAIM_MEM_GB_PER_GPU and $RECLAIM_CPUS_PER_GPU to match a
# different node shape.
MEM_GB_PER_GPU = int(os.environ.get("RECLAIM_MEM_GB_PER_GPU") or 100)
CPUS_PER_GPU = int(os.environ.get("RECLAIM_CPUS_PER_GPU") or 60)

# Bytes of each stream a step keeps in RAM. The *complete* stream is on disk (the teed
# ``log_path``), and the agent is handed that path, so memory only has to cover what the
# caller actually reads back: ``output.shape`` keeps 40k chars (head 25% / tail 75%) and
# ``output.tail`` 4k. We retain a head and a tail region of this size each — over 60k
# chars apiece even for 4-byte UTF-8 — and let the middle go, which ``shape`` was going
# to elide anyway. A training step can print hundreds of MB, and the buffer used to hold
# every byte of it for the life of the step on a node shared with the sweep's vLLM
# worker, which the kernel then OOM-killed.
STREAM_KEEP_BYTES = 256 * 1024

# salloc prints "... allocation <jobid>" (Pending then Granted share the number).
_JOBID_RE = re.compile(r"allocation (\d+)")
# srun into a dead/expired allocation fails with one of these — the held session is
# gone (hit --time or was cancelled), so the caller should drop it and re-acquire.
# The first three are the scancel/revoke path; the last three are the --time-expiry
# path (a held job drains COMPLETING -> EXPIRED, each with its own srun wording), which
# is the common one for short held sessions and must drop the session just the same.
_LOST_MARKERS = (
    "invalid job id",
    "unable to confirm allocation",
    "job allocation has been revoked",
    "already completing or completed",  # "Unable to create step ... Job/step already completing or completed"
    "has expired",                      # "Slurm job N has expired"
    "expired or invalid job",           # "Check SLURM_JOB_ID environment variable. Expired or invalid job N"
)


class SlurmConfigError(Exception):
    """Unusable cluster substrate for a GPU allocation (no account/partition, or a
    gpu count outside the node): a domain error, not a library-level SystemExit."""


@dataclass
class StepResult:
    """Outcome of one GPU step run into the held allocation."""

    ok: bool
    returncode: int
    stdout: str
    stderr: str
    elapsed_s: float
    command: list[str]


@dataclass
class SessionHandle:
    """Outcome of acquiring a held allocation — ``jobid`` set iff ``ok``."""

    ok: bool
    jobid: str | None
    stderr: str


def _require_target(cluster: Cluster, gpus: int, partition: str | None) -> None:
    """Validate the operator-set substrate before building any allocation argv."""
    if not cluster.account or not partition:
        raise SlurmConfigError(
            f"cluster {cluster.name!r} has no account/partition for a GPU allocation "
            "(pass --account/--partition or pick a cluster profile that sets them)."
        )
    if gpus < 1 or gpus > cluster.gpus_per_node:
        raise SlurmConfigError(
            f"run_gpu gpus={gpus} out of range for {cluster.name!r} "
            f"(1..{cluster.gpus_per_node} per node)."
        )


def build_acquire(
    cluster: Cluster, *, gpus: int, minutes: int, partition: str | None = None
) -> list[str]:
    """Argv that holds one allocation (``salloc --no-shell``) and returns once granted.

    ``partition`` overrides the profile's default pool for this allocation (the agent
    picks it from ``list_partitions``); ``None`` falls back to ``cluster.partition``.

    Host RAM and CPU cores scale with the GPU request rather than riding the
    partition's per-core default (often 1000 MB/core, which the GPU->core binding
    alone leaves too tight for dataloaders/preprocessing): ``--mem`` totals
    ``MEM_GB_PER_GPU`` per GPU and ``--cpus-per-gpu`` pins ``CPUS_PER_GPU`` cores to
    each GPU (per-GPU, so it survives any future multi-task step). Both are sized so
    an SU bill of ``max(gpus, ceil(cores/72), ceil(mem_GB/110))`` stays at the GPU
    count (60/72 and 100/110 both round up to 1 per GPU) and never charges for more of
    the node than the GPUs already imply.
    """
    part = partition or cluster.partition
    _require_target(cluster, gpus, part)
    return [
        "salloc",
        "--no-shell",
        "-A", cluster.account,
        "-p", part,
        "--nodes=1",
        f"--gpus={int(gpus)}",
        f"--mem={MEM_GB_PER_GPU * int(gpus)}G",
        f"--cpus-per-gpu={CPUS_PER_GPU}",
        f"--time={int(minutes)}",
    ]


def build_srun(
    workspace: Path | str,
    cmd: str,
    *,
    jobid: str,
    sandbox: "Sandbox | None" = None,
) -> list[str]:
    """Argv that runs ``cmd`` into the already-held allocation ``jobid``.

    ``sandbox`` (always active for agent steps) wraps the payload in the episode's
    Apptainer container, so the untrusted GPU step runs inside the read-only image and
    is write-confined to the episode's dirs on the compute node. ``srun`` itself — the
    trusted launcher — stays outside the wrap.
    """
    # --unbuffered: forward step output as it arrives instead of line-buffering it in
    # slurmstepd, so the streamed evidence log (run_in_session's log_path) holds
    # everything the step printed even when SLURM kills it at the --time wall.
    return [
        "srun",
        f"--jobid={jobid}",
        "--ntasks=1",
        "--unbuffered",
        *env.exec_argv(workspace, cmd, on_gpu=True, sandbox=sandbox),
    ]


def decode(blob: bytes | str | None) -> str:
    """Text of a captured stream, whether the caller got str or raw bytes.

    ``subprocess.run(..., text=True)`` only decodes on the *normal* return path: the
    output it attaches to a raised ``TimeoutExpired`` is still ``bytes``. Concatenating
    that onto a str raises ``TypeError: can't concat str to bytes``, which is exactly
    how a queue-wait timeout used to take down the whole ``run_gpu`` call — the agent
    got a bare type error instead of "salloc timed out", so it could not tell a
    starved queue from a broken command and blind-retried the step.

    ``workspace_bash`` captures in binary for the same reason (text mode's universal
    newlines would turn every progress redraw into its own line) and decodes here too.
    """
    if blob is None:
        return ""
    if isinstance(blob, str):
        return blob
    return blob.decode("utf-8", errors="replace")


def acquire_session(
    cluster: Cluster,
    *,
    gpus: int,
    minutes: int,
    timeout: float | None = None,
    partition: str | None = None,
) -> SessionHandle:
    """Hold a GPU allocation; block through the queue wait, return once it is granted."""
    argv = build_acquire(cluster, gpus=gpus, minutes=minutes, partition=partition)
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        # Keep whatever salloc printed before we gave up: the "Pending job allocation
        # <id>" line and the QOS/reservation reason are the only signal the agent gets
        # about *why* the queue never granted a node.
        printed = "\n".join(
            part for part in (decode(exc.stdout), decode(exc.stderr)) if part.strip()
        )
        waited = f" after {exc.timeout:.0f}s" if exc.timeout else ""
        return SessionHandle(
            ok=False,
            jobid=None,
            stderr=f"{printed}\n[salloc timed out{waited} waiting for the allocation]".lstrip("\n"),
        )
    found = _JOBID_RE.findall(decode(proc.stdout) + "\n" + decode(proc.stderr))
    jobid = found[-1] if found else None
    return SessionHandle(
        ok=jobid is not None and proc.returncode == 0,
        jobid=jobid,
        stderr=decode(proc.stderr),
    )


def run_in_session(
    workspace: Path | str,
    cmd: str,
    *,
    jobid: str,
    timeout: float | None = None,
    sandbox: "Sandbox | None" = None,
    log_path: Path | str | None = None,
) -> StepResult:
    """Run one command into the held allocation and time it (queue wait already paid).

    Output is *streamed*, not just captured: each chunk is teed to ``log_path`` (the
    episode's ``evidence/gpu_step_<n>.log``) as it arrives, so when the step dies —
    our timeout, the --time wall, scancel — everything printed up to the kill is on
    disk and in the returned (partial) ``StepResult`` instead of dying in a pipe
    buffer with the process.

    Only the head and tail of each stream are kept in memory (``STREAM_KEEP_BYTES``
    each); the log file is the complete record.
    """
    argv = build_srun(workspace, cmd, jobid=jobid, sandbox=sandbox)
    start = time.monotonic()
    try:
        # Own process group: a timeout kill must take out srun's whole tree, or an
        # orphan keeps the pipes open and the pumps block long past the kill.
        proc = subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL, start_new_session=True,
        )
    except OSError as exc:
        return StepResult(
            ok=False, returncode=127, stdout="", stderr=f"{type(exc).__name__}: {exc}",
            elapsed_s=time.monotonic() - start, command=argv,
        )
    log = open(log_path, "ab", buffering=0) if log_path else None
    try:
        out_buf = _StreamCapture()
        err_buf = _StreamCapture()
        pumps = [
            threading.Thread(target=_pump, args=(proc.stdout, out_buf, log), daemon=True),
            threading.Thread(target=_pump, args=(proc.stderr, err_buf, log), daemon=True),
        ]
        for t in pumps:
            t.start()
        timed_out = False
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_tree(proc)
            proc.wait()
        for t in pumps:
            t.join(timeout=10)
    finally:
        if log is not None:
            log.close()
    stdout = out_buf.text()
    stderr = err_buf.text()
    if timed_out:
        stderr += "\n[step exceeded timeout]"
    return StepResult(
        ok=proc.returncode == 0 and not timed_out,
        returncode=124 if timed_out else proc.returncode,
        stdout=stdout,
        stderr=stderr,
        elapsed_s=time.monotonic() - start,
        command=argv,
    )


def _kill_tree(proc: subprocess.Popen) -> None:
    """Kill the step's whole process group (falling back to just the leader)."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (OSError, ProcessLookupError):
        proc.kill()


class _StreamCapture:
    """The head and the tail of one streamed pipe, each bounded to ``keep`` bytes.

    The head fills first and then freezes; every later chunk lands in a deque that
    evicts from the front once it holds ``keep`` bytes. So a step that prints far more
    than the cap costs a fixed ~2x``keep`` of RAM instead of its whole output, and what
    ran through the middle is on disk (the teed log) only — the same middle
    ``output.shape`` elides before the agent ever sees it.
    """

    __slots__ = ("keep", "_head", "_head_len", "_tail", "_tail_len")

    def __init__(self, keep: int = STREAM_KEEP_BYTES) -> None:
        self.keep = keep
        self._head: list[bytes] = []
        self._head_len = 0
        self._tail: deque[bytes] = deque()
        self._tail_len = 0

    def add(self, chunk: bytes) -> None:
        if self._head_len < self.keep:
            room = self.keep - self._head_len
            self._head.append(chunk[:room])
            self._head_len += min(room, len(chunk))
            chunk = chunk[room:]
            if not chunk:
                return
        self._tail.append(chunk)
        self._tail_len += len(chunk)
        # Drop whole chunks from the front while the rest still covers ``keep`` bytes;
        # the newest chunk is never dropped, however large it is.
        while len(self._tail) > 1 and self._tail_len - len(self._tail[0]) >= self.keep:
            self._tail_len -= len(self._tail.popleft())

    def text(self) -> str:
        """Everything retained, decoded once (chars the agent may still read)."""
        return (b"".join(self._head) + b"".join(self._tail)).decode("utf-8", errors="replace")


def _pump(pipe: IO[bytes] | None, buf: _StreamCapture, log: IO[bytes] | None) -> None:
    """Drain one pipe: retain the head/tail in memory and tee every chunk to the log.

    The log write happens per-chunk (unbuffered handle), so the on-disk file is
    current the moment the process is killed; both streams share one log in arrival
    order — the interleaving a terminal would have shown. Only the retained window is
    bounded; the tee is always complete.
    """
    if pipe is None:
        return
    try:
        for chunk in iter(lambda: pipe.read1(65536), b""):
            buf.add(chunk)
            if log is not None:
                try:
                    log.write(chunk)
                except OSError:
                    log = None  # keep collecting for the result even if the disk sink dies
    except (OSError, ValueError):
        pass
    finally:
        try:
            pipe.close()
        except OSError:
            pass


def release_session(jobid: str) -> None:
    """Free the held allocation (``scancel``); best-effort, never raises."""
    try:
        subprocess.run(["scancel", str(jobid)], capture_output=True, text=True, timeout=60)
    except (subprocess.SubprocessError, OSError):
        pass


def session_lost(step: StepResult) -> bool:
    """Whether a step failed because the held allocation is gone (expired/cancelled)."""
    if step.ok:
        return False
    blob = (step.stderr or "").lower()
    return any(marker in blob for marker in _LOST_MARKERS)
