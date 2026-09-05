"""Per-run telemetry sidecar: a metrics beacon inside each held GPU allocation.

``gpu_session.ensure_session`` calls ``start`` the moment an allocation is
granted; ``release``/``drop_lost`` call ``stop``. The beacon is one detached
``srun --overlap`` step into the held jobid running
``python -m reclaim_repro.metrics_beacon --role run``, so the run's JIT GPU
node shows up in the viewer's host panel while the agent works. Same opt-in as
``telemetry_sink`` (``event_sink.credentials_from_env``) and the same discipline:
every entry point swallows everything — telemetry may never break a run or a test.

The live ``srun`` client Popens live in a module registry keyed by jobid
(rather than on ``GpuSession``) so ``context.py`` stays plumbing-free. ``stop``
only terminates the *local* srun client — the ``scancel`` on release kills the
remote step itself; we just must not leave the client lingering.
"""

from __future__ import annotations

import subprocess
import sys

from reclaim_repro.event_sink import credentials_from_env

_BEACONS: dict[str, subprocess.Popen] = {}


def start(ctx, jobid: str) -> None:
    """Spawn the detached beacon into the held allocation; silent no-op on any failure.

    ``srun --jobid`` targets the allocation explicitly, so the orchestrator's
    SLURM_* env scrub doesn't matter; the venv is on shared FS, so
    ``sys.executable`` resolves on the worker node too.
    """
    try:
        if credentials_from_env() is None or jobid in _BEACONS:
            return
        from reclaim_repro.telemetry_sink import run_id_of
        run_id = run_id_of(ctx)
        if not run_id:
            return
        argv = [
            "srun", f"--jobid={jobid}", "--overlap", "--ntasks=1", "--nodes=1",
            "--output=/dev/null", "--error=/dev/null",
            sys.executable, "-m", "reclaim_repro.metrics_beacon",
            "--role", "run", "--run-id", run_id, "--interval", "20",
        ]
        _BEACONS[jobid] = subprocess.Popen(
            argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    except Exception:  # noqa: BLE001 — telemetry may never break a run
        pass


def stop(jobid: str | None) -> None:
    """Terminate the local srun client (scancel kills the step itself); never raises."""
    try:
        proc = _BEACONS.pop(jobid, None) if jobid else None
        if proc is not None:
            proc.terminate()
    except Exception:  # noqa: BLE001 — telemetry may never break a run
        pass
