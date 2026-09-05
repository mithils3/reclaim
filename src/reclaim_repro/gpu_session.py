"""Held-GPU-session lifecycle: acquire once, reuse, bill wall clock, release.

``run_gpu`` no longer allocates+releases a GPU per command. The first call ACQUIRES
one SLURM allocation that stays up; every later call runs into it with no re-queue;
and the agent RELEASES it when done. Because the node is held across the agent's
reasoning/install gaps too, the meter bills **wall clock** — ``gpus x (held seconds)
x hw`` — charged incrementally so the budget ceiling still fires mid-hold.

The seam is small and shared:

* ``ensure_session`` — acquire if none is held (queue wait excluded: the clock starts
  only once the allocation is granted);
* ``charge_accrued`` — bill the wall held since the last charge and advance the
  marker (so the running total equals real held wall, never double-counted). Called
  by ``run_gpu`` after each step *and* by the loop guardrail between rounds, so a
  long reasoning gap on a held node still depletes the budget;
* ``release`` — bill the final sliver, ``scancel``, clear the session, record it;
* ``teardown_all`` — release any session still held at loop end (crash/interrupt
  safety net over the per-episode release).

``tools/run_gpu.py`` is the thin tool wrapper over this; ``slurm.py`` is the raw
salloc/srun/scancel substrate underneath.
"""

from __future__ import annotations

import time
from typing import Iterable

from reclaim_repro import budget as budget_mod
from reclaim_repro import evidence as evidence_mod
from reclaim_repro import run_beacon, slurm
from reclaim_repro.context import ExecutionContext, GpuSession


def ensure_session(
    ctx: ExecutionContext,
    *,
    gpus: int,
    minutes: int,
    timeout: float | None = None,
    partition: str | None = None,
) -> tuple[GpuSession | None, str | None]:
    """Return the live held session, acquiring one if none is held.

    On success the budget clock starts *now* (the allocation is granted; the queue
    wait that ``acquire_session`` just blocked through is not billed). ``partition``
    overrides the profile's default pool for this allocation (``None`` keeps it).
    Returns ``(session, None)`` or ``(None, error)``.
    """
    if ctx.session is not None:
        return ctx.session, None
    if ctx.cluster is None:
        return None, "no cluster profile bound for this episode"
    try:
        handle = slurm.acquire_session(
            ctx.cluster, gpus=gpus, minutes=minutes, timeout=timeout, partition=partition
        )
    except slurm.SlurmConfigError as exc:
        # Unusable substrate (no account/partition, bad gpu count): a clean acquire
        # failure surfaced to the model, not a library-level process crash.
        return None, str(exc)
    if not handle.ok or not handle.jobid:
        # One-line reason the acquire failed, from salloc's stderr.
        tail = (handle.stderr or "").strip().splitlines()
        return None, (tail[-1] if tail else "salloc did not report a job allocation")
    now = time.monotonic()
    session = GpuSession(
        jobid=handle.jobid,
        gpus=gpus,
        minutes=minutes,
        hw=ctx.cluster.hw,
        started=now,
        last_charged=now,
        partition=partition or ctx.cluster.partition,
    )
    ctx.session = session
    # Telemetry sidecar into the fresh allocation; no-op unless the telemetry env is set.
    run_beacon.start(ctx, handle.jobid)
    return session, None


def held_seconds(session: GpuSession) -> float:
    """Total wall the allocation has been held (grant → now)."""
    return max(0.0, time.monotonic() - session.started)


def charge_accrued(ctx: ExecutionContext) -> float:
    """Bill wall held since the last charge against the budget; return the cost.

    No-op (returns 0.0) when no session is held or no budget is bound. Advances
    ``last_charged`` so repeated calls never double-count the same wall.
    """
    session = ctx.session
    if session is None or ctx.budget is None:
        return 0.0
    now = time.monotonic()
    seconds = max(0.0, now - session.last_charged)
    session.last_charged = now
    cost = budget_mod.step_cost_hours(session.gpus, seconds, session.hw)
    ctx.budget.consume(cost)
    return cost


def drop_lost(ctx: ExecutionContext) -> None:
    """Clear a session whose allocation is already gone (expired/cancelled) — no scancel."""
    charge_accrued(ctx)
    if ctx.session is not None:
        run_beacon.stop(ctx.session.jobid)  # the step died with the allocation; drop the client
    ctx.session = None


def release(ctx: ExecutionContext, reason: str = "done") -> dict | None:
    """Bill the final held wall, ``scancel`` the allocation, clear and record it."""
    session = ctx.session
    if session is None:
        return None
    final_charge = charge_accrued(ctx)
    slurm.release_session(session.jobid)
    run_beacon.stop(session.jobid)  # scancel killed the step; don't leave the srun client
    record = {
        "jobid": session.jobid,
        "gpus": session.gpus,
        "hw": session.hw,
        "partition": session.partition,
        "held_seconds": round(held_seconds(session), 1),
        "final_charge_h100_hours": round(final_charge, 4),
        "reason": reason,
    }
    ctx.session = None
    if ctx.evidence is not None:
        evidence_mod.append_trajectory(ctx.evidence, {"type": "gpu_release", **record})
    return record


def teardown_all(contexts: Iterable[ExecutionContext]) -> None:
    """Release every still-held session at loop end (safety net over per-episode release)."""
    for ctx in contexts:
        try:
            release(ctx, "teardown")
        except Exception:  # teardown must never mask the real run outcome
            continue
