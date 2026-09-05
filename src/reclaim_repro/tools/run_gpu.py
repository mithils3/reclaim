"""The GPU step — the reproduction agent's interactive GPU shell.

Everything else the agent does (clone, venv, installs, edits, inspecting data) is
cheap CPU work in ``workspace_bash``; ``run_gpu`` is the one tool that provisions a
GPU. The first call ACQUIRES one ``salloc`` allocation that **stays held**; every
later call runs into the same node (``slurm.run_in_session``) with no new queue
wait, so the agent installs torch, verifies CUDA, and runs the experiment as
successive ``run_gpu`` calls on one allocation instead of re-queueing each time.
The agent frees it with ``release=true`` the moment it is done; ``gpu_session``
also releases it at episode teardown.

Every step's output is *streamed* to ``evidence/gpu_step_<n>.log`` as it arrives
(``slurm.run_in_session`` tees it), so a step killed at the --time wall or our
timeout still leaves everything it printed on disk — and this tool returns the
tail instead of nothing. Blind re-runs of killed jobs are a large compute sink.

The tool is the budget meter's enforcement point. The held node is billed by
**wall clock** — ``gpus x (held seconds) x hw`` — because the GPU is reserved across
the agent's reasoning/install gaps, not only while a command runs:

* **before acquiring** -- ``budget.affordable`` refuses to start a session whose
  worst case (``gpus x minutes x hw``, the ``--time`` pre-authorization) would
  overspend the remaining H100-hour budget;
* **before reusing** -- a launch onto a held session with under
  ``STALE_LAUNCH_SECONDS`` of --time left rotates the session out (release +
  re-acquire) so the command lands on a fresh full-length hold instead of a doomed
  one — never a dead-end refusal that leaves the spent session bound;
* **after each step / on release** -- ``gpu_session.charge_accrued`` bills the wall
  held since the last charge (the loop guardrail charges it between rounds too, so a
  long reasoning gap on a held node still depletes the budget);
* **either way** -- one structured row lands in ``evidence/trajectory.jsonl`` and the
  command in ``evidence/commands.log`` so the auditor can re-trace what ran and cost.

When the charge drives the budget to zero the loop's guardrail releases the session
and force-finals the episode on the next round (``loop.apply_guardrails``).
"""

from __future__ import annotations

import os
import shlex
from pathlib import Path
from typing import Any

from reclaim_vllm.config.config import RUN_FILE_DEFAULT_CHARS, bounded, function_tool

from reclaim_repro import budget as budget_mod
from reclaim_repro import evidence as evidence_mod
from reclaim_repro import gpu_session, slurm
from reclaim_repro.context import ExecutionContext
from reclaim_repro.sandbox import CONTAINER_EVIDENCE
from reclaim_repro.tools import output as output_mod

# Defaults/bounds for the model-set knobs, applied on the call that *starts* a
# session. ``gpus`` is capped to the node by ``slurm.build_acquire``; ``minutes`` is
# the SLURM ``--time`` (the hold's lifetime and the budget pre-authorization).
DEFAULT_GPUS = 1
DEFAULT_MINUTES = 30
MAX_MINUTES = 24 * 60
# Bound an acquire that never returns (wedged in queue) without killing a legitimate
# long queue wait: wait the hold's own wall cap plus this grace before giving up.
#
# 8h is deliberately generous: on a congested batch pool a successful acquire can take
# many hours, so a shorter grace kills waits the queue would have granted. Queue wait is not billed (``ensure_session``
# starts the meter at grant), so a longer grace costs job wall clock, not H100-hours.
# Raise it with REPRO_QUEUE_GRACE_HOURS when the cluster is congested; lower it when a
# sweep is wall-bound and would rather fail fast onto another partition.
DEFAULT_QUEUE_GRACE_HOURS = 8.0


def queue_grace_seconds() -> float:
    """Grace past the hold's own ``--time`` before an un-granted acquire is given up on."""
    raw = os.environ.get("REPRO_QUEUE_GRACE_HOURS", "").strip()
    try:
        hours = float(raw) if raw else DEFAULT_QUEUE_GRACE_HOURS
    except ValueError:
        hours = DEFAULT_QUEUE_GRACE_HOURS  # a typo in a sweep's env must not wedge the run
    return max(0.0, hours) * 3600.0
# Chars of streamed output returned when the step was killed (wall/timeout): the
# tail is where the last checkpoint line / progress state / traceback is.
KILL_TAIL_CHARS = 4000
# Warn the model when the held allocation is within this many seconds of its --time
# wall: past it SLURM reclaims the node mid-step and any unsaved state is lost.
SESSION_WARN_SECONDS = 120
# Refuse to LAUNCH a new command on a session with less than this left on its
# --time. Runs that started multi-minute jobs into a sub-2-minute hold lost them;
# a launch that near the wall can only be killed, so this is a deterministic guard
# rather than a heuristic.
STALE_LAUNCH_SECONDS = 120


def run_gpu(arguments: dict[str, Any], ctx: ExecutionContext) -> dict[str, Any]:
    """Run one command on the held GPU allocation (acquiring/releasing as asked)."""
    for field, label in (("cluster", "cluster profile"), ("workspace", "workspace directory"), ("budget", "compute budget")):
        if getattr(ctx, field) is None:
            return {"ok": False, "tool": "run_gpu", "error": f"No {label} bound for this episode."}

    command = str(arguments.get("command") or "").strip()
    release_after = bool(arguments.get("release"))
    if not command:
        if release_after:
            return _release_only(ctx)
        return {"ok": False, "tool": "run_gpu", "error": "Missing GPU command to run."}

    cap = ctx.cluster.gpus_per_node
    gpus = bounded(arguments.get("gpus"), DEFAULT_GPUS, cap)
    minutes = bounded(arguments.get("minutes"), DEFAULT_MINUTES, MAX_MINUTES)
    note = clamp_note(arguments.get("gpus"), gpus, cap)

    # Pre-launch staleness rotation: a held session within STALE_LAUNCH_SECONDS of (or
    # past) its --time wall is spent — a command launched onto it can only be killed,
    # and past the wall SLURM has already reclaimed the node. Rotate it out (release +
    # re-acquire below) rather than refuse. The old refuse-and-keep path dead-ended:
    # the spent session stayed bound, so every later call refused too, and once past
    # the wall it never reached the lost-session detection in run_in_session — it sat
    # as a zombie at "~0s left" blocking all re-acquisition, and one batch burned ~80
    # rounds on exactly that. Budget still gates the fresh acquire below.
    rotate_note = None
    if ctx.session is not None:
        stale_remaining = ctx.session.minutes * 60 - gpu_session.held_seconds(ctx.session)
        if stale_remaining < STALE_LAUNCH_SECONDS:
            rotate_note = stale_rotation(ctx.session, max(0.0, stale_remaining), minutes)
            gpu_session.release(ctx, "stale-rotate")  # bill final sliver + scancel + clear
        else:
            note = reuse_note(arguments, ctx.session) or note

    # Acquire the session if none is held (or the spent one was just rotated out);
    # pre-authorize its worst-case hold first.
    if ctx.session is None:
        affordable, reason = budget_mod.affordable(ctx.budget, gpus, minutes, ctx.cluster.hw)
        if not affordable:
            _record(ctx, command, gpus, minutes, ctx.cluster.hw, step=None, cost=0.0, refused=reason)
            return _refused(ctx, command, reason)
        # partition picks the pool for THIS allocation (default = the cluster profile's);
        # discover the choices with list_partitions. Fixed until the session is released.
        partition = str(arguments.get("partition") or "").strip() or None
        session, err = gpu_session.ensure_session(
            ctx, gpus=gpus, minutes=minutes, partition=partition,
            timeout=minutes * 60 + queue_grace_seconds(),
        )
        if session is None:
            return {"ok": False, "tool": "run_gpu", "command": command, "error": f"could not acquire GPU allocation: {err}"}
        if rotate_note:  # surface the rotation (plus any clamp note) on the fresh hold
            note = f"{rotate_note} {note}" if note else rotate_note
    else:
        session = ctx.session

    log_path = evidence_mod.next_gpu_log(ctx.evidence) if ctx.evidence is not None else None
    step = slurm.run_in_session(
        ctx.workspace, command, jobid=session.jobid,
        timeout=session.minutes * 60 + 600, sandbox=ctx.sandbox, log_path=log_path,
    )
    log_ref = _log_ref(ctx, log_path)
    if slurm.session_lost(step):
        gpu_session.drop_lost(ctx)
        return {
            "ok": False,
            "tool": "run_gpu",
            "command": command,
            "error": "GPU session expired or was cancelled (hit --time or scancel); "
            "the next run_gpu call will start a fresh allocation.",
            "output_log": log_ref,
            "stdout_tail": output_mod.tail(step.stdout, KILL_TAIL_CHARS),
            "stderr": output_mod.tail(step.stderr, KILL_TAIL_CHARS),
            "remaining_h100_hours": round(ctx.budget.remaining(), 4),
        }

    cost = gpu_session.charge_accrued(ctx)
    held = gpu_session.held_seconds(session)
    session_remaining = max(0.0, session.minutes * 60 - held)
    _record(ctx, command, session.gpus, session.minutes, session.hw, step=step, cost=cost, held=held)
    if release_after:
        gpu_session.release(ctx, "agent")

    elide_note = f" — full output in {log_ref}" if log_ref else ""
    stdout, t_out = output_mod.shape(step.stdout, RUN_FILE_DEFAULT_CHARS, note=elide_note)
    stderr, t_err = output_mod.shape(step.stderr, RUN_FILE_DEFAULT_CHARS, note=elide_note)
    result = {
        "ok": step.ok,
        "tool": "run_gpu",
        "command": command,
        "gpus": session.gpus,
        "minutes": session.minutes,
        "hw": session.hw,
        "partition": session.partition,
        "returncode": step.returncode,
        "run_seconds": round(step.elapsed_s, 1),
        "held_seconds": round(held, 1),
        "session_remaining_seconds": 0.0 if release_after else round(session_remaining, 1),
        "cost_h100_hours": round(cost, 4),
        "remaining_h100_hours": round(ctx.budget.remaining(), 4),
        "budget_exhausted": ctx.budget.exhausted(),
        "session_released": release_after,
        "session_jobid": None if release_after else session.jobid,
        "output_log": log_ref,
        "stdout": stdout,
        "stderr": stderr,
        "truncated": t_out or t_err,
    }
    warning = None if release_after else expiry_warning(session, session_remaining)
    if warning:
        result["session_expiry_warning"] = warning
    if note:
        result["note"] = note
    return result


def _refused(ctx: ExecutionContext, command: str, reason: str) -> dict[str, Any]:
    return {
        "ok": False,
        "tool": "run_gpu",
        "command": command,
        "error": f"run_gpu refused: {reason}",
        "remaining_h100_hours": round(ctx.budget.remaining(), 4),
    }


def _log_ref(ctx: ExecutionContext, log_path: Path | None) -> str | None:
    """The step log's path *as the agent can reach it* (container path when sandboxed)."""
    if log_path is None:
        return None
    if ctx.sandbox is not None:
        return f"{CONTAINER_EVIDENCE}/{log_path.name}"
    return str(log_path)


def _release_only(ctx: ExecutionContext) -> dict[str, Any]:
    """Honor ``release=true`` with no command: free the held allocation, if any."""
    record = gpu_session.release(ctx, "agent")
    if record is None:
        return {"ok": True, "tool": "run_gpu", "note": "no GPU session was held.", "session_released": False}
    return {
        "ok": True,
        "tool": "run_gpu",
        "session_released": True,
        "held_seconds": record["held_seconds"],
        "cost_h100_hours": record["final_charge_h100_hours"],
        "remaining_h100_hours": round(ctx.budget.remaining(), 4) if ctx.budget else None,
    }


def _record(
    ctx: ExecutionContext,
    command: str,
    gpus: int,
    minutes: int,
    hw: str,
    *,
    step: slurm.StepResult | None,
    cost: float,
    held: float = 0.0,
    refused: str | None = None,
) -> None:
    """Append one trajectory row + a commands.log line for this step."""
    if ctx.evidence is None:
        return
    returncode = None if step is None else step.returncode
    row: dict[str, Any] = {
        "type": "run_gpu",
        "command": command,
        "gpus": gpus,
        "minutes": minutes,
        "hw": hw,
        "returncode": returncode,
        "run_seconds": None if step is None else round(step.elapsed_s, 1),
        "held_seconds": round(held, 1),
        "cost_h100_hours": round(cost, 4),
        "remaining_h100_hours": round(ctx.budget.remaining(), 4) if ctx.budget else None,
    }
    if refused is not None:
        row["refused"] = refused
    if step is not None:
        row["argv"] = " ".join(shlex.quote(a) for a in step.command)
    evidence_mod.append_trajectory(ctx.evidence, row)
    evidence_mod.log_command(
        ctx.evidence,
        f"run_gpu ({gpus} gpu x {minutes} min {hw}): {command}",
        returncode=returncode,
        cwd=ctx.workspace,
        duration_s=None if step is None else step.elapsed_s,
    )


RUN_GPU_HANDLERS = {"run_gpu": run_gpu}

# Message quality is load-bearing here: agents were observed burning sessions on
# exactly the semantics these spell out (``minutes`` on a reuse call
# does NOT extend the hold), so each message states the semantics and the recovery
# action, not just the failure.


def clamp_note(requested: Any, effective: int, cap: int) -> str | None:
    """Tell the agent when its requested GPU count was clamped to the node cap."""
    if requested in (None, ""):
        return None
    asked = _safe_int(requested)
    if asked is not None and asked != effective:
        return f"requested gpus={asked} clamped to {effective} (node capacity is {cap})."
    return None


def expiry_warning(session: Any, remaining_seconds: float) -> str | None:
    """Loud heads-up when the held allocation is about to hit its --time wall.

    Past the wall SLURM reclaims the node mid-step, so anything not written to disk
    (training state, in-memory results) is gone. Surfaced so the model checkpoints and
    re-acquires a fresh/longer hold *before* it loses the node, not after.
    """
    if remaining_seconds > SESSION_WARN_SECONDS:
        return None
    return (
        f"SESSION ENDING: ~{remaining_seconds / 60:.1f} min left on this {session.minutes}-min "
        f"allocation (jobid {session.jobid}) before SLURM reclaims the node and loses any unsaved "
        "state. Save results/checkpoints to disk now. Long work left? release=true and start a "
        "fresh session with a larger minutes= (or finish what fits in the time remaining)."
    )


def stale_rotation(session: Any, remaining_seconds: float, new_minutes: int) -> str:
    """Explain that a spent held session was auto-released and a fresh one acquired.

    A session within ``STALE_LAUNCH_SECONDS`` of (or past) its ``--time`` wall can only
    run doomed commands, so ``run_gpu`` rotates it out — release + re-acquire in the
    same call — instead of refusing every launch until the agent manually releases.
    (Refuse-and-keep dead-ended: the spent session stayed bound, so once past its wall
    it sat as a zombie blocking all re-acquisition.) States that ``minutes`` is fixed
    per allocation so the agent sizes the next hold to the whole job.
    """
    return (
        f"prior session (jobid {session.jobid}) had only ~{remaining_seconds:.0f}s of its "
        f"{session.minutes}-min --time left, so it was released and a fresh {new_minutes}-min "
        "allocation was acquired for this command. minutes= is fixed per allocation — this new "
        f"hold lasts {new_minutes} min from now; size minutes= to the whole job to avoid "
        "mid-run rotations."
    )


def reuse_note(arguments: dict[str, Any], session: Any) -> str | None:
    """Warn when gpus/minutes are passed to a call that reuses a live session."""
    asked_gpus = arguments.get("gpus")
    if asked_gpus not in (None, "") and _safe_int(asked_gpus) not in (None, session.gpus):
        return (
            f"a GPU session is already held ({session.gpus} gpu, jobid {session.jobid}); "
            "gpus/minutes are fixed until you release it (run_gpu release=true) and start a new one."
        )
    return None


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def run_gpu_tool(gpus_per_node: int) -> dict:
    """Build the ``run_gpu`` schema, advertising this cluster's per-node GPU cap."""
    return function_tool(
        "run_gpu",
        "Run ONE command on a real GPU (training, evaluation, scoring, verifying the "
        "container's torch.cuda, nvidia-smi). The GPU allocation is HELD across calls: the "
        "first run_gpu acquires it (you may wait in the queue once) and every later "
        "run_gpu runs on the SAME node with NO new queue wait, so install → verify → "
        "run as successive calls. You are billed WALL-CLOCK for the whole time the "
        "allocation is held — gpus x held-time x hw, including while you reason or "
        "install between commands — so set release=true the moment you are done with "
        "the GPU to stop the meter (re-acquire later if you need it again). The command "
        "runs with the workspace as its cwd; cost and remaining budget are returned and "
        "recorded to evidence/. Each step's FULL stdout/stderr is streamed to the file named "
        "in the result's output_log (under /repro/evidence/) as it runs — it survives even if "
        "the step is killed, so read/grep that file instead of re-running a job just to see "
        "its output. Each result also reports session_remaining_seconds — the "
        "wall left before this allocation hits its `minutes` (--time) cap and SLURM reclaims "
        "the node (losing any unsaved state); when it runs low a session_expiry_warning tells "
        "you to checkpoint to disk and, if you need more time, release and re-acquire. If you "
        "launch a command onto a hold that is already within ~2 min of (or past) its --time "
        "wall, run_gpu auto-rotates: it releases the spent allocation and acquires a fresh one "
        "sized to this call's minutes=, then runs your command on it (reported in the result's "
        "note) — so a spent session never blocks you.",
        {
            "command": {
                "type": "string",
                "description": "The GPU command to run (e.g. `python train.py ...`). Omit only with release=true to just free the session.",
            },
            "gpus": {
                "type": "integer",
                "default": DEFAULT_GPUS,
                "minimum": 1,
                "maximum": gpus_per_node,
                "description": (
                    f"GPUs to hold (1-{gpus_per_node}, one node); set only on the call that STARTS "
                    "the session. Default to 1 — many real runs are correctly single-GPU and a 1-GPU "
                    "FULL run is a real run, not a smoke test. Take more ONLY when the model/batch "
                    "won't fit in one GPU's memory or a parallelizable run won't finish in your "
                    "wall-clock; throughput then scales with gpus (wall budget ~same)."
                ),
            },
            "minutes": {
                "type": "integer",
                "default": DEFAULT_MINUTES,
                "minimum": 1,
                "maximum": MAX_MINUTES,
                "description": "Max lifetime of the held allocation (SLURM --time) and the budget "
                "pre-authorization; set on the call that STARTS the session. Pick ~ how long you "
                "will hold it. IGNORED on a reuse call — it cannot extend a held session; to get "
                "more time, release=true and re-acquire with a larger value.",
            },
            "release": {
                "type": "boolean",
                "default": False,
                "description": "Set true when you are done with the GPU: frees the allocation after "
                "this command (or immediately if no command) so wall-clock billing stops.",
            },
            "partition": {
                "type": "string",
                "description": "SLURM partition (node pool) to allocate on. Omit to use this "
                "cluster's default; call list_partitions to see the alternatives (e.g. a "
                "faster-queueing interactive pool). Takes effect only on the call that STARTS "
                "the session; fixed until you release it.",
            },
        },
        [],  # command is enforced by the handler (it is optional only with release=true)
    )
