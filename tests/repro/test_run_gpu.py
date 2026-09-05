from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from reclaim_repro import evidence
from reclaim_repro.cluster import GPU_PARTITION, resolve_cluster
from reclaim_repro.context import Budget, ExecutionContext, GpuSession
from reclaim_repro.slurm import SessionHandle, StepResult
from reclaim_repro.tools import build_repro_tools, execute_repro_tool_call
from reclaim_repro.tools.run_gpu import run_gpu, run_gpu_tool


def _ctx(root: Path, *, budget_hours: float = 8.0) -> ExecutionContext:
    ws, ref, ev = root / "workspace", root / "reference", root / "evidence"
    for p in (ws, ref, ev):
        p.mkdir(parents=True, exist_ok=True)
    evidence.init_evidence(ev)
    return ExecutionContext(
        arxiv_id="x",
        workspace=ws,
        reference=ref,
        evidence=ev,
        budget=Budget(total_h100_hours=budget_hours),
        cluster=resolve_cluster(),
    )


def _handle(jobid: str | None = "555", *, ok: bool = True, stderr: str = "") -> SessionHandle:
    return SessionHandle(ok=ok, jobid=jobid, stderr=stderr)


def _step(stdout: str = "ok", stderr: str = "", *, rc: int = 0, elapsed: float = 1.0) -> StepResult:
    return StepResult(ok=rc == 0, returncode=rc, stdout=stdout, stderr=stderr, elapsed_s=elapsed, command=["srun"])


def _patch(*, acquire=None, run=None, release=None):
    """Patch the three slurm seams the session lifecycle calls (shared module attrs)."""
    return (
        mock.patch("reclaim_repro.slurm.acquire_session", return_value=acquire or _handle()),
        mock.patch("reclaim_repro.slurm.run_in_session", return_value=run if run is not None else _step()),
        mock.patch("reclaim_repro.slurm.release_session", side_effect=release or (lambda *_: None)),
    )


class SessionLifecycleTests(unittest.TestCase):
    def test_first_call_acquires_session_then_runs_into_it(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = _ctx(Path(d))
            acq, run, rel = _patch(acquire=_handle("555"), run=_step("torch 2.11"))
            with acq as a, run as r, rel:
                res = run_gpu({"command": "python -c 'import torch'", "gpus": 1, "minutes": 30}, ctx)
            self.assertTrue(res["ok"], res)
            self.assertEqual(res["session_jobid"], "555")
            self.assertFalse(res["session_released"])
            self.assertEqual(ctx.session.jobid, "555")
            a.assert_called_once()
            r.assert_called_once()
            self.assertIn("torch 2.11", res["stdout"])

    def test_partition_arg_is_passed_to_acquire_and_reported(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = _ctx(Path(d))
            acq, run, rel = _patch(acquire=_handle("777"))
            with acq as a, run, rel:
                res = run_gpu(
                    {"command": "python eval.py", "partition": "gpu-interactive"}, ctx
                )
            self.assertEqual(a.call_args.kwargs["partition"], "gpu-interactive")
            self.assertEqual(res["partition"], "gpu-interactive")
            self.assertEqual(ctx.session.partition, "gpu-interactive")

    def test_default_partition_is_the_profile_default(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = _ctx(Path(d))
            acq, run, rel = _patch(acquire=_handle("778"))
            with acq as a, run, rel:
                res = run_gpu({"command": "python eval.py"}, ctx)
            self.assertIsNone(a.call_args.kwargs["partition"])  # nothing forced
            self.assertEqual(res["partition"], GPU_PARTITION)  # session records the profile default

    def test_second_call_reuses_allocation_without_reacquiring(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = _ctx(Path(d))
            acq, run, rel = _patch(acquire=_handle("555"))
            with acq as a, run as r, rel:
                run_gpu({"command": "python a.py"}, ctx)
                res = run_gpu({"command": "python b.py"}, ctx)
            a.assert_called_once()  # acquired once, reused for the second step
            self.assertEqual(r.call_count, 2)
            self.assertEqual(res["session_jobid"], "555")
            # srun ran into the held jobid, not a fresh allocation.
            self.assertEqual(r.call_args.kwargs["jobid"], "555")

    def test_release_true_frees_the_allocation_after_the_command(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = _ctx(Path(d))
            acq, run, rel = _patch(acquire=_handle("555"))
            with acq, run, rel as scancel:
                run_gpu({"command": "python a.py"}, ctx)
                res = run_gpu({"command": "python score.py", "release": True}, ctx)
            self.assertTrue(res["session_released"])
            self.assertIsNone(ctx.session)
            scancel.assert_called_once_with("555")

    def test_release_only_with_no_command_frees_session(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = _ctx(Path(d))
            acq, run, rel = _patch(acquire=_handle("555"))
            with acq, run as r, rel as scancel:
                run_gpu({"command": "python a.py"}, ctx)
                res = run_gpu({"release": True}, ctx)
            self.assertTrue(res["ok"], res)
            self.assertTrue(res["session_released"])
            self.assertEqual(r.call_count, 1)  # the bare release ran no srun
            scancel.assert_called_once_with("555")

    def test_lost_session_is_surfaced_and_cleared(self):
        lost = _step(stderr="srun: error: Unable to confirm allocation for job 555: Invalid job id", rc=1)
        with tempfile.TemporaryDirectory() as d:
            ctx = _ctx(Path(d))
            acq, run, rel = _patch(acquire=_handle("555"), run=lost)
            with acq, run, rel:
                res = run_gpu({"command": "python a.py"}, ctx)
            self.assertFalse(res["ok"])
            self.assertIn("expired", res["error"])
            self.assertIsNone(ctx.session)  # dropped so the next call re-acquires


class StalenessGuardTests(unittest.TestCase):
    def _held_ctx(self, d: Path, *, minutes: int, held_seconds: float) -> ExecutionContext:
        ctx = _ctx(d)
        now = time.monotonic()
        ctx.session = GpuSession(
            jobid="555", gpus=1, minutes=minutes, hw="h100",
            started=now - held_seconds, last_charged=now, partition=GPU_PARTITION,
        )
        return ctx

    def test_rotates_out_a_nearly_expired_session(self):
        with tempfile.TemporaryDirectory() as d:
            # 30-min hold with ~29.5 min already held -> ~30s left, under the guard.
            ctx = self._held_ctx(Path(d), minutes=30, held_seconds=30 * 60 - 30)
            acq, run, rel = _patch(acquire=_handle("999"))
            with acq as a, run as r, rel as scancel:
                res = run_gpu({"command": "python train.py", "minutes": 120}, ctx)
            # The spent session is released and a fresh one acquired; the command runs
            # on the new hold rather than dead-ending.
            self.assertTrue(res["ok"], res)
            scancel.assert_called_once_with("555")  # old hold torn down
            a.assert_called_once()  # fresh allocation acquired
            r.assert_called_once()  # command ran on the fresh hold
            self.assertEqual(ctx.session.jobid, "999")
            self.assertEqual(res["session_jobid"], "999")
            self.assertEqual(res["minutes"], 120)  # sized to this call's minutes=
            self.assertIn("released", res["note"])

    def test_zombie_session_past_its_wall_is_rotated_not_deadlocked(self):
        # Regression for the stale-session deadlock: a 5-min hold held past its --time
        # wall (SLURM already reclaimed it) must not sit bound at "~0s left" refusing
        # every call — it must rotate out and re-acquire so work can continue.
        with tempfile.TemporaryDirectory() as d:
            ctx = self._held_ctx(Path(d), minutes=5, held_seconds=30 * 60)
            acq, run, rel = _patch(acquire=_handle("999"))
            with acq as a, run as r, rel as scancel:
                res = run_gpu({"command": "python eval.py", "minutes": 30}, ctx)
            self.assertTrue(res["ok"], res)
            scancel.assert_called_once_with("555")
            a.assert_called_once()
            r.assert_called_once()
            self.assertEqual(ctx.session.jobid, "999")

    def test_fresh_session_still_runs(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = self._held_ctx(Path(d), minutes=30, held_seconds=10.0)
            acq, run, rel = _patch()
            with acq, run as r, rel:
                res = run_gpu({"command": "python train.py"}, ctx)
            self.assertTrue(res["ok"], res)
            r.assert_called_once()

    def test_rotate_that_cannot_afford_reacquire_clears_the_session(self):
        # Even when the fresh acquire is unaffordable, the spent hold must be released
        # (scancel + cleared), so the refusal is a clean budget stop — not a zombie
        # left bound to deadlock the next call.
        with tempfile.TemporaryDirectory() as d:
            ctx = self._held_ctx(Path(d), minutes=30, held_seconds=30 * 60 - 30)
            ctx.budget = Budget(total_h100_hours=0.01)  # cannot afford a fresh hold
            acq, run, rel = _patch(acquire=_handle("999"))
            with acq as a, run as r, rel as scancel:
                res = run_gpu({"command": "python train.py", "minutes": 120}, ctx)
            self.assertFalse(res["ok"])
            self.assertIn("refused", res["error"])
            scancel.assert_called_once_with("555")  # spent hold torn down
            a.assert_not_called()  # never acquired (unaffordable)
            r.assert_not_called()
            self.assertIsNone(ctx.session)  # cleared -> next call retries cleanly

    def test_bare_release_still_works_on_a_stale_session(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = self._held_ctx(Path(d), minutes=30, held_seconds=30 * 60 - 30)
            acq, run, rel = _patch()
            with acq, run, rel as scancel:
                res = run_gpu({"release": True}, ctx)
            self.assertTrue(res["ok"], res)
            self.assertTrue(res["session_released"])
            scancel.assert_called_once_with("555")


class OutputPersistenceTests(unittest.TestCase):
    def test_step_log_path_is_allocated_and_passed_to_srun(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = _ctx(Path(d))
            acq, run, rel = _patch()
            with acq, run as r, rel:
                res = run_gpu({"command": "python a.py"}, ctx)
                first = r.call_args_list[0].kwargs["log_path"]
                self.assertEqual(first.name, "gpu_step_0000.log")
                self.assertEqual(first.parent, ctx.evidence)
                # once a step's log exists on disk, the next step gets the next seq
                first.write_text("out")
                run_gpu({"command": "python b.py"}, ctx)
                second = r.call_args_list[1].kwargs["log_path"]
                self.assertEqual(second.name, "gpu_step_0001.log")
            # no sandbox in tests -> the agent-facing ref is the host path
            self.assertEqual(res["output_log"], str(first))

    def test_lost_session_returns_the_streamed_tail(self):
        lost = _step(
            stdout="epoch 1\n" * 500 + "last checkpoint saved: ckpt_9.pt\n",
            stderr="srun: error: Slurm job 555 has expired",
            rc=1,
        )
        with tempfile.TemporaryDirectory() as d:
            ctx = _ctx(Path(d))
            acq, run, rel = _patch(run=lost)
            with acq, run, rel:
                res = run_gpu({"command": "python train.py"}, ctx)
            self.assertFalse(res["ok"])
            self.assertIn("ckpt_9.pt", res["stdout_tail"])  # the tail survives the kill
            self.assertIn("output_log", res)

    def test_progress_spam_is_stripped_and_result_line_kept(self):
        spam = "".join(f"\r {p}%|██| {p}/100" for p in range(100))
        noisy = _step(stdout=f"start\n{spam}\nfinal accuracy: 0.913\n")
        with tempfile.TemporaryDirectory() as d:
            ctx = _ctx(Path(d))
            acq, run, rel = _patch(run=noisy)
            with acq, run, rel:
                res = run_gpu({"command": "python eval.py"}, ctx)
            self.assertIn("final accuracy: 0.913", res["stdout"])
            self.assertNotIn("1%", res["stdout"])  # intermediate frames collapsed
            self.assertFalse(res["truncated"])


class QueueGraceTests(unittest.TestCase):
    """How long an un-granted acquire is allowed to sit in the queue.

    The bound handed to ``acquire_session`` is the hold's own ``--time`` plus this
    grace. A 4h grace sat inside the observed tail (successful acquires have reached
    6.2h), so ordinary asks were being killed mid-queue.
    """

    def _timeout_for(self, minutes: int) -> float:
        with tempfile.TemporaryDirectory() as d:
            ctx = _ctx(Path(d))
            acq, run, rel = _patch()
            with acq as a, run, rel:
                run_gpu({"command": "python train.py", "minutes": minutes}, ctx)
            return a.call_args.kwargs["timeout"]

    def test_default_grace_is_eight_hours_past_the_hold(self):
        self.assertEqual(self._timeout_for(120), 120 * 60 + 8 * 3600)

    def test_env_override_widens_the_grace(self):
        with mock.patch.dict(os.environ, {"REPRO_QUEUE_GRACE_HOURS": "14"}):
            self.assertEqual(self._timeout_for(120), 120 * 60 + 14 * 3600)

    def test_env_override_accepts_fractional_hours(self):
        with mock.patch.dict(os.environ, {"REPRO_QUEUE_GRACE_HOURS": "0.5"}):
            self.assertEqual(self._timeout_for(60), 60 * 60 + 1800)

    def test_garbage_env_falls_back_to_the_default(self):
        # A typo in a sweep's export must not wedge a 48h job.
        for bad in ("abc", "", "   "):
            with mock.patch.dict(os.environ, {"REPRO_QUEUE_GRACE_HOURS": bad}):
                self.assertEqual(self._timeout_for(60), 60 * 60 + 8 * 3600)

    def test_negative_grace_clamps_to_zero(self):
        with mock.patch.dict(os.environ, {"REPRO_QUEUE_GRACE_HOURS": "-3"}):
            self.assertEqual(self._timeout_for(60), 60 * 60)


class RunGpuGuardrailTests(unittest.TestCase):
    def test_refuses_to_start_a_session_that_overspends_and_does_not_acquire(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = _ctx(Path(d), budget_hours=1.0)
            acq, run, rel = _patch()
            with acq as a, run, rel:
                res = run_gpu({"command": "python train.py", "gpus": 4, "minutes": 60}, ctx)
            a.assert_not_called()  # refused before any allocation
            self.assertFalse(res["ok"])
            self.assertIn("refused", res["error"])
            self.assertEqual(ctx.budget.spent_h100_hours, 0.0)
            self.assertIn('"refused"', (ctx.evidence / "trajectory.jsonl").read_text())

    def test_acquire_failure_is_reported(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = _ctx(Path(d))
            acq, run, rel = _patch(acquire=_handle(None, ok=False, stderr="salloc: error: out of nodes"))
            with acq, run, rel:
                res = run_gpu({"command": "python a.py"}, ctx)
            self.assertFalse(res["ok"])
            self.assertIn("could not acquire", res["error"])
            self.assertIsNone(ctx.session)

    def test_missing_command_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = _ctx(Path(d))
            self.assertFalse(run_gpu({"command": "  "}, ctx)["ok"])

    def test_records_trajectory_and_command_log(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = _ctx(Path(d))
            acq, run, rel = _patch()
            with acq, run, rel:
                run_gpu({"command": "python train.py", "minutes": 5}, ctx)
            traj = (ctx.evidence / "trajectory.jsonl").read_text()
            self.assertIn('"type": "run_gpu"', traj)
            self.assertIn("python train.py", traj)
            self.assertIn("run_gpu", (ctx.evidence / "commands.log").read_text())


class GpuChoiceTests(unittest.TestCase):
    def test_schema_max_gpus_tracks_node_capacity(self):
        for cap in (4, 8):
            schema = run_gpu_tool(cap)["function"]["parameters"]["properties"]["gpus"]
            self.assertEqual(schema["maximum"], cap)
            self.assertEqual(schema["minimum"], 1)
        tool = next(t for t in build_repro_tools(8) if t["function"]["name"] == "run_gpu")
        self.assertEqual(tool["function"]["parameters"]["properties"]["gpus"]["maximum"], 8)

    def test_release_param_is_advertised(self):
        props = run_gpu_tool(4)["function"]["parameters"]["properties"]
        self.assertIn("release", props)
        self.assertEqual(props["release"]["type"], "boolean")

    def test_agent_gpu_choice_sizes_the_session(self):
        captured = {}

        def fake_acquire(cluster, *, gpus, minutes, timeout=None, partition=None):
            captured["gpus"] = gpus
            return _handle("555")

        with tempfile.TemporaryDirectory() as d:
            ctx = _ctx(Path(d))  # profile: 4 GPU/node
            with mock.patch("reclaim_repro.slurm.acquire_session", side_effect=fake_acquire), \
                 mock.patch("reclaim_repro.slurm.run_in_session", return_value=_step()), \
                 mock.patch("reclaim_repro.slurm.release_session"):
                res = run_gpu({"command": "python train.py", "gpus": 3, "minutes": 10}, ctx)
            self.assertEqual(captured["gpus"], 3)
            self.assertEqual(res["gpus"], 3)
            self.assertNotIn("note", res)

    def test_over_capacity_request_is_clamped_and_noted(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = _ctx(Path(d))  # profile: cap 4
            acq, run, rel = _patch()
            with acq, run, rel:
                res = run_gpu({"command": "python train.py", "gpus": 9, "minutes": 10}, ctx)
            self.assertEqual(res["gpus"], 4)
            self.assertIn("clamped", res["note"])


class DispatchTests(unittest.TestCase):
    def test_run_gpu_is_advertised_and_routed(self):
        names = {t["function"]["name"] for t in build_repro_tools(4)}
        self.assertEqual(
            names,
            {"workspace_bash", "write_file", "edit_file", "update_plan", "fetch_url", "list_partitions", "run_gpu"},
        )

    def test_execute_routes_run_gpu_through_context(self):
        call = {"function": {"name": "run_gpu", "arguments": {"command": "python x.py", "minutes": 5}}}
        with tempfile.TemporaryDirectory() as d:
            ctx = _ctx(Path(d))
            acq, run, rel = _patch()
            with acq, run, rel:
                res = execute_repro_tool_call(call, ctx)
            self.assertTrue(res["ok"], res)
            self.assertEqual(res["tool"], "run_gpu")

    def test_unknown_tool_reports_available(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = _ctx(Path(d))
            res = execute_repro_tool_call({"function": {"name": "nope", "arguments": {}}}, ctx)
            self.assertFalse(res["ok"])
            self.assertIn("run_gpu", res["error"])


if __name__ == "__main__":
    unittest.main()
