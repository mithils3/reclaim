from __future__ import annotations

import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from reclaim_repro import evidence, gpu_session
from reclaim_repro.cluster import resolve_cluster
from reclaim_repro.context import Budget, ExecutionContext, GpuSession
from reclaim_repro.slurm import SessionHandle, SlurmConfigError


def _ctx(root: Path, *, budget_hours: float = 8.0) -> ExecutionContext:
    ev = root / "evidence"
    ev.mkdir(parents=True, exist_ok=True)
    evidence.init_evidence(ev)
    return ExecutionContext(
        arxiv_id="x",
        workspace=root,
        evidence=ev,
        budget=Budget(total_h100_hours=budget_hours),
        cluster=resolve_cluster(),
    )


def _held(ctx: ExecutionContext, *, gpus: int, seconds_ago: float) -> GpuSession:
    now = time.monotonic()
    ctx.session = GpuSession(
        jobid="555", gpus=gpus, minutes=60, hw="gh200",
        started=now - seconds_ago, last_charged=now - seconds_ago,
    )
    return ctx.session


class ChargeAccruedTests(unittest.TestCase):
    def test_charges_wall_clock_held_since_last_charge(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = _ctx(Path(d))
            _held(ctx, gpus=2, seconds_ago=30.0)  # 2 GPUs held ~30s
            cost = gpu_session.charge_accrued(ctx)
            # 2 gpu x (30/3600) h x 1.0 (gh200) = 0.016667 H100-h.
            self.assertAlmostEqual(cost, 2 * 30 / 3600, delta=1e-3)
            self.assertAlmostEqual(ctx.budget.spent_h100_hours, 2 * 30 / 3600, delta=1e-3)

    def test_second_charge_immediately_after_is_near_zero(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = _ctx(Path(d))
            _held(ctx, gpus=1, seconds_ago=10.0)
            gpu_session.charge_accrued(ctx)
            again = gpu_session.charge_accrued(ctx)  # no new wall held -> ~0, never double-counts
            self.assertAlmostEqual(again, 0.0, delta=1e-3)

    def test_no_session_is_a_noop(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = _ctx(Path(d))
            self.assertEqual(gpu_session.charge_accrued(ctx), 0.0)
            self.assertEqual(ctx.budget.spent_h100_hours, 0.0)


class EnsureAndReleaseTests(unittest.TestCase):
    def test_ensure_acquires_sets_session_and_starts_the_clock(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = _ctx(Path(d))
            handle = SessionHandle(ok=True, jobid="77", stderr="")
            with mock.patch("reclaim_repro.slurm.acquire_session", return_value=handle):
                session, err = gpu_session.ensure_session(ctx, gpus=2, minutes=30)
            self.assertIsNone(err)
            self.assertEqual(session.jobid, "77")
            self.assertAlmostEqual(session.started, session.last_charged, delta=1e-3)

    def test_ensure_reuses_an_existing_session(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = _ctx(Path(d))
            held = _held(ctx, gpus=1, seconds_ago=5.0)
            with mock.patch("reclaim_repro.slurm.acquire_session") as acquire:
                session, err = gpu_session.ensure_session(ctx, gpus=4, minutes=30)
            acquire.assert_not_called()
            self.assertIs(session, held)
            self.assertIsNone(err)

    def test_ensure_reports_a_failed_acquire(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = _ctx(Path(d))
            handle = SessionHandle(ok=False, jobid=None, stderr="salloc: error: boom")
            with mock.patch("reclaim_repro.slurm.acquire_session", return_value=handle):
                session, err = gpu_session.ensure_session(ctx, gpus=1, minutes=5)
            self.assertIsNone(session)
            self.assertIn("boom", err)

    def test_ensure_turns_a_config_error_into_a_clean_failure(self):
        # A SlurmConfigError from the argv builder must not crash the loop: it
        # surfaces as an acquire failure carrying the same message.
        with tempfile.TemporaryDirectory() as d:
            ctx = _ctx(Path(d))
            with mock.patch(
                "reclaim_repro.slurm.acquire_session",
                side_effect=SlurmConfigError("cluster 'bare' has no account/partition"),
            ):
                session, err = gpu_session.ensure_session(ctx, gpus=1, minutes=5)
            self.assertIsNone(session)
            self.assertIn("no account/partition", err)

    def test_release_charges_scancels_clears_and_records(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = _ctx(Path(d))
            _held(ctx, gpus=1, seconds_ago=20.0)
            with mock.patch("reclaim_repro.slurm.release_session") as scancel:
                record = gpu_session.release(ctx, "agent")
            scancel.assert_called_once_with("555")
            self.assertIsNone(ctx.session)
            self.assertAlmostEqual(ctx.budget.spent_h100_hours, 20 / 3600, delta=1e-3)
            self.assertEqual(record["reason"], "agent")
            self.assertIn('"type": "gpu_release"', (ctx.evidence / "trajectory.jsonl").read_text())

    def test_release_without_a_session_is_a_noop(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = _ctx(Path(d))
            self.assertIsNone(gpu_session.release(ctx))

    def test_teardown_all_releases_each_held_session(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = _ctx(Path(d) / "a"), _ctx(Path(d) / "b")
            _held(a, gpus=1, seconds_ago=1.0)
            with mock.patch("reclaim_repro.slurm.release_session") as scancel:
                gpu_session.teardown_all([a, b])  # b holds nothing -> skipped
            scancel.assert_called_once_with("555")
            self.assertIsNone(a.session)


if __name__ == "__main__":
    unittest.main()
