from __future__ import annotations

import shlex
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from reclaim_repro.cluster import GPU_ACCOUNT, GPU_PARTITION, Cluster, resolve_cluster
from reclaim_repro.slurm import (
    STREAM_KEEP_BYTES,
    SlurmConfigError,
    StepResult,
    acquire_session,
    build_acquire,
    build_srun,
    release_session,
    run_in_session,
    session_lost,
)
from reclaim_repro.tools import output


class BuildAcquireTests(unittest.TestCase):
    def test_builds_a_held_salloc_for_the_profile(self):
        argv = build_acquire(resolve_cluster(), gpus=4, minutes=90)
        self.assertEqual(argv[0], "salloc")
        # --no-shell holds the allocation and returns once granted (no step runs yet).
        self.assertIn("--no-shell", argv)
        self.assertIn("-A", argv)
        self.assertIn(GPU_ACCOUNT, argv)
        self.assertIn("-p", argv)
        self.assertIn(GPU_PARTITION, argv)
        self.assertIn("--gpus=4", argv)
        self.assertIn("--time=90", argv)
        # Host RAM/CPU scale with the GPU count (100 GB + 60 cores per GPU);
        # --cpus-per-gpu is per-GPU so it stays 60 regardless of the count.
        self.assertIn("--mem=400G", argv)
        self.assertIn("--cpus-per-gpu=60", argv)
        # Acquire holds the node only; the command is spliced in later by build_srun.
        self.assertNotIn("srun", argv)

    def test_partition_override_replaces_the_profile_default(self):
        # The model picks a pool (from list_partitions) without touching the account.
        argv = build_acquire(
            resolve_cluster(), gpus=1, minutes=10, partition="gpu-interactive"
        )
        self.assertIn("gpu-interactive", argv)
        self.assertNotIn(GPU_PARTITION, argv)  # default not used
        self.assertIn(GPU_ACCOUNT, argv)  # account still the profile's

    def test_partition_override_satisfies_a_profile_without_a_default(self):
        # A bare cluster (no pinned partition) is allocatable once the model names one.
        bare = Cluster(name="bare", hw="h100", gpus_per_node=1, account="acct")
        argv = build_acquire(bare, gpus=1, minutes=5, partition="some-pool")
        self.assertIn("some-pool", argv)

    def test_rejects_profile_without_account(self):
        bare = Cluster(name="bare", hw="h100", gpus_per_node=1)
        # A domain error (not SystemExit): library code must not exit the process.
        with self.assertRaises(SlurmConfigError):
            build_acquire(bare, gpus=1, minutes=1)

    def test_rejects_gpus_over_node_capacity(self):
        with self.assertRaises(SlurmConfigError):
            build_acquire(resolve_cluster(), gpus=8, minutes=1)


class BuildSrunTests(unittest.TestCase):
    def test_runs_into_held_jobid_bare_without_sandbox(self):
        argv = build_srun("/ws", "python train.py", jobid="100001")
        self.assertEqual(argv[0], "srun")
        self.assertIn("--jobid=100001", argv)
        self.assertIn("--ntasks=1", argv)
        inner = argv[-1]
        # No sandbox passed -> bare body; CUDA comes from the image at runtime, not a
        # host `module load`, so the payload carries neither.
        self.assertNotIn("apptainer", inner)
        self.assertNotIn("module load", inner)
        self.assertEqual(inner, "cd /ws && python train.py")

    def test_sandbox_splices_apptainer_after_srun(self):
        from reclaim_repro.sandbox import CONTAINER_WORKSPACE, Bind, Sandbox

        sb = Sandbox(image="/img.sif", binds=(Bind("/host/ws", CONTAINER_WORKSPACE),))
        argv = build_srun("/host/ws", "python train.py", jobid="42", sandbox=sb)
        # srun (the trusted launcher) stays outside; the apptainer wrap is spliced after,
        # and the payload cd's to the short container workdir.
        self.assertEqual(argv[0], "srun")
        self.assertIn("--jobid=42", argv)
        self.assertIn("apptainer", argv)
        self.assertIn("--nv", argv)  # GPU step
        self.assertEqual(argv[-3:], ["bash", "-lc", "cd /repro/workspace && python train.py"])


class AcquireSessionTests(unittest.TestCase):
    def test_parses_granted_jobid(self):
        granted = subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout="",
            stderr="salloc: Pending job allocation 100001\nsalloc: Granted job allocation 100001\n",
        )
        with mock.patch("reclaim_repro.slurm.subprocess.run", return_value=granted) as run:
            handle = acquire_session(resolve_cluster(), gpus=1, minutes=5)
        self.assertTrue(handle.ok, handle.stderr)
        self.assertEqual(handle.jobid, "100001")
        self.assertEqual(run.call_args.args[0][0], "salloc")

    def test_failed_acquire_has_no_jobid(self):
        failed = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="salloc: error: Unable to allocate resources\n"
        )
        with mock.patch("reclaim_repro.slurm.subprocess.run", return_value=failed):
            handle = acquire_session(resolve_cluster(), gpus=1, minutes=5)
        self.assertFalse(handle.ok)
        self.assertIsNone(handle.jobid)

    def test_queue_timeout_with_byte_output_reports_the_timeout(self):
        """A starved queue must surface as "salloc timed out", not TypeError.

        ``subprocess.run(text=True)`` decodes only on the normal return path — the
        output hung on a raised ``TimeoutExpired`` is still bytes. Concatenating that
        onto a str used to raise ``TypeError: can't concat str to bytes``, which
        escaped ``run_gpu`` entirely: 45 calls across 14 runs of the Qwen3.6-27B
        medium sweep died this way, each burning its full ``minutes + 4h`` queue
        grace and returning no reason at all.
        """
        timed_out = subprocess.TimeoutExpired(
            cmd=["salloc"], timeout=21600.0,
            output=b"salloc: Pending job allocation 100002\n",
            stderr=b"salloc: job queued and waiting for resources\n",
        )
        with mock.patch("reclaim_repro.slurm.subprocess.run", side_effect=timed_out):
            handle = acquire_session(
                resolve_cluster(), gpus=1, minutes=120, timeout=21600.0
            )
        self.assertFalse(handle.ok)
        self.assertIsNone(handle.jobid)
        self.assertIn("salloc timed out", handle.stderr)
        self.assertIn("21600s", handle.stderr)
        # The queue's own explanation survives, so the agent can tell a starved
        # queue from a broken command instead of blind-retrying the step.
        self.assertIn("job queued and waiting for resources", handle.stderr)
        self.assertIn("Pending job allocation 100002", handle.stderr)

    def test_queue_timeout_accepts_str_output_too(self):
        """Same path when the platform hands back already-decoded text."""
        timed_out = subprocess.TimeoutExpired(
            cmd=["salloc"], timeout=300.0, output="", stderr="salloc: still pending\n"
        )
        with mock.patch("reclaim_repro.slurm.subprocess.run", side_effect=timed_out):
            handle = acquire_session(resolve_cluster(), gpus=1, minutes=5, timeout=300.0)
        self.assertFalse(handle.ok)
        self.assertIn("salloc timed out", handle.stderr)
        self.assertIn("still pending", handle.stderr)

    def test_queue_timeout_with_no_output_is_still_readable(self):
        timed_out = subprocess.TimeoutExpired(cmd=["salloc"], timeout=60.0)
        with mock.patch("reclaim_repro.slurm.subprocess.run", side_effect=timed_out):
            handle = acquire_session(resolve_cluster(), gpus=1, minutes=5, timeout=60.0)
        self.assertFalse(handle.ok)
        self.assertTrue(handle.stderr.startswith("[salloc timed out"), handle.stderr)


class RunInSessionTests(unittest.TestCase):
    """The srun argv is faked to plain bash so the *streaming* path runs for real."""

    def _run(self, script: str, **kwargs):
        with mock.patch(
            "reclaim_repro.slurm.build_srun", return_value=["bash", "-c", script]
        ) as build:
            result = run_in_session("/ws", script, jobid="9", **kwargs)
        self.assertEqual(build.call_args.kwargs["jobid"], "9")
        return result

    def test_runs_step_and_populates_result(self):
        result = self._run("echo hello-step; echo err-line >&2")
        self.assertTrue(result.ok, result.stderr)
        self.assertIn("hello-step", result.stdout)
        self.assertIn("err-line", result.stderr)
        self.assertGreaterEqual(result.elapsed_s, 0.0)

    def test_output_is_teed_to_the_log_as_it_streams(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "gpu_step_0000.log"
            result = self._run("echo one; echo two >&2", log_path=log)
            self.assertTrue(result.ok)
            blob = log.read_text()
            self.assertIn("one", blob)
            self.assertIn("two", blob)  # both streams share the log, arrival order

    def test_timeout_kills_but_returns_partial_output_and_keeps_the_log(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "gpu_step_0000.log"
            # Prints progress, then would print the "result" after a long sleep;
            # the kill must still return everything printed before it.
            result = self._run("echo before-kill; sleep 30; echo after", timeout=1.0, log_path=log)
            self.assertFalse(result.ok)
            self.assertEqual(result.returncode, 124)
            self.assertIn("before-kill", result.stdout)
            self.assertIn("exceeded timeout", result.stderr)
            self.assertIn("before-kill", log.read_text())  # survived the kill on disk

    def test_output_under_the_retention_cap_round_trips_byte_for_byte(self):
        # Anything short enough to fit the head buffer must come back exactly as
        # printed — the bounded capture only ever drops the middle of a huge stream.
        payload = "".join(f"line {i:04d} {'x' * 40}\n" for i in range(200))
        result = self._run(f"printf '%s' {shlex.quote(payload)}")
        self.assertTrue(result.ok, result.stderr)
        self.assertEqual(result.stdout, payload)

    def test_huge_output_is_bounded_but_keeps_head_and_tail(self):
        """A step printing far more than the cap costs fixed RAM, not its whole output.

        The complete stream is still teed to the evidence log (that is the record the
        auditor and the agent read); memory only has to carry what the callers slice
        back out — ``output.shape``'s 40k-char window and ``output.tail``'s 4k — so the
        head and the tail both have to survive.
        """
        import tempfile

        spam = 8 * STREAM_KEEP_BYTES  # ~2 MB, well past both retention regions
        script = (
            f"{shlex.quote(sys.executable)} -c "
            + shlex.quote(
                "import sys\n"
                "sys.stdout.write('HEAD-MARK\\n')\n"
                f"sys.stdout.write('x' * {spam})\n"
                "sys.stdout.write('\\nTAIL-MARK\\n')\n"
            )
        )
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "gpu_step_0000.log"
            result = self._run(script, log_path=log)
            self.assertTrue(result.ok, result.stderr)
            # The tee is complete: everything printed is on disk.
            self.assertGreaterEqual(log.stat().st_size, spam)
            # What we hold in RAM is not: two capped regions plus at most one chunk.
            self.assertLess(len(result.stdout), 2 * STREAM_KEEP_BYTES + 65536 + 4096)
            self.assertGreaterEqual(len(result.stdout), 2 * STREAM_KEEP_BYTES)
            # Both ends survive, so everything downstream reads the same chars it
            # would have read from the unbounded buffer.
            self.assertTrue(result.stdout.startswith("HEAD-MARK\n"))
            self.assertTrue(result.stdout.endswith("\nTAIL-MARK\n"))
            shaped, clipped = output.shape(result.stdout, 40_000)
            self.assertTrue(clipped)
            self.assertTrue(shaped.startswith("HEAD-MARK\n"))
            self.assertIn("TAIL-MARK", output.tail(result.stdout, 4_000))

    def test_srun_argv_is_unbuffered(self):
        argv = build_srun("/ws", "python t.py", jobid="9")
        self.assertIn("--unbuffered", argv)


class ReleaseAndLostTests(unittest.TestCase):
    def test_release_session_scancels(self):
        with mock.patch("reclaim_repro.slurm.subprocess.run") as run:
            release_session("100001")
        self.assertEqual(run.call_args.args[0], ["scancel", "100001"])

    def test_session_lost_detects_expired_allocation(self):
        lost = StepResult(
            ok=False, returncode=1, stdout="",
            stderr="srun: error: Unable to confirm allocation for job 555: Invalid job id specified",
            elapsed_s=0.1, command=["srun"],
        )
        ok = StepResult(ok=True, returncode=0, stdout="x", stderr="", elapsed_s=0.1, command=["srun"])
        self.assertTrue(session_lost(lost))
        self.assertFalse(session_lost(ok))

    def test_session_lost_detects_time_wall_expiry(self):
        # A held allocation that hit its --time wall drains COMPLETING -> EXPIRED; each
        # stage produces a distinct srun message and every one must drop the session so
        # the next run_gpu re-acquires instead of hammering the dead jobid.
        for stderr in (
            "srun: error: Unable to create step for job 100003: Job/step already completing or completed",
            "srun: error: Slurm job 100003 has expired",
            "srun: Check SLURM_JOB_ID environment variable. Expired or invalid job 100003",
        ):
            step = StepResult(ok=False, returncode=1, stdout="", stderr=stderr, elapsed_s=0.1, command=["srun"])
            self.assertTrue(session_lost(step), stderr)


if __name__ == "__main__":
    unittest.main()
