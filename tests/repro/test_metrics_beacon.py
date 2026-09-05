"""Host telemetry beacon: pure parsers (nvidia-smi CSV, /proc fixtures), the
--dry-run --once row contract, and the per-run beacon hook in gpu_session.

Everything runs without a network, nvidia-smi, or SLURM: the parsers take text
(not paths), the dry-run test patches the /proc + nvidia-smi readers, and the
gpu_session tests mock salloc/Popen — the suite must pass on a GPU-less machine.
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from reclaim_repro import gpu_session, metrics_beacon, run_beacon
from reclaim_repro.context import ExecutionContext
from reclaim_repro.slurm import SessionHandle
from tests.repro.test_gpu_session import _ctx

_MIN_ENV = {"RECLAIM_TELEMETRY_URL": "https://telemetry.example", "RECLAIM_TELEMETRY_KEY": "k"}

# index, util%, mem.used MiB, mem.total MiB, power W, temp C (csv,noheader,nounits)
_NVSMI_CSV = "0, 93, 73830, 100147, 610.42, 52\n1, 0, 4, 100147, 88.10, 31\n"

# Aggregate line deltas: idle 500 (idle+iowait), total 700 -> busy 28.6 %.
_STAT_A = "cpu  100 0 100 700 100 0 0 0 0 0\ncpu0 100 0 100 700 100 0 0 0 0 0\n"
_STAT_B = "cpu  200 0 200 1200 100 0 0 0 0 0\ncpu0 200 0 200 1200 100 0 0 0 0 0\n"

# 100 GiB total, 30 GiB available -> 70.0 used / 100.0 total.
_MEMINFO = "MemTotal:       104857600 kB\nMemFree:        1048576 kB\nMemAvailable:   31457280 kB\n"
_LOADAVG = "3.14 2.71 1.61 2/2048 12345\n"

_METRICS_KEYS = {"host", "role", "batch_id", "run_id",
                 "cpu_pct", "mem_used_gb", "mem_total_gb", "load1", "gpus"}


class NvidiaCsvTests(unittest.TestCase):
    def test_csv_parses_to_the_contract_shape(self):
        gpus = metrics_beacon.parse_nvidia_csv(_NVSMI_CSV)
        # MiB -> GiB at 1 decimal; power/temp/util collapse to ints.
        self.assertEqual(gpus[0], {"i": 0, "util": 93, "mem": 72.1, "mem_total": 97.8,
                                   "power": 610, "temp": 52})
        self.assertEqual(gpus[1], {"i": 1, "util": 0, "mem": 0.0, "mem_total": 97.8,
                                   "power": 88, "temp": 31})

    def test_na_cells_become_null_fields(self):
        gpus = metrics_beacon.parse_nvidia_csv("0, [N/A], 73830, 100147, [N/A], 52\n")
        self.assertIsNone(gpus[0]["util"])
        self.assertIsNone(gpus[0]["power"])
        self.assertEqual(gpus[0]["temp"], 52)

    def test_garbage_or_empty_output_is_none(self):
        self.assertIsNone(metrics_beacon.parse_nvidia_csv("not,a,gpu,row"))
        self.assertIsNone(metrics_beacon.parse_nvidia_csv(""))


class ProcParserTests(unittest.TestCase):
    def test_stat_deltas_give_busy_pct(self):
        a = metrics_beacon.parse_stat(_STAT_A)
        b = metrics_beacon.parse_stat(_STAT_B)
        self.assertEqual(metrics_beacon.cpu_pct_between(a, b), 28.6)

    def test_stat_without_an_aggregate_line_is_none(self):
        self.assertIsNone(metrics_beacon.parse_stat("cpu0 1 2 3 4 5 6 7 8\n"))
        self.assertIsNone(metrics_beacon.cpu_pct_between(None, (0.0, 0.0)))

    def test_meminfo_used_is_total_minus_available_in_gib(self):
        self.assertEqual(metrics_beacon.parse_meminfo(_MEMINFO), (70.0, 100.0))
        self.assertEqual(metrics_beacon.parse_meminfo("MemTotal: 1 kB\n"), (None, None))

    def test_loadavg_first_field(self):
        self.assertEqual(metrics_beacon.parse_loadavg(_LOADAVG), 3.14)
        self.assertIsNone(metrics_beacon.parse_loadavg(""))


class DryRunTests(unittest.TestCase):
    """--dry-run --once must emit the exact contract rows with no telemetry env."""

    def _run(self, argv: list[str]) -> dict:
        stat_reads = [_STAT_A, _STAT_B]

        def fake_read(path: str) -> str:
            if path == "/proc/stat":
                return stat_reads.pop(0)
            return {"/proc/meminfo": _MEMINFO, "/proc/loadavg": _LOADAVG}.get(path, "")

        buf = io.StringIO()
        with mock.patch.dict("os.environ", {}, clear=True), \
             mock.patch("reclaim_repro.metrics_beacon._read", side_effect=fake_read), \
             mock.patch("reclaim_repro.metrics_beacon.sample_gpus",
                        return_value=metrics_beacon.parse_nvidia_csv(_NVSMI_CSV)), \
             redirect_stdout(buf):
            rc = metrics_beacon.main(argv)
        self.assertEqual(rc, 0)
        return json.loads(buf.getvalue())

    def test_rows_carry_the_contract_keys_and_values(self):
        doc = self._run(["--role", "master", "--dry-run", "--once"])
        metrics = doc["host_metrics"]
        self.assertEqual(set(metrics), _METRICS_KEYS)
        self.assertEqual(set(doc["host_status"]), _METRICS_KEYS | {"log_tail"})
        self.assertEqual(metrics["role"], "master")
        self.assertIsNone(metrics["run_id"])
        self.assertEqual(metrics["cpu_pct"], 28.6)
        self.assertEqual((metrics["mem_used_gb"], metrics["mem_total_gb"]), (70.0, 100.0))
        self.assertEqual(metrics["load1"], 3.14)
        self.assertEqual(metrics["gpus"][0]["mem"], 72.1)
        self.assertIsNone(doc["host_status"]["log_tail"])  # no --log-file given

    def test_run_role_stamps_the_run_id(self):
        doc = self._run(["--role", "run", "--run-id", "rid-42", "--dry-run", "--once"])
        self.assertEqual(doc["host_metrics"]["run_id"], "rid-42")

    def test_log_file_tail_lands_in_host_status(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "slurm.out"
            log.write_text("\n".join(f"line {i}" for i in range(200)) + "\n")
            doc = self._run(["--role", "master", "--dry-run", "--once",
                             "--log-file", str(log)])
        tail = doc["host_status"]["log_tail"]
        self.assertEqual(len(tail.splitlines()), 120)  # last 120 of 200 lines
        self.assertTrue(tail.endswith("line 199"))
        self.assertNotIn("line 79", tail.splitlines())

    def test_run_role_without_run_id_is_a_usage_error(self):
        with self.assertRaises(SystemExit), redirect_stdout(io.StringIO()):
            metrics_beacon.main(["--role", "run", "--dry-run", "--once"])

    def test_no_env_and_not_dry_run_exits_zero(self):
        buf = io.StringIO()
        with mock.patch.dict("os.environ", {}, clear=True), redirect_stdout(buf):
            rc = metrics_beacon.main(["--role", "master", "--once"])
        self.assertEqual(rc, 0)  # opt-in telemetry: unset env is never an error
        self.assertIn("telemetry off", buf.getvalue())


class RunBeaconHookTests(unittest.TestCase):
    """ensure_session spawns the per-run beacon; release/drop_lost terminate it."""

    def setUp(self):
        run_beacon._BEACONS.clear()

    def _acquire(self, ctx: ExecutionContext, env: dict[str, str]) -> mock.Mock:
        handle = SessionHandle(ok=True, jobid="777", stderr="")
        popen = mock.patch("reclaim_repro.run_beacon.subprocess.Popen", return_value=mock.Mock())
        with mock.patch.dict("os.environ", env, clear=True), \
             mock.patch("reclaim_repro.slurm.acquire_session", return_value=handle), \
             popen as spawned:
            session, err = gpu_session.ensure_session(ctx, gpus=1, minutes=30)
        self.assertIsNone(err)
        self.assertEqual(session.jobid, "777")
        return spawned

    def test_acquire_with_telemetry_env_spawns_the_beacon(self):
        with tempfile.TemporaryDirectory() as d:
            spawned = self._acquire(_ctx(Path(d) / "run123"), _MIN_ENV)
        spawned.assert_called_once()
        argv = spawned.call_args[0][0]
        self.assertEqual(argv[:2], ["srun", "--jobid=777"])
        self.assertIn("--overlap", argv)
        self.assertIn("reclaim_repro.metrics_beacon", argv)
        self.assertEqual(argv[argv.index("--role") + 1], "run")
        self.assertEqual(argv[argv.index("--run-id") + 1], "run123")
        self.assertTrue(spawned.call_args.kwargs["start_new_session"])

    def test_release_terminates_the_beacon_client(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = _ctx(Path(d) / "run123")
            spawned = self._acquire(ctx, _MIN_ENV)
            with mock.patch("reclaim_repro.slurm.release_session"):
                gpu_session.release(ctx, "agent")
        spawned.return_value.terminate.assert_called_once()
        self.assertEqual(run_beacon._BEACONS, {})

    def test_drop_lost_terminates_the_beacon_client(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = _ctx(Path(d) / "run123")
            spawned = self._acquire(ctx, _MIN_ENV)
            gpu_session.drop_lost(ctx)
        spawned.return_value.terminate.assert_called_once()
        self.assertEqual(run_beacon._BEACONS, {})

    def test_without_telemetry_env_nothing_is_spawned(self):
        with tempfile.TemporaryDirectory() as d:
            spawned = self._acquire(_ctx(Path(d) / "run123"), {})
        spawned.assert_not_called()
        self.assertEqual(run_beacon._BEACONS, {})


if __name__ == "__main__":
    unittest.main()
