from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from reclaim_repro.cluster import DEFAULT_CLUSTER, GPU_PARTITION, resolve_cluster
from reclaim_repro.context import ExecutionContext
from reclaim_repro.tools.partitions import list_partitions


def _ctx() -> ExecutionContext:
    return ExecutionContext(arxiv_id="x", cluster=resolve_cluster())


def _completed(stdout: str = "", *, rc: int = 0, stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=rc, stdout=stdout, stderr=stderr)


_SINFO = (
    "batch*|up|2-00:00:00|3/5/0/8|gpu:h200:4\n"
    "batch*|up|2-00:00:00|3/5/0/8|gpu:h200:4\n"   # duplicate state row -> collapses to one
    "batch-interactive|up|2:00:00|1/3/0/4|gpu:h200:4\n"
)


class ListPartitionsTests(unittest.TestCase):
    def test_parses_sinfo_and_dedupes_per_partition(self):
        with mock.patch(
            "reclaim_repro.tools.partitions.subprocess.run", return_value=_completed(_SINFO)
        ):
            res = list_partitions({}, _ctx())
        self.assertTrue(res["ok"])
        parts = {p["partition"]: p for p in res["partitions"]}
        self.assertEqual(set(parts), {"batch", "batch-interactive"})  # deduped
        self.assertTrue(parts["batch"]["is_slurm_default"])         # `*` marker stripped + flagged
        self.assertFalse(parts["batch-interactive"]["is_slurm_default"])
        self.assertEqual(parts["batch"]["nodes_allocated_idle_other_total"], "3/5/0/8")
        self.assertEqual(parts["batch-interactive"]["timelimit"], "2:00:00")

    def test_reports_known_defaults_and_active_cluster(self):
        with mock.patch(
            "reclaim_repro.tools.partitions.subprocess.run", return_value=_completed(_SINFO)
        ):
            res = list_partitions({}, _ctx())
        self.assertEqual(res["active_cluster"], DEFAULT_CLUSTER)
        self.assertEqual(res["active_default_partition"], GPU_PARTITION)
        self.assertEqual(
            res["known_cluster_defaults"][DEFAULT_CLUSTER]["default_partition"], GPU_PARTITION
        )
        self.assertEqual(set(res["known_cluster_defaults"]), {DEFAULT_CLUSTER})

    def test_sinfo_absent_degrades_to_defaults_with_a_note(self):
        with mock.patch(
            "reclaim_repro.tools.partitions.subprocess.run", side_effect=FileNotFoundError()
        ):
            res = list_partitions({}, _ctx())
        self.assertTrue(res["ok"])                 # still useful: defaults are returned
        self.assertEqual(res["partitions"], [])
        self.assertIn("sinfo", res["note"])
        self.assertEqual(
            res["known_cluster_defaults"][DEFAULT_CLUSTER]["default_partition"], GPU_PARTITION
        )

    def test_sinfo_nonzero_exit_is_a_note_not_a_crash(self):
        with mock.patch(
            "reclaim_repro.tools.partitions.subprocess.run",
            return_value=_completed("", rc=1, stderr="slurm down"),
        ):
            res = list_partitions({}, _ctx())
        self.assertTrue(res["ok"])
        self.assertEqual(res["partitions"], [])
        self.assertIn("slurm down", res["note"])


if __name__ == "__main__":
    unittest.main()
