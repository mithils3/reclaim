"""The auditor sink's wire behaviour: which tables it writes and with what policy.

``AuditSink`` shares its queue/worker/HTTP half with the reproduce sink via
``PostgrestEventSink``. These tests pin the parts that are audit-specific and that
a shared base could silently change: the two table paths, the ignore-duplicates
conflict policy on the events insert, the key-union padding every PostgREST bulk
insert needs, and the swallow-all discipline (a failing transport must count, not
raise, because the on-disk trace is the source of truth).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from reclaim_vllm.runtime.audit_sink import AuditSink, SinkConfig  # noqa: E402


def _cfg(**kw) -> SinkConfig:
    base = dict(url="https://db.example", service_key="k", graded_run_id="run-1",
                model="m", host="h")
    base.update(kw)
    return SinkConfig(**base)


def _drain(sink: AuditSink) -> None:
    """Let the worker thread finish; close() joins it."""
    sink.close(timeout=5.0)


class AuditSinkWireTests(unittest.TestCase):
    def _run_events(self, events, *, cfg=None, request=None):
        """Feed events through a sink with a stubbed transport; return the calls."""
        calls = []

        def fake_request(url, **kw):
            calls.append({"url": url, **kw})
            return (request or (lambda: (201, "")))()

        with mock.patch("reclaim_repro.postgrest.request", side_effect=fake_request):
            sink = AuditSink(cfg or _cfg())
            for kind, meta, payload in events:
                sink.on_event(kind, meta, payload)
            _drain(sink)
        return calls, sink

    def test_run_upsert_and_events_go_to_the_audit_tables(self) -> None:
        calls, sink = self._run_events([
            ("round_open", {"custom_id": "2506.05271", "round_index": 0},
             {"message": {"content": "thinking"}}),
        ])
        paths = [c["url"].replace("https://db.example", "") for c in calls]
        self.assertIn("/rest/v1/audit_runs?on_conflict=audit_run_id", paths)
        self.assertIn("/rest/v1/audit_events?on_conflict=audit_run_id,seq", paths)
        self.assertEqual(sink.failed, 0)
        self.assertEqual(sink.dropped, 0)

    def test_events_insert_ignores_duplicates(self) -> None:
        """A colliding (audit_run_id, seq) must not take the whole batch down."""
        calls, _ = self._run_events([
            ("round_open", {"custom_id": "x", "round_index": 0}, {"message": {"content": "a"}}),
        ])
        insert = next(c for c in calls if "audit_events" in c["url"])
        self.assertEqual(insert["prefer"], "resolution=ignore-duplicates,return=minimal")

    def test_bulk_insert_pads_rows_to_the_key_union(self) -> None:
        """PostgREST rejects a bulk insert whose rows have differing keys."""
        calls, _ = self._run_events([
            ("round_open", {"custom_id": "x", "round_index": 0}, {"message": {"content": "a"}}),
            ("call_start", {"custom_id": "x", "round_index": 0},
             {"call": {"function": {"name": "run_bash", "arguments": '{"command": "ls"}'}}}),
            ("call_result", {"custom_id": "x", "round_index": 0},
             {"result": {"ok": True, "returncode": 0, "stdout": "out"}}),
        ])
        inserts = [c for c in calls if "audit_events" in c["url"]]
        for c in inserts:
            keys = {frozenset(row) for row in c["body"]}
            self.assertEqual(len(keys), 1, "rows in one insert must share a key set")

    def test_sequence_numbers_are_dense_and_ordered(self) -> None:
        calls, _ = self._run_events([
            ("round_open", {"custom_id": "x", "round_index": 0}, {"message": {"content": "a"}}),
            ("call_start", {"custom_id": "x", "round_index": 0},
             {"call": {"function": {"name": "t", "arguments": "{}"}}}),
            ("final", {"custom_id": "x", "round_index": 1},
             {"message": {"content": "done"}, "exit_reason": "natural", "verdict": {"score": 4}}),
        ])
        seqs = [row["seq"] for c in calls if "audit_events" in c["url"] for row in c["body"]]
        self.assertEqual(seqs, sorted(seqs))
        self.assertEqual(seqs, list(range(len(seqs))))

    def test_final_patches_the_verdict_onto_the_run_row(self) -> None:
        calls, _ = self._run_events([
            ("final", {"custom_id": "x", "round_index": 2},
             {"message": {"content": "done"}, "exit_reason": "natural",
              "verdict": {"score": 3, "verdict": "partial", "reproduced": False,
                          "has_high_cheat_flag": True}}),
        ])
        patch = next(c for c in calls if c["method"] == "PATCH")
        self.assertIn("audit_runs?audit_run_id=eq.", patch["url"])
        self.assertEqual(patch["body"]["score"], 3)
        self.assertEqual(patch["body"]["verdict"], "partial")
        self.assertIs(patch["body"]["reproduced"], False)
        self.assertIs(patch["body"]["has_high_cheat_flag"], True)
        self.assertEqual(patch["body"]["status"], "finished")

    def test_attempt_discriminates_the_audit_run_id(self) -> None:
        """Two graders of one run must not land on the same row."""
        first, _ = self._run_events(
            [("round_open", {"custom_id": "x", "round_index": 0}, {"message": {}})])
        second, _ = self._run_events(
            [("round_open", {"custom_id": "x", "round_index": 0}, {"message": {}})],
            cfg=_cfg(attempt="b"))
        id_of = lambda calls: next(  # noqa: E731
            c["body"][0]["audit_run_id"] for c in calls if "audit_runs" in c["url"])
        self.assertNotEqual(id_of(first), id_of(second))

    def test_transport_failure_is_counted_not_raised(self) -> None:
        def boom():
            raise OSError("dns")

        calls, sink = self._run_events(
            [("round_open", {"custom_id": "x", "round_index": 0}, {"message": {}})],
            request=boom)
        self.assertGreater(sink.failed, 0)

    def test_http_error_status_is_counted(self) -> None:
        _, sink = self._run_events(
            [("round_open", {"custom_id": "x", "round_index": 0}, {"message": {}})],
            request=lambda: (409, "duplicate key"))
        self.assertGreater(sink.failed, 0)

    def test_events_without_a_custom_id_are_ignored(self) -> None:
        calls, sink = self._run_events([("round_open", {"round_index": 0}, {"message": {}})])
        self.assertEqual(calls, [])
        self.assertEqual(sink.failed, 0)


if __name__ == "__main__":
    unittest.main()
