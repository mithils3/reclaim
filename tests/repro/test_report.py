from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from reclaim_repro.context import ExecutionContext
from reclaim_repro.evidence import init_evidence
from reclaim_repro.report import (
    REPORT_FILENAME,
    REPORT_RESPONSE_FORMAT,
    REPORT_SCHEMA_NAME,
    coerce_report,
    validate_report,
    write_episode_report,
)
from reclaim_vllm.tools.run_dir_tools import run_dir_manifest


def good_report() -> dict:
    return {
        "paper_id": "2510.21323",
        "claim": "Method X reaches 76.5% top-1 on ImageNet-1k.",
        "what_ran": "Built a uv venv, cloned the repo, ran eval on the full val split.",
        "scoring_command": "python eval.py --ckpt out/best.pt --split val",
        "measurements": [
            {
                "metric": "top-1 accuracy (%)",
                "observed_value": "76.3%",
                "reference_value": "76.5%",
                "scope": "ImageNet-1k val",
                "evidence": ["evidence/REPORT.md", "evidence/commands.log:42"],
            }
        ],
        "agent_assessment": "partial",
        "changes_made": "Pinned torch 2.3 to match the CUDA in the image.",
        "blockers": "",
        "evidence_files": ["REPORT.md", "commands.log"],
    }


class ReportSchemaTests(unittest.TestCase):
    def test_response_format_is_a_named_json_schema(self) -> None:
        self.assertEqual(REPORT_RESPONSE_FORMAT["type"], "json_schema")
        self.assertEqual(REPORT_RESPONSE_FORMAT["json_schema"]["name"], REPORT_SCHEMA_NAME)
        schema = REPORT_RESPONSE_FORMAT["json_schema"]["schema"]
        self.assertFalse(schema["additionalProperties"])
        self.assertIn("measurements", schema["properties"])


class ValidateReportTests(unittest.TestCase):
    def test_good_report_is_valid(self) -> None:
        self.assertEqual(validate_report(good_report()), [])

    def test_non_object_rejected(self) -> None:
        self.assertTrue(validate_report(["not", "a", "dict"]))

    def test_missing_required_field_flagged(self) -> None:
        bad = good_report()
        del bad["measurements"]
        errors = validate_report(bad)
        self.assertTrue(any("missing required field: measurements" in e for e in errors))

    def test_bad_assessment_enum_flagged(self) -> None:
        bad = good_report()
        bad["agent_assessment"] = "totally_reproduced"
        self.assertTrue(any("agent_assessment" in e for e in validate_report(bad)))

    def test_measurements_must_be_list(self) -> None:
        bad = good_report()
        bad["measurements"] = {"metric": "x"}
        self.assertTrue(any("measurements must be an array" in e for e in validate_report(bad)))

    def test_measurement_evidence_must_be_string_list(self) -> None:
        bad = good_report()
        bad["measurements"][0]["evidence"] = "evidence/REPORT.md"
        self.assertTrue(any("evidence must be an array" in e for e in validate_report(bad)))

    def test_null_reference_value_is_allowed(self) -> None:
        ok = good_report()
        ok["measurements"][0]["reference_value"] = None
        self.assertEqual(validate_report(ok), [])


class CoerceReportTests(unittest.TestCase):
    def test_valid_content_becomes_complete_report(self) -> None:
        report = coerce_report(
            json.dumps(good_report()), paper_id="2510.21323", exit_reason="natural"
        )
        self.assertEqual(report["report_status"], "complete")
        self.assertEqual(report["exit_reason"], "natural")
        self.assertEqual(report["agent_assessment"], "partial")
        # The persisted complete report still satisfies the schema's required fields.
        self.assertEqual(validate_report(report), [])

    def test_paper_id_is_owned_by_the_harness(self) -> None:
        spoofed = good_report()
        spoofed["paper_id"] = "9999.99999"
        report = coerce_report(json.dumps(spoofed), paper_id="2510.21323", exit_reason="round_limit")
        self.assertEqual(report["paper_id"], "2510.21323")

    def test_unparseable_content_degrades_not_crashes(self) -> None:
        report = coerce_report("the run did not finish", paper_id="2510.21323", exit_reason="budget_exhausted")
        self.assertEqual(report["report_status"], "degraded")
        self.assertEqual(report["paper_id"], "2510.21323")
        self.assertTrue(report["validation_errors"])
        self.assertIn("did not finish", report["raw"])

    def test_schema_invalid_json_degrades(self) -> None:
        # Parses as JSON but misses required fields -> degraded, with errors recorded.
        report = coerce_report('{"paper_id": "2510.21323"}', paper_id="2510.21323", exit_reason="natural")
        self.assertEqual(report["report_status"], "degraded")
        self.assertTrue(any("missing required field" in e for e in report["validation_errors"]))


class WriteEpisodeReportTests(unittest.TestCase):
    def test_report_lands_in_bundle_and_shows_in_auditor_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paper_dir = Path(tmp) / "2510.21323"
            run_dir = paper_dir / "8h" / "run1"
            evidence = run_dir / "evidence"
            init_evidence(evidence)
            ctx = ExecutionContext(arxiv_id="2510.21323", run_dir=run_dir, evidence=evidence)

            path = write_episode_report(ctx, json.dumps(good_report()), "natural")

            self.assertIsNotNone(path)
            self.assertEqual(path, run_dir / REPORT_FILENAME)
            written = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(written["report_status"], "complete")
            self.assertEqual(validate_report(written), [])

            # The auditor reads run_dir_manifest(<runs-dir>/<paper_id>); the
            # report must appear there so the existing auditor grades it unchanged.
            manifest = run_dir_manifest(paper_dir)
            self.assertIn("report.json", manifest)

            # The write is recorded as a trajectory step in the evidence.
            traj = (evidence / "trajectory.jsonl").read_text(encoding="utf-8")
            self.assertIn('"type": "report"', traj)

    def test_no_run_dir_is_a_noop(self) -> None:
        ctx = ExecutionContext(arxiv_id="2510.21323")
        self.assertIsNone(write_episode_report(ctx, json.dumps(good_report()), "natural"))


if __name__ == "__main__":
    unittest.main()
