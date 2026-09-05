"""Tests for the audit-mode finalizer: the anti-cheat cap is enforced in code."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from reclaim_vllm.audit.audit import finalize_audit_row  # noqa: E402
from reclaim_vllm.audit.inputs import build_audit_prompt, claim_block  # noqa: E402
from reclaim_vllm.runtime.run_health import INCOMPLETE_EXIT_REASONS  # noqa: E402


def _score_row(score, flags=None, execution_verified=True) -> dict:
    return {
        "paper_id": "2505.10978",
        "central_claim": "86.7% on ALFWorld",
        "target_metric": "success_rate",
        "reference_value": 86.7,
        "op": "abs_rel_within",
        "tolerance": 0.05,
        "execution_verified": execution_verified,
        "execution_evidence": "ran run_alfworld.sh, eval log present",
        "measured_value": 86.7,
        "measured_citation": "out/res.json:1",
        "cheat_flags": flags or [],
        "value_comparison": "86.7 within 5% of 86.7",
        "methodology_notes": "",
        "score": score,
        "rationale": "x",
    }


def test_high_flag_caps_score_to_zero():
    flags = [{"kind": "hardcoded_constant", "evidence": "print(86.7) in eval.py:40", "severity": "high"}]
    row = finalize_audit_row(_score_row(10, flags), {"exit_reason": "natural"})
    assert row["score"] == 0
    assert row["reported_score"] == 10
    assert row["verdict"] == "disqualified"
    assert row["reproduced"] is False
    assert row["has_high_cheat_flag"] is True
    assert row["verification_status"] == "verified"


def test_score_10_is_reproduced():
    row = finalize_audit_row(_score_row(10), {"exit_reason": "natural"})
    assert row["verdict"] == "reproduced"
    assert row["reproduced"] is True
    assert "reported_score" not in row


def test_score_8_is_reproduced_boundary():
    row = finalize_audit_row(_score_row(8), {"exit_reason": "natural"})
    assert row["verdict"] == "reproduced"
    assert row["reproduced"] is True


def test_score_7_is_partial_boundary():
    row = finalize_audit_row(_score_row(7), {"exit_reason": "natural"})
    assert row["verdict"] == "partial"
    assert row["reproduced"] is False


def test_score_6_is_partial():
    flags = [{"kind": "cherry_picked_metric", "evidence": "minor", "severity": "low"}]
    row = finalize_audit_row(_score_row(6, flags), {"exit_reason": "natural"})
    assert row["verdict"] == "partial"
    assert row["score"] == 6  # low flag does not cap


def test_score_5_is_not_reproduced():
    row = finalize_audit_row(_score_row(5), {"exit_reason": "natural"})
    assert row["verdict"] == "not_reproduced"
    assert row["reproduced"] is False


def test_score_3_is_not_reproduced():
    # Score 3 (post-freeze: right experiment killed before the number; pre-freeze:
    # availability ceiling) carries no separate verdict.
    row = finalize_audit_row(_score_row(3), {"exit_reason": "natural"})
    assert row["verdict"] == "not_reproduced"
    assert row["reproduced"] is False


def test_high_score_without_execution_is_capped_unverifiable():
    # The finalizer must not certify a reproduction the auditor says never ran:
    # execution_verified=False + score 10 is incoherent and caps to unverifiable.
    row = finalize_audit_row(_score_row(10, execution_verified=False), {"exit_reason": "natural"})
    assert row["score"] == 1
    assert row["reported_score"] == 10
    assert row["verdict"] == "unverifiable"
    assert row["reproduced"] is False


def test_partial_score_without_execution_is_capped_unverifiable():
    row = finalize_audit_row(_score_row(6, execution_verified=False), {"exit_reason": "natural"})
    assert row["score"] == 1
    assert row["reported_score"] == 6
    assert row["reproduced"] is False


def test_score_3_without_execution_is_not_capped():
    # Score 3 (killed before the number) by definition never computed the metric,
    # so execution_verified=False is coherent; the cap only touches >=6.
    row = finalize_audit_row(_score_row(3, execution_verified=False), {"exit_reason": "natural"})
    assert row["score"] == 3
    assert row["verdict"] == "not_reproduced"
    assert "reported_score" not in row


def test_score_1_no_execution_is_unverifiable():
    row = finalize_audit_row(_score_row(1, execution_verified=False), {"exit_reason": "natural"})
    assert row["verdict"] == "unverifiable"
    assert row["reproduced"] is False


def test_score_0_is_disqualified():
    row = finalize_audit_row(_score_row(0, execution_verified=False), {"exit_reason": "natural"})
    assert row["verdict"] == "disqualified"
    assert row["reproduced"] is False


def test_out_of_range_score_is_degraded():
    row = finalize_audit_row(_score_row(11), {"exit_reason": "natural"})
    assert row["verification_status"] == "degraded"


def test_missing_paper_id_is_degraded():
    bad = _score_row(5)
    bad["paper_id"] = ""
    row = finalize_audit_row(bad, {"exit_reason": "natural"})
    assert row["verification_status"] == "degraded"


def test_claim_block_uses_central_claim():
    block = claim_block({"central_claim": "msf-CNN uses 50% less RAM", "claim_evidence": {"x": 1}})
    assert "msf-CNN uses 50% less RAM" in block
    assert "Reported numbers" in block


def test_claim_block_adopts_pinned_match_target():
    # The classifier now pins a coherent tuple; the auditor adopts it verbatim and
    # sets only op/tolerance — it does not re-derive the bar.
    block = claim_block(
        {
            "central_claim": "86.7% on ALFWorld",
            "match_target": {
                "config": "ReAct, GPT-4o",
                "metric": "success_rate",
                "value": "86.7%",
                "scope": "ALFWorld test",
                "match_bar_kind": "point_estimate",
            },
        }
    )
    assert "Pinned success bar" in block
    assert "adopt verbatim" in block.lower()
    # the pinned tuple's fields are present for the auditor to copy
    for token in ("ReAct, GPT-4o", "86.7%", "ALFWorld test", "point_estimate"):
        assert token in block, token
    # it must NOT tell the auditor to re-derive the bar
    assert "Derive the C1 match bar" not in block


def test_claim_block_falls_back_to_derive_when_no_tuple_pinned():
    # Legacy rows without a pinned match_target still ask the auditor to derive.
    block = claim_block({"central_claim": "A beats B", "claim_evidence": {"x": 1}})
    assert "Pinned success bar" not in block
    assert "Derive the C1 match bar" in block


def test_build_audit_prompt_fills_placeholders():
    template = "CLAIM:{CENTRAL_CLAIM}\nRUBRIC:{RUBRIC}\nBUNDLE:{RUN_BUNDLE}"
    out = build_audit_prompt(template, "RUBRIC_TEXT", {"central_claim": "C"}, "2505.10978", None)
    assert "{CENTRAL_CLAIM}" not in out and "{RUBRIC}" not in out and "{RUN_BUNDLE}" not in out
    assert "RUBRIC_TEXT" in out
    assert "unverifiable" in out  # no run dir bound → unverifiable


def test_build_audit_prompt_lists_run_directory(tmp_path):
    run_dir = tmp_path / "2505.10978"
    run_dir.mkdir()
    (run_dir / "train.log").write_text("success_rate: 86.7\n", encoding="utf-8")
    out = build_audit_prompt(
        "{CENTRAL_CLAIM}\n{RUBRIC}\n{RUN_BUNDLE}", "R", {"central_claim": "C"}, "2505.10978", tmp_path
    )
    assert "train.log" in out
    assert "AGENT RUN DIRECTORY" in out


def test_forced_exits_are_incomplete():
    for reason in INCOMPLETE_EXIT_REASONS:
        row = finalize_audit_row(_score_row(8), {"exit_reason": reason})
        assert row["verification_status"] == "incomplete", reason
