"""Deterministic post-processing for audit-mode rows.

The LLM auditor proposes a granular 0-10 reproduction ``score`` (audit_schema.py);
this module enforces the non-negotiable anti-cheat rules in code so they do not
depend on the auditor's goodwill: (1) any HIGH-severity cheat flag caps the score
at 0, and (2) a partial/reproduced score (>=6) asserts a run actually computed the
metric, so if the auditor itself reports ``execution_verified=False`` that grade is
incoherent and is capped to the unverifiable ceiling -- an un-executed run can never
count as reproduced. It then derives the coarse ``verdict`` and the ``reproduced``
boolean from the (possibly capped) score, plus the run-health ``verification_status``
(degraded when the auditor output is malformed).
"""

from __future__ import annotations

from typing import Any

from reclaim_vllm.schema.audit import SCORE_MAX, SCORE_MIN, SEVERITIES
from reclaim_vllm.runtime.run_health import INCOMPLETE_EXIT_REASONS, loop_exit_reason

DEGRADED = "degraded"
INCOMPLETE = "incomplete"
VERIFIED = "verified"

# A run counts toward the headline reproduction rate at or above this score.
REPRODUCED_MIN_SCORE = 8
# At or above this score the verdict asserts a run computed the metric (partial or
# better), so it is only coherent when execution_verified is True.
PARTIAL_MIN_SCORE = 6
# Ceiling an un-executed verdict is capped to: a metric that was never run cannot be
# confirmed, only left unverifiable.
UNVERIFIABLE_SCORE = 1


def finalize_audit_row(parsed: dict[str, Any], tool_loop: dict[str, Any]) -> dict[str, Any]:
    row = dict(parsed)
    exit_reason = loop_exit_reason(tool_loop)
    row["exit_reason"] = exit_reason

    flags = _normalize_flags(parsed.get("cheat_flags"))
    row["cheat_flags"] = flags
    high_flags = [f for f in flags if f.get("severity") == "high"]
    row["has_high_cheat_flag"] = bool(high_flags)

    score = _normalize_score(parsed.get("score"))
    execution = bool(parsed.get("execution_verified"))
    # Enforce the §3 rule in code, not in the prompt: a high-severity provenance
    # flag caps the score at 0 regardless of what the auditor assigned.
    if score is not None and high_flags and score > SCORE_MIN:
        row["reported_score"] = score
        score = SCORE_MIN
    # A partial/reproduced score claims a run computed the metric. If the auditor
    # simultaneously reports execution_verified=False the grade contradicts itself,
    # so cap it to the unverifiable ceiling: no execution => never reproduced.
    elif score is not None and not execution and score >= PARTIAL_MIN_SCORE:
        row["reported_score"] = score
        score = UNVERIFIABLE_SCORE
    row["score"] = score
    row["verdict"] = _verdict(score, execution, bool(high_flags))
    row["reproduced"] = score is not None and score >= REPRODUCED_MIN_SCORE

    row["verification_status"] = _verification_status(parsed, score, exit_reason)
    return row


def _normalize_score(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and SCORE_MIN <= value <= SCORE_MAX:
        return value
    return None


def _verdict(score: int | None, execution_verified: bool, cheated: bool) -> str | None:
    if score is None:
        return None
    # Score 0 is reserved for an integrity break: the HIGH-flag cap lands here, and
    # the auditor also assigns a raw 0 for a report that contradicts its own evidence.
    if cheated or score == SCORE_MIN:
        return "disqualified"
    if score >= REPRODUCED_MIN_SCORE:  # 8-10
        return "reproduced"
    if score >= 6:  # 6-7
        return "partial"
    if score == 1 and not execution_verified:
        return "unverifiable"
    # 2-5 (and a 1 that somehow ran). Post-freeze, 3 = right experiment killed by
    # resources before the number; pre-freeze rows used 3 for the availability
    # ceiling (`blocked`). Either way it is a not_reproduced.
    return "not_reproduced"


def _normalize_flags(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    flags = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        severity = item.get("severity")
        flags.append(
            {
                "kind": str(item.get("kind") or "unknown"),
                "evidence": str(item.get("evidence") or ""),
                "severity": severity if severity in SEVERITIES else "low",
            }
        )
    return flags


def _verification_status(parsed: dict[str, Any], score: int | None, exit_reason: str) -> str:
    if score is None or not _is_valid(parsed):
        return DEGRADED
    if exit_reason in INCOMPLETE_EXIT_REASONS:
        return INCOMPLETE
    return VERIFIED


def _is_valid(parsed: dict[str, Any]) -> bool:
    if not isinstance(parsed, dict):
        return False
    if not isinstance(parsed.get("paper_id"), str) or not parsed["paper_id"].strip():
        return False
    return isinstance(parsed.get("execution_verified"), bool)
