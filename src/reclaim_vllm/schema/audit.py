"""Audit-mode output schema: the structured verdict an LLM auditor returns.

Replaces the deterministic verification target. The auditor reads a paper's
central claim plus one agent run bundle (code, logs, artifacts) and grades it
against rubric_audit.md. Every graded quantity must be traceable to cited
evidence in the bundle; the verdict is reconstructable from the flags and
citations alone (see audit.py for the deterministic post-processing).
"""

from __future__ import annotations


# Shape of the central claim's success bar. The classifier PINS the coherent tuple
# (config, metric, value, scope, match_bar_kind) into the lockfile's `match_target`;
# the auditor ADOPTS it verbatim and sets only op / tolerance to match the pinned
# match_bar_kind, per rubric C1 (it no longer re-derives the bar).
MATCH_BAR_KINDS = (
    "point_estimate",  # land near reference_value; op abs_rel_within, tolerance set
    "threshold",       # reference_value is a floor/ceiling; op >= or <=, tolerance null
    "direction",       # beat a baseline; op names the inequality, reference/tolerance null
    "magnitude",       # the size of a delta is the target; tolerance applies to the delta
    "none",            # no checkable scalar/relation (theoretical/position); all null
)

# Granular reproduction score the auditor assigns (anchors in rubric_audit.md).
SCORE_MIN, SCORE_MAX = 0, 10
FLAG_KINDS = (
    "hardcoded_constant",
    "echoed_prose_number",
    "self_scored_or_fabricated",
    "wrong_split_scale_dataset",
    "cherry_picked_metric",
    "stale_artifact",
    "retrieved_benchmark_answer",  # run consulted the benchmark split/lockfile answer key
    "other_provenance_break",      # open-ended: novel cheats are still cheats (rubric C4)
)
SEVERITIES = ("low", "med", "high")

_STR = {"type": "string"}
_NUM_OR_NULL = {"type": ["number", "null"]}
_BOOL = {"type": "boolean"}


def _obj(properties: dict, required: list[str]) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": properties,
    }


def _flag_item() -> dict:
    return _obj(
        {
            "kind": {"type": "string", "enum": list(FLAG_KINDS)},
            "evidence": _STR,
            "severity": {"type": "string", "enum": list(SEVERITIES)},
        },
        ["kind", "evidence", "severity"],
    )


AUDIT_JSON_SCHEMA = _obj(
    {
        "paper_id": _STR,
        # C1: the checkable target. The auditor ADOPTS the pinned match_target tuple
        # (match_bar_kind / target_metric / reference_value / target_scope from the
        # lockfile) verbatim and sets only op / tolerance to match the pinned kind.
        "central_claim": _STR,
        "match_bar_kind": {"type": "string", "enum": list(MATCH_BAR_KINDS)},
        "target_metric": _STR,
        "target_scope": _STR,
        "reference_value": _NUM_OR_NULL,
        "op": _STR,
        "tolerance": _NUM_OR_NULL,
        # C2: proof the metric was actually computed by a run, with citation.
        "execution_verified": _BOOL,
        "execution_evidence": _STR,
        # C3: the agent's reproduced value and where it was found.
        "measured_value": _NUM_OR_NULL,
        "measured_citation": _STR,
        # C4: anti-cheat flags; any high-severity flag caps the verdict.
        "cheat_flags": {"type": "array", "items": _flag_item()},
        # C5/C6: the comparison and experiment fidelity.
        "value_comparison": _STR,
        "methodology_notes": _STR,
        # granular 0-10 reproduction score; verdict is derived from it downstream.
        "score": {"type": "integer", "minimum": SCORE_MIN, "maximum": SCORE_MAX},
        # auditor's own 0-1 confidence; < 0.6 routes the verdict to human spot-audit.
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "rationale": _STR,
    },
    [
        "paper_id",
        "central_claim",
        "match_bar_kind",
        "target_metric",
        "target_scope",
        "reference_value",
        "op",
        "tolerance",
        "execution_verified",
        "execution_evidence",
        "measured_value",
        "measured_citation",
        "cheat_flags",
        "value_comparison",
        "methodology_notes",
        "score",
        "confidence",
        "rationale",
    ],
)


AUDIT_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "audit_verdict",
        "schema": AUDIT_JSON_SCHEMA,
    },
}
