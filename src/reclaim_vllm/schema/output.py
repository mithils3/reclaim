from __future__ import annotations

from typing import Any

from reclaim_vllm.schema.audit import MATCH_BAR_KINDS


SIGNAL_NAMES = (
    "code_available",
    "dataset_available",
    "weights_available",
    "dataset_is_standard",
)
VERIFICATION_STATES = (
    "tool_verified",
    "tool_searched_not_found",
    "tool_failed",
    "paper_text_only",
    "not_applicable",
)
PAPER_KINDS = ("empirical", "theoretical", "position", "survey")
NON_EMPIRICAL_TIER = "Out-of-Scope-Non-Empirical"
H100_BASIS_KINDS = (
    "paper_reported",
    "derived_from_config",
    "comparable_experiment",
    "compute_unspecified",
)
# NOTE: the classifier PINS one coherent success-bar tuple per paper in
# `match_target` — (config, metric, value, scope, match_bar_kind) — where running
# `config` and measuring `metric` over `scope` can actually yield `value`. The
# auditor ADOPTS this tuple verbatim and sets only `op` / `tolerance` to
# match the pinned `match_bar_kind` (see MATCH_BAR_KINDS in schema/audit.py and
# rubric C1). The classifier still does not pin op/tolerance.


def match_target_schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["config", "metric", "value", "scope", "match_bar_kind"],
        "properties": {
            # The exact runnable configuration that PRODUCES `value`.
            "config": {"type": "string"},
            # Anchor metric name, units explicit.
            "metric": {"type": "string"},
            # Paper's reported value for `metric` under `config` + `scope`. A string
            # so it tolerates "2.37x", "76.5%", "8.56 kB", or a range.
            "value": {"type": "string"},
            # Dataset / split / benchmark-SET the value is measured over.
            "scope": {"type": "string"},
            # Shape of the bar; the auditor sets op/tolerance to match it.
            "match_bar_kind": {"type": "string", "enum": list(MATCH_BAR_KINDS)},
        },
    }


def signal_schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["value", "verification", "evidence"],
        "properties": {
            "value": {"type": "boolean"},
            "verification": {"type": "string", "enum": list(VERIFICATION_STATES)},
            "evidence": {"type": "string"},
        },
    }


def h100_estimate_schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "hours",
            "basis_kind",
            "gpu_count",
            "gpu_type",
            "wallclock_hours",
            "h100_equivalent_multiplier",
            "basis",
        ],
        "properties": {
            "hours": {"type": "number"},
            "basis_kind": {"type": "string", "enum": list(H100_BASIS_KINDS)},
            "gpu_count": {"type": ["number", "null"]},
            "gpu_type": {"type": ["string", "null"]},
            "wallclock_hours": {"type": ["number", "null"]},
            "h100_equivalent_multiplier": {"type": ["number", "null"]},
            "basis": {"type": "string"},
        },
    }


FINAL_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "repro_artifact_classification",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "central_claim",
                "claim_evidence",
                "paper_kind",
                "mre_config",
                "match_target",
                "verified_links",
                "signals",
                "agent_task",
                "h100_estimate",
            ],
            "properties": {
                "central_claim": {"type": "string"},
                "claim_evidence": {"type": "string"},
                "paper_kind": {"type": "string", "enum": list(PAPER_KINDS)},
                "mre_config": {"type": "string"},
                "match_target": match_target_schema(),
                "verified_links": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["paper_or_project", "code", "dataset", "weights"],
                    "properties": {
                        "paper_or_project": {"type": "array", "items": {"type": "string"}},
                        "code": {"type": "array", "items": {"type": "string"}},
                        "dataset": {"type": "array", "items": {"type": "string"}},
                        "weights": {"type": "array", "items": {"type": "string"}},
                    },
                },
                "signals": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "code_available",
                        "dataset_available",
                        "weights_available",
                        "dataset_is_standard",
                    ],
                    "properties": {
                        "code_available": signal_schema(),
                        "dataset_available": signal_schema(),
                        "weights_available": signal_schema(),
                        "dataset_is_standard": signal_schema(),
                    },
                },
                "agent_task": {"type": "string"},
                "h100_estimate": h100_estimate_schema(),
            },
        },
    },
}

FINAL_JSON_SCHEMA = FINAL_RESPONSE_FORMAT["json_schema"]["schema"]


def deterministic_score_and_tier(row: dict[str, Any]) -> tuple[int, str] | None:
    signals = row.get("signals")
    if not isinstance(signals, dict):
        return None

    values = {name: signal_value(signals.get(name)) for name in SIGNAL_NAMES}
    if any(value is None for value in values.values()):
        return None

    code_available = values["code_available"]
    dataset_available = values["dataset_available"]
    weights_available = values["weights_available"]
    dataset_is_standard = values["dataset_is_standard"]

    score = 0
    if not code_available:
        score += 2
    if not dataset_is_standard and not dataset_available:
        score += 3
    if not weights_available:
        score += 1
    return score, tier_for_score(score, dataset_available, dataset_is_standard)


def normalize_score_and_tier(row: dict[str, Any]) -> dict[str, Any]:
    computed = deterministic_score_and_tier(row)
    if computed is None:
        return row

    score, tier = computed
    normalized = dict(row)
    if (
        "score" in normalized
        and normalized.get("score") != score
        and "reported_score" not in normalized
    ):
        normalized["reported_score"] = normalized.get("score")
    if (
        "tier" in normalized
        and normalized.get("tier") != tier
        and "reported_tier" not in normalized
    ):
        normalized["reported_tier"] = normalized.get("tier")
    normalized["score"] = score
    normalized["tier"] = tier
    return normalized


def signal_value(signal: Any) -> bool | None:
    if not isinstance(signal, dict):
        return None
    value = signal.get("value")
    return value if isinstance(value, bool) else None


def tier_for_score(
    score: int,
    dataset_available: bool,
    dataset_is_standard: bool,
) -> str:
    if score == 0:
        return "Easy"
    if score == 1:
        return "Medium"
    if score == 2:
        return "Hard"
    if score == 3 and (dataset_available or dataset_is_standard):
        return "Hard"
    return "Artifact-Blocked"
