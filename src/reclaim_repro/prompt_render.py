"""Render the reproduce prompt from one lockfile row.

``render_reproduce_prompt`` fills every ``{PLACEHOLDER}`` in
``prompts/prompt_reproduce.txt`` and refuses to return a prompt with any
placeholder left unfilled. Pure: a row (+ budget) in, a string out — the row
accessors live in ``dataset`` and the in-container path constants in
``sandbox``.
"""

from __future__ import annotations

import re

from reclaim_repro import sandbox
from reclaim_repro.dataset import arxiv_id_of, band_of, format_hours

# Every uppercase {TOKEN} the template carries. Literal JSON shown to the agent is
# lowercase on purpose, so this regex only ever matches a real placeholder.
_PLACEHOLDER_RE = re.compile(r"\{[A-Z][A-Z0-9_]*\}")


def render_reproduce_prompt(template: str, row: dict, *, budget: float) -> str:
    rendered = template
    for token, value in _replacements(row, budget).items():
        rendered = rendered.replace(token, value)
    leftover = sorted(set(_PLACEHOLDER_RE.findall(rendered)))
    if leftover:
        raise ValueError(f"reproduce prompt has unfilled placeholders: {', '.join(leftover)}")
    return rendered


def _replacements(row: dict, budget: float) -> dict[str, str]:
    return {
        "{ARXIV_ID}": arxiv_id_of(row) or "(unknown)",
        "{PAPER_KIND}": _text_or(row.get("paper_kind"), "empirical"),
        "{TIER}": _text_or(row.get("tier"), "(untiered)"),
        "{BAND}": band_of(row),
        "{BUDGET_H100_HOURS}": format_hours(budget),
        "{CENTRAL_CLAIM}": _text_or(row.get("central_claim"), "(no central claim recorded)"),
        "{CLAIM_EVIDENCE}": _text_or(row.get("claim_evidence"), "(no reported numbers recorded)"),
        "{VERIFIED_LINKS}": _verified_links_block(row),
        "{SIGNALS}": _signals_block(row),
        "{MATCH_TARGET}": _match_target_block(row),
        # Short, stable in-container paths the agent actually sees (sandbox.py remaps the
        # long per-run host dirs onto these), so the prompt never hands it the long path.
        "{WORKSPACE_DIR}": sandbox.CONTAINER_WORKSPACE,
        "{REFERENCE_DIR}": sandbox.CONTAINER_REFERENCE,
        "{EVIDENCE_DIR}": sandbox.CONTAINER_EVIDENCE,
        "{RUN_DIR}": sandbox.CONTAINER_RUN,
    }


def _text_or(value: object, fallback: str) -> str:
    text = str(value).strip() if value is not None else ""
    return text or fallback


def _verified_links_block(row: dict) -> str:
    links = row.get("verified_links") or {}
    labels = (
        ("Code", "code"),
        ("Paper / project", "paper_or_project"),
        ("Dataset", "dataset"),
        ("Weights / checkpoints", "weights"),
    )
    lines: list[str] = []
    for label, key in labels:
        urls = [u for u in (links.get(key) or []) if isinstance(u, str) and u.strip()]
        if urls:
            lines.append(f"{label}:")
            lines.extend(f"  - {url}" for url in urls)
    if not lines:
        return (
            "(No released artifacts for this paper — locate the code yourself or "
            "re-implement the method from the paper.)"
        )
    return "\n".join(lines)


def _match_target_block(row: dict) -> str:
    """Render the lockfile row's anchor target MINUS its ``config``.

    Hands the agent the success bar to match — metric, reported value, scope, and bar
    shape — while still withholding ``match_target['config']`` (the exact runnable
    configuration), which the agent must derive from the paper. Degrades to a clear
    fallback when no target is recorded.
    """
    target = row.get("match_target") or {}
    fields = (
        ("Metric", "metric"),
        ("Target value", "value"),
        ("Scope", "scope"),
        ("Bar shape", "match_bar_kind"),
    )
    lines: list[str] = []
    for label, key in fields:
        text = _text_or(target.get(key), "")
        if text:
            lines.append(f"  - {label}: {text}")
    if not lines:
        return (
            "(No anchor target recorded — derive the metric, value, and scope from the "
            "paper yourself.)"
        )
    return "\n".join(lines)


def _signals_block(row: dict) -> str:
    """Render the lockfile row's pre-assessed artifact availability for the prompt.

    The lockfile carries per-paper ``signals`` (code / dataset / weights /
    dataset-is-standard + a verification level and evidence). Surfacing it spares the
    agent the rounds it otherwise burns rediscovering, e.g., that a repo is a release
    stub — while the prompt still tells it to verify against its exact MRE. Degrades to
    a clear fallback when a row predates signals.
    """
    signals = row.get("signals") or {}
    labels = (
        ("Code", "code_available"),
        ("Dataset", "dataset_available"),
        ("Weights / checkpoints", "weights_available"),
        ("Dataset is a standard benchmark", "dataset_is_standard"),
    )
    lines: list[str] = []
    for label, key in labels:
        sig = signals.get(key)
        if not isinstance(sig, dict) or "value" not in sig:
            continue
        value = "yes" if sig.get("value") else "no"
        verification = _text_or(sig.get("verification"), "unverified")
        line = f"  - {label}: {value} (classifier verification: {verification})"
        evidence = _text_or(sig.get("evidence"), "")
        if evidence:
            line += f" — {evidence}"
        lines.append(line)
    if not lines:
        return (
            "(No pre-assessed availability recorded — determine code/dataset/weights "
            "availability yourself.)"
        )
    return "\n".join(lines)
