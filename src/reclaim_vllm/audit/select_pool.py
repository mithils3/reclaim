"""Select the band-stratified human-audit candidate pool from an extracted run.

Implements the band-stratified selection rule described in the paper's
dataset-construction section, over the three evaluated tiers (Easy / Medium / Hard;
Artifact-Blocked is logged-only). Eligible rows are ``verification_status == "verified"`` with audited
H100 hours at or below the 96-hour cap, bucketed by compute band, filled from the
methodology's per-tier band weights (10/7/8 per 25) scaled to the pool size,
cheapest-first inside each band. Deficits in an expensive band refill from the next
cheapest band in the same tier.

Audited hours adjudicate the H100 arithmetic audit: when the recompute flagged a
mismatch and the stated number looks inflated (sane multiplier), the recomputed
``gpu_count x wallclock x multiplier`` value wins over the model's stated hours.

Usage::

    python -m reclaim_vllm.audit.select_pool \
        --run <classifier-run-dir> \
        --out <pool-prefix> --total 200

Writes ``<out>_extracted.jsonl`` (selected rows + selection fields),
``<out>_trace.jsonl`` (matching traces, streamed), and ``<out>_summary.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from collections.abc import Iterator

from reclaim_vllm.audit.h100 import as_number, band_labels, h100_band

EVAL_TIERS = ("Easy", "Medium", "Hard")
H100_CAP = 96.0
BAND_ORDER = band_labels(max_hours=H100_CAP)  # the ladder, truncated at the cap
BAND_WEIGHTS = {"0-8": 10, "8-32": 7, "32-96": 8}  # per 25 selected
# Stated hours adjudicated correct by hand despite an arithmetic mismatch flag
# (fields captured one of two stages; see the v5 H100 audit).
MANUAL_KEEP_STATED = {"2511.08214"}
# Recomputed hours are only trusted when the H100-equivalent multiplier is sane;
# above this the structured fields themselves are garbage (degraded-row pattern).
MULTIPLIER_SANE_MAX = 1.1


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def audited_h100_hours(row: dict[str, Any]) -> tuple[float | None, bool]:
    """Adjudicated hours and whether the recompute replaced the stated value."""
    estimate = row.get("h100_estimate") or {}
    stated = as_number(estimate.get("hours"))
    if (
        row.get("h100_arithmetic_mismatch")
        and row.get("custom_id") not in MANUAL_KEEP_STATED
        and (as_number(estimate.get("h100_equivalent_multiplier")) or 0) <= MULTIPLIER_SANE_MAX
    ):
        recomputed = as_number(row.get("h100_recomputed_hours"))
        if recomputed is not None:
            return recomputed, True
    return stated, False


def eligible(row: dict[str, Any]) -> bool:
    if row.get("verification_status") != "verified":
        return False
    if row.get("tier") not in EVAL_TIERS:
        return False
    hours, _ = audited_h100_hours(row)
    return hours is not None and 0 <= hours <= H100_CAP


def largest_remainder(weights: dict[str, float], total: int) -> dict[str, int]:
    """Apportion ``total`` across keys proportionally to ``weights``."""
    scale = total / sum(weights.values())
    exact = {k: w * scale for k, w in weights.items()}
    counts = {k: int(v) for k, v in exact.items()}
    leftover = total - sum(counts.values())
    for k in sorted(exact, key=lambda k: exact[k] - counts[k], reverse=True)[:leftover]:
        counts[k] += 1
    return counts


def select_tier(rows: list[dict[str, Any]], target: int) -> list[dict[str, Any]]:
    """Fill band quotas, refilling each deficit from the next cheapest band."""
    by_band: dict[str, list[dict[str, Any]]] = {b: [] for b in BAND_ORDER}
    for row in rows:
        by_band[row["selection_band"]].append(row)
    for bucket in by_band.values():
        bucket.sort(key=lambda r: (r["audited_h100_hours"], r["custom_id"]))

    quotas = largest_remainder(BAND_WEIGHTS, target)
    selected: list[dict[str, Any]] = []
    deficit = 0
    for band in reversed(BAND_ORDER):  # most expensive first so refills cascade down
        want = quotas[band] + deficit
        take = by_band[band][:want]
        selected.extend(take)
        deficit = want - len(take)
    if deficit:  # tier-wide shortage: take whatever is left, cheapest first
        chosen = {r["custom_id"] for r in selected}
        rest = sorted(
            (r for r in rows if r["custom_id"] not in chosen),
            key=lambda r: (r["audited_h100_hours"], r["custom_id"]),
        )
        selected.extend(rest[:deficit])
    return sorted(selected, key=lambda r: (r["audited_h100_hours"], r["custom_id"]))


def select_pool(rows: list[dict[str, Any]], total: int) -> dict[str, list[dict[str, Any]]]:
    pool: dict[str, list[dict[str, Any]]] = {t: [] for t in EVAL_TIERS}
    for row in rows:
        if not eligible(row):
            continue
        hours, adjudicated = audited_h100_hours(row)
        row = dict(row)
        row["audited_h100_hours"] = hours
        row["h100_hours_adjudicated"] = adjudicated
        row["selection_band"] = h100_band(hours)
        pool[row["tier"]].append(row)

    tier_targets = largest_remainder({t: 1 for t in EVAL_TIERS}, total)
    return {t: select_tier(pool[t], tier_targets[t]) for t in EVAL_TIERS}


def write_outputs(base: Path, run: Path, selection: dict[str, list[dict[str, Any]]]) -> None:
    selected = [row for tier in EVAL_TIERS for row in selection[tier]]
    ids = {row["custom_id"] for row in selected}

    extracted_out = Path(f"{base}_extracted.jsonl")
    with extracted_out.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(selected)} rows to {extracted_out}")

    trace_in = Path(f"{run}_trace.jsonl")
    trace_out = Path(f"{base}_trace.jsonl")
    kept = 0
    if trace_in.exists():
        with trace_out.open("w", encoding="utf-8") as handle:
            for row in iter_jsonl(trace_in):
                if str(row.get("custom_id")) in ids:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    kept += 1
        print(f"Wrote {kept} traces to {trace_out}")
    else:
        print(f"  ! no trace file at {trace_in}; skipped trace filtering")

    summary = {
        "source_run": str(run),
        "total_selected": len(selected),
        "h100_cap": H100_CAP,
        "band_weights_per_25": BAND_WEIGHTS,
        "tiers": {
            tier: {
                "selected": len(rows),
                "bands": {b: sum(1 for r in rows if r["selection_band"] == b) for b in BAND_ORDER},
                "adjudicated_hours": sum(1 for r in rows if r["h100_hours_adjudicated"]),
                "h100_flagged": sum(1 for r in rows if r.get("h100_needs_human_review")),
                "total_audited_h100_hours": round(sum(r["audited_h100_hours"] for r in rows), 1),
            }
            for tier, rows in selection.items()
        },
    }
    summary_out = Path(f"{base}_summary.json")
    summary_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {summary_out}")
    for tier in EVAL_TIERS:
        info = summary["tiers"][tier]
        bands = "  ".join(f"{b}:{n}" for b, n in info["bands"].items())
        print(f"  {tier:<7} {info['selected']:>3} selected   {bands}   "
              f"flagged:{info['h100_flagged']}  adjudicated:{info['adjudicated_hours']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="extracted/trace basename (no suffix)")
    parser.add_argument("--out", type=Path, required=True, help="output basename (no suffix)")
    parser.add_argument("--total", type=int, default=200, help="pool size across all tiers")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = list(iter_jsonl(Path(f"{args.run}_extracted.jsonl")))
    selection = select_pool(rows, args.total)
    write_outputs(args.out, args.run, selection)


if __name__ == "__main__":
    main()
