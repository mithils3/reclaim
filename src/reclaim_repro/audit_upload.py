"""Push auditor verdicts into the run-log database (opt-in, standalone).

The auditor (``reclaim_vllm --mode audit``) writes graded verdict rows to a JSONL
(``--extracted-output``), keyed by ``paper_id``, and deliberately never touches the
database: the classifier/auditor core stays storage-free. This step closes that
gap: it reads that JSONL and PATCHes the ``audit_*`` columns onto the matching
``repro_runs`` row, so the run viewer can show the score/verdict and drive its
"reproduced" status from the grader instead of a hand-applied tag.

Mapping a per-paper verdict to a ``run_id``: the repro sink drops ``stats.json``
(carrying ``run_id``) into each bundle root, so ``<runs-dir>/<paper_id>/stats.json``
names the run that produced the graded bundle. When that file is absent (the run
predates the sink, or stats upload was off) we fall back to the most recently
updated ``repro_runs`` row for the arxiv id.

Config mirrors ``telemetry_sink``: ``RECLAIM_TELEMETRY_URL`` (or ``--telemetry-url``) +
``RECLAIM_TELEMETRY_KEY`` (the service key bypasses RLS to write). Best-effort and
idempotent -- re-running overwrites the same row with the same verdict.

    python -m reclaim_repro.audit_upload \
      --verdicts audit_2505.18513_extracted.jsonl \
      --runs-dir $RECLAIM_SCRATCH/agent_runs
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
from pathlib import Path
from typing import Any

from reclaim_repro import postgrest
from reclaim_repro.event_sink import now_iso, service_key_from_env

HTTP_TIMEOUT = 15.0


def _request(method: str, url: str, key: str, body: Any = None) -> tuple[int, str]:
    prefer = "return=minimal" if method == "PATCH" else None
    return postgrest.request(url, service_key=key, method=method, body=body,
                             prefer=prefer, timeout=HTTP_TIMEOUT)


def run_id_from_stats(runs_dir: Path, paper_id: str) -> str | None:
    """The sink drops stats.json (carrying run_id) into each run's bundle root, at
    <runs-dir>/<arxiv_id>/<budget>h/<run_id>/stats.json. Grade the newest (the run
    the auditor just walked); '**' also matches a flat <arxiv_id>/stats.json."""
    paper_dir = Path(runs_dir) / paper_id
    try:
        found = sorted(paper_dir.glob("**/stats.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        return None
    for stats in found:
        try:
            rid = json.loads(stats.read_text()).get("run_id")
        except (OSError, ValueError):
            continue
        if rid:
            return rid
    return None


def run_id_from_db(base: str, key: str, arxiv_id: str) -> str | None:
    """Fallback: the most recently updated repro_runs row for this paper."""
    q = urllib.parse.urlencode(
        {"arxiv_id": f"eq.{arxiv_id}", "select": "run_id", "order": "updated_at.desc", "limit": "1"}
    )
    code, text = _request("GET", f"{base}/rest/v1/repro_runs?{q}", key)
    if not code or code >= 300:
        return None
    try:
        rows = json.loads(text)
    except ValueError:
        return None
    return rows[0]["run_id"] if rows else None


def audit_fields(row: dict[str, Any], model: str | None) -> dict[str, Any]:
    """The audit_* column payload derived from one finalized verdict row."""
    fields = {
        "audit_score": row.get("score"),
        "audit_reported_score": row.get("reported_score"),
        "audit_verdict": row.get("verdict"),
        "audit_reproduced": row.get("reproduced"),
        "audit_has_high_cheat_flag": row.get("has_high_cheat_flag"),
        "audit_flags": row.get("cheat_flags") or [],
        "audit_rationale": row.get("rationale"),
        "audited_at": now_iso(),
    }
    if model:
        fields["audit_model"] = model
    return fields


def paper_of(row: dict[str, Any]) -> str | None:
    """The extracted audit rows are keyed by custom_id (io.py always sets it);
    paper_id only appears when the model emits it in schema-valid JSON. Mirror
    load_mre_records' key resolution so a graded row is never dropped."""
    pid = row.get("custom_id") or row.get("paper_id") or row.get("arxiv_id")
    return str(pid) if pid else None


def load_verdicts(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if paper_of(obj):
            out.append(obj)
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="reclaim_repro.audit_upload", description=__doc__)
    p.add_argument("--verdicts", type=Path, required=True,
                   help="Auditor --extracted-output JSONL (one finalized row per paper_id).")
    p.add_argument("--runs-dir", type=Path, required=True,
                   help="Root of the agent run bundles; <runs-dir>/<arxiv_id>/stats.json names the run_id.")
    p.add_argument("--telemetry-url", default=os.environ.get("RECLAIM_TELEMETRY_URL"),
                   help="Telemetry service base URL (default: $RECLAIM_TELEMETRY_URL).")
    p.add_argument("--run-id",
                   help="Patch this exact run_id for every verdict row (single-paper use). "
                        "Skips the stats.json / latest-row lookup — pass it when the caller "
                        "already knows which run was graded.")
    p.add_argument("--audit-model", help="Grader model id to record on the row (audit_model).")
    p.add_argument("--dry-run", action="store_true", help="Resolve run_ids and print, but don't PATCH.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    base = (args.telemetry_url or "").rstrip("/")
    key = service_key_from_env()
    if not base or not key:
        print("audit_upload: set RECLAIM_TELEMETRY_URL (or --telemetry-url) and RECLAIM_TELEMETRY_KEY",
              file=sys.stderr)
        return 2

    verdicts = load_verdicts(args.verdicts)
    if not verdicts:
        print(f"audit_upload: no verdict rows in {args.verdicts}", file=sys.stderr)
        return 1

    patched = skipped = failed = 0
    for row in verdicts:
        pid = paper_of(row)
        # A degraded audit row (unparseable model output) carries no score/verdict —
        # nothing to grade with, so flag it rather than silently patching nulls.
        if row.get("score") is None and not row.get("verdict"):
            print(f"  {pid}: audit produced no verdict (degraded output) -- skipped", file=sys.stderr)
            skipped += 1
            continue
        run_id = args.run_id or run_id_from_stats(args.runs_dir, pid)
        if not run_id and not args.dry_run:
            run_id = run_id_from_db(base, key, pid)
        if not run_id:
            print(f"  {pid}: no run_id (no stats.json, no repro_runs row) -- skipped", file=sys.stderr)
            skipped += 1
            continue
        fields = audit_fields(row, args.audit_model)
        tag = f"score={fields['audit_score']} verdict={fields['audit_verdict']}"
        if args.dry_run:
            print(f"  {pid} -> {run_id}: {tag} (dry-run)")
            continue
        q = urllib.parse.urlencode({"run_id": f"eq.{run_id}"})
        code, text = _request("PATCH", f"{base}/rest/v1/repro_runs?{q}", key, fields)
        if code and code < 300:
            print(f"  {pid} -> {run_id}: {tag}")
            patched += 1
        else:
            print(f"  {pid} -> {run_id}: PATCH failed HTTP {code} {text[:160]}", file=sys.stderr)
            failed += 1

    print(f"audit_upload: {patched} patched, {skipped} skipped, {failed} failed "
          f"({len(verdicts)} verdicts)", file=sys.stderr)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
