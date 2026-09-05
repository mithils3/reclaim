"""JSONL and extraction helpers for the code-availability re-check of
Reimplement-tier rows.

The one-shot re-check has already run. What remains here is the library the
publish and report tools import: read the recorded responses, extract the
classification rows, and gate on completeness.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from reclaim_vllm.schema.output import normalize_score_and_tier

POOL = Path("outputs/v5/audit_pool_extracted.jsonl")
RAW_NAME = "results_raw.jsonl"
EXTRACTED_NAME = "recheck_extracted.jsonl"


def iter_jsonl(path: Path, *, on_error: str = "raise") -> Iterator[dict[str, Any]]:
    """Stream the JSON objects of a JSONL file, skipping blank lines.

    This is the one JSONL reader for the recheck pipeline and the tools built on
    it. ``on_error="raise"`` (the default) lets a missing file and a torn line
    both blow up, which is what the data-merging paths want. ``on_error="report"``
    notes each problem on stderr and keeps going, which is what the app builders
    want when a 200 MB dump has one bad line in it.
    """
    if on_error == "report" and not path.exists():
        print(f"  ! missing {path}", file=sys.stderr)
        return
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as err:
                if on_error == "raise":
                    raise
                print(f"  ! {path.name}:{line_no} bad json: {err.msg}", file=sys.stderr)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    """Truncate ``path`` and write one JSON object per line. Returns the count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def hard_no_code_ids(pool_path: Path) -> list[str]:
    ids = []
    for row in iter_jsonl(pool_path):
        code = (row.get("signals") or {}).get("code_available") or {}
        if row.get("tier") == "Hard" and not code.get("value"):
            ids.append(str(row["custom_id"]))
    return ids


def raw_rows(raw_path: Path) -> dict[str, dict]:
    """Every recorded response body, keyed by custom_id. Empty until the run starts."""
    if not raw_path.exists():
        return {}
    return {str(row["custom_id"]): row["body"] for row in iter_jsonl(raw_path)}


def completed(rows: dict[str, dict]) -> dict[str, dict]:
    """The subset whose API call actually finished; the rest are retryable."""
    return {cid: body for cid, body in rows.items() if body.get("status") == "completed"}


def completed_raw_rows(raw_path: Path) -> dict[str, dict]:
    return completed(raw_rows(raw_path))


def output_text(body: dict) -> str:
    return "".join(
        content.get("text") or ""
        for item in body.get("output") or []
        if item.get("type") == "message"
        for content in item.get("content") or []
        if content.get("type") == "output_text"
    )


def parse_result(custom_id: str, body: dict) -> dict:
    row = {
        "custom_id": custom_id,
        "model": body.get("model"),
        "response_status": body.get("status"),
        "web_search_calls": sum(
            1 for i in body.get("output") or [] if i.get("type") == "web_search_call"
        ),
        "usage": body.get("usage"),
    }
    if body.get("status") != "completed":
        return {**row, "error": f"response status {body.get('status')}"}
    text = output_text(body)
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {**row, "error": "unparseable output", "raw_text": text[:2000]}
    return normalize_score_and_tier({**row, **parsed})


def collect(directory: Path) -> None:
    rows = [
        parse_result(cid, body)
        for cid, body in sorted(raw_rows(directory / RAW_NAME).items())
    ]
    out = directory / EXTRACTED_NAME
    write_jsonl(out, rows)
    parsed = [r for r in rows if "error" not in r]
    found = [
        r["custom_id"]
        for r in parsed
        if (r.get("signals") or {}).get("code_available", {}).get("value")
    ]
    tiers = dict(Counter(row.get("tier") or "?" for row in parsed))
    print(f"Wrote {len(rows)} rows to {out} ({len(rows) - len(parsed)} errors)", flush=True)
    print(f"code_available now true for {len(found)}: {found}", flush=True)
    print(f"Recomputed tiers: {tiers}", flush=True)


def add_recheck_args(parser: argparse.ArgumentParser) -> None:
    """The three flags every downstream recheck consumer shares."""
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")


def wait_for_recheck(directory: Path, poll_seconds: int) -> None:
    """Block until every hard/no-code paper has a completed response."""
    total = len(hard_no_code_ids(POOL))
    raw_path = directory / RAW_NAME
    while True:
        done = len(completed_raw_rows(raw_path))
        print(f"recheck progress: {done}/{total}", flush=True)
        if done >= total:
            return
        time.sleep(poll_seconds)


def collect_recheck(
    directory: Path,
    allow_partial: bool,
    *,
    incomplete: str = "recheck is incomplete: {done}/{total}",
    partial: str = "recheck is incomplete; reporting partial results: {done}/{total}",
    row_errors: str = "{count} recheck rows have extraction errors",
) -> Path:
    """Extract the raw responses and return the path to the extracted rows.

    Exits when the run is short of ``hard_no_code_ids`` without ``allow_partial``,
    or when any extracted row carries an error. The three messages are the
    caller's own wording so each operational script keeps the output its operator
    reads.
    """
    total = len(hard_no_code_ids(POOL))
    done = len(completed_raw_rows(directory / RAW_NAME))
    if done < total and not allow_partial:
        raise SystemExit(incomplete.format(done=done, total=total))
    if done < total:
        print(partial.format(done=done, total=total), flush=True)
    collect(directory)
    extracted = directory / EXTRACTED_NAME
    errors = [row for row in iter_jsonl(extracted) if "error" in row]
    if errors:
        raise SystemExit(row_errors.format(count=len(errors)))
    return extracted
