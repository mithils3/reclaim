"""Load MRE records from a JSONL source and index them by paper id.

Each line is one JSON record; rows are keyed by the first present of
``custom_id`` / ``paper_id`` / ``arxiv_id``. Consumers today are the audit
runner (``--claims``) and the repro lockfile loader, which both read whatever
fields their prompt needs off the returned row.

The source may be a local JSONL path or an HF reference of the form
``hf://datasets/<owner>/<name>/<file>`` so a cluster run can pull the record
pool straight from the Hub.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_mre_records(source) -> dict[str, dict]:
    path = _resolve_records_path(source)
    records: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        arxiv_id = str(row.get("custom_id") or row.get("paper_id") or row.get("arxiv_id") or "")
        if arxiv_id:
            records[arxiv_id] = row
    if not records:
        raise SystemExit(f"No MRE records found in {source}")
    return records


def _resolve_records_path(source) -> Path:
    parsed = _parse_hf_spec(str(source))
    if parsed is None:
        return Path(str(source))
    repo_id, filename = parsed
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset"))


def _parse_hf_spec(spec: str) -> tuple[str, str] | None:
    # A Path() round-trip (argparse type=Path on --claims) collapses the "hf://"
    # double slash to "hf:/", so accept both forms. Check "hf://" first: it also
    # starts with "hf:/", and matching the longer prefix keeps `rest` correct.
    for prefix in ("hf://", "hf:/"):
        if spec.startswith(prefix):
            rest = spec[len(prefix) :]
            break
    else:
        return None
    if rest.startswith("datasets/"):
        rest = rest[len("datasets/") :]
    segments = [segment for segment in rest.split("/") if segment]
    if len(segments) < 3:
        raise SystemExit(
            f"Bad hf:// mre-records spec: {spec} "
            "(expected hf://datasets/<owner>/<name>/<file>)"
        )
    return "/".join(segments[:2]), "/".join(segments[2:])
