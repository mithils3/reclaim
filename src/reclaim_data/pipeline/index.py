"""Index stage: pre-matched arxiv ids from the ai-conferences/NeurIPS2025 dataset.

Rows without an arxiv_id are dropped; nothing here fuzzy-matches titles.
"""

from __future__ import annotations

import csv
import re
import sys
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

from .common import INDEX_FILENAME, arxiv_eprint_url


NEURIPS_DATASET = "ai-conferences/NeurIPS2025"
NEURIPS_DATASET_COLUMNS = ["title", "paper_url", "type", "arxiv_id", "arxiv_id_source"]
INDEX_COLUMNS = ["arxiv_id", "title", "openreview_id", "arxiv_id_source", "type", "source_url"]
ARXIV_ID_RE = re.compile(r"(?i)(?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?")


@dataclass(frozen=True)
class IndexRecord:
    arxiv_id: str
    title: str
    openreview_id: str
    arxiv_id_source: str
    type: str
    source_url: str


def stage_index(data_dir: Path, force: bool) -> list[IndexRecord]:
    path = data_dir / INDEX_FILENAME
    if path.is_file() and not force:
        records = read_index_csv(path)
        print(
            f"Loaded {len(records)} papers from {path} (pass --force to refetch).",
            file=sys.stderr,
        )
        return records
    records = load_neurips_index()
    write_index_csv(path, records)
    print(f"Wrote {len(records)} papers to {path}.", file=sys.stderr)
    return records


def load_neurips_index() -> list[IndexRecord]:
    from datasets import load_dataset

    dataset = load_dataset(NEURIPS_DATASET, split="train")
    dataset = dataset.select_columns(NEURIPS_DATASET_COLUMNS)

    records: list[IndexRecord] = []
    seen: set[str] = set()
    total = missing = malformed = duplicates = 0
    for row in dataset:
        total += 1
        arxiv_id = str(row.get("arxiv_id") or "").strip()
        if not arxiv_id:
            missing += 1
            continue
        if not ARXIV_ID_RE.fullmatch(arxiv_id):
            malformed += 1
            continue
        base_id = re.sub(r"v\d+$", "", arxiv_id)
        if base_id in seen:
            duplicates += 1
            continue
        seen.add(base_id)
        records.append(
            IndexRecord(
                arxiv_id=arxiv_id,
                title=" ".join(str(row.get("title") or "").split()),
                openreview_id=parse_openreview_id(str(row.get("paper_url") or "")),
                arxiv_id_source=str(row.get("arxiv_id_source") or ""),
                type=str(row.get("type") or ""),
                source_url=arxiv_eprint_url(arxiv_id),
            )
        )
    print(
        f"{NEURIPS_DATASET}: {total} rows, kept {len(records)} "
        f"({missing} without arxiv_id, {malformed} malformed, {duplicates} duplicates).",
        file=sys.stderr,
    )
    return records


def parse_openreview_id(paper_url: str) -> str:
    if not paper_url:
        return ""
    parsed = urllib.parse.urlparse(paper_url)
    note_id = urllib.parse.parse_qs(parsed.query).get("id", [""])[0]
    if note_id:
        return note_id
    parts = [part for part in parsed.path.split("/") if part]
    if parts and parts[-1] not in {"forum", "pdf"}:
        return parts[-1]
    return ""


def write_index_csv(path: Path, records: list[IndexRecord]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(record.__dict__)


def read_index_csv(path: Path) -> list[IndexRecord]:
    if not path.is_file():
        raise SystemExit(
            f"Missing index CSV: {path}. Run with the index stage first, e.g. "
            f"--stages index,sources,supplements,bundle"
        )
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in INDEX_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(
                f"{path} is missing column(s): {', '.join(missing)}. "
                f"Re-run the index stage with --force."
            )
        return [
            IndexRecord(
                arxiv_id=row["arxiv_id"],
                title=row.get("title") or "",
                openreview_id=row.get("openreview_id") or "",
                arxiv_id_source=row.get("arxiv_id_source") or "",
                type=row.get("type") or "",
                source_url=row.get("source_url") or arxiv_eprint_url(row["arxiv_id"]),
            )
            for row in reader
            if row.get("arxiv_id")
        ]
