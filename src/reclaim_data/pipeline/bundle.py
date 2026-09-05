"""Bundle stage: assemble the one-row-per-paper Parquet dataset from extracted files."""

from __future__ import annotations

import hashlib
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import (
    BUNDLE_DIRNAME,
    MANIFEST_FILENAME,
    SOURCES_DIRNAME,
    SUPPLEMENTS_DIRNAME,
    load_manifest_map,
    safe_dirname,
)
from .index import NEURIPS_DATASET, IndexRecord
from .output import (
    SCHEMA,
    ShardedBundleWriter,
    write_bundle_readme,
    write_bundle_stats,
)


TEXT_EXTENSIONS = {
    ".aux",
    ".bbl",
    ".bib",
    ".bst",
    ".cfg",
    ".cls",
    ".csv",
    ".def",
    ".dtx",
    ".enc",
    ".html",
    ".idx",
    ".ins",
    ".json",
    ".ldf",
    ".log",
    ".md",
    ".out",
    ".sty",
    ".tex",
    ".txt",
    ".yaml",
    ".yml",
}


@dataclass
class Stats:
    papers: int = 0
    papers_with_tex: int = 0
    papers_with_supplement: int = 0
    papers_skipped_no_source: int = 0
    tex_files: int = 0
    supplement_files: int = 0
    bytes: int = 0
    shards: int = 0


def stage_bundle(
    records: list[IndexRecord],
    data_dir: Path,
    *,
    shard_size_mb: int,
    batch_size_mb: int,
    batch_rows: int,
    compression: str,
    force: bool,
) -> Stats:
    output_dir = data_dir / BUNDLE_DIRNAME
    if output_dir.exists():
        if not force:
            raise SystemExit(f"{output_dir} already exists. Pass --force to replace it.")
        shutil.rmtree(output_dir)
    output_data_dir = output_dir / "data"
    output_data_dir.mkdir(parents=True, exist_ok=True)

    sources_dir = data_dir / SOURCES_DIRNAME
    supplements_dir = data_dir / SUPPLEMENTS_DIRNAME
    sources_manifest = load_manifest_map(sources_dir / MANIFEST_FILENAME)
    supplements_manifest = load_manifest_map(supplements_dir / MANIFEST_FILENAME)

    writer = ShardedBundleWriter(
        output_data_dir=output_data_dir,
        target_shard_bytes=shard_size_mb * 1024 * 1024,
        batch_bytes=batch_size_mb * 1024 * 1024,
        batch_rows=batch_rows,
        compression=compression,
        schema=SCHEMA,
    )
    stats = Stats()
    start = time.monotonic()
    ordered = sorted(records, key=lambda record: record.arxiv_id)
    for record in ordered:
        source_dir = sources_dir / safe_dirname(record.arxiv_id)
        if not source_dir.is_dir() or not any(source_dir.iterdir()):
            stats.papers_skipped_no_source += 1
            continue
        row, logical_bytes = make_bundle_row(
            record,
            sources_manifest.get(record.arxiv_id),
            supplements_manifest.get(record.arxiv_id),
            source_dir,
            supplements_dir,
        )
        writer.write(row, logical_bytes)
        update_stats(stats, row, logical_bytes)
        if stats.papers % 100 == 0:
            elapsed = max(time.monotonic() - start, 0.001)
            print(
                f"[{stats.papers}/{len(ordered)}] bundled papers "
                f"({stats.papers / elapsed:.1f} papers/s)",
                file=sys.stderr,
            )
    stats.shards = writer.close()
    write_bundle_readme(output_dir, stats, NEURIPS_DATASET)
    write_bundle_stats(
        output_dir,
        stats,
        {
            "neurips_dataset": NEURIPS_DATASET,
            "data_dir": str(data_dir),
            "shard_size_mb": shard_size_mb,
            "batch_size_mb": batch_size_mb,
            "batch_rows": batch_rows,
            "compression": compression,
        },
    )
    print(
        f"Bundled {stats.papers} papers ({stats.papers_skipped_no_source} skipped without "
        f"sources) into {stats.shards} shard(s) in {output_dir}.",
        file=sys.stderr,
    )
    return stats


def make_bundle_row(
    record: IndexRecord,
    source_row: dict[str, str] | None,
    supplement_row: dict[str, str] | None,
    source_dir: Path,
    supplements_dir: Path,
) -> tuple[dict[str, Any], int]:
    tex_files = collect_tex_files(source_dir)
    supplement_files = artifact_files(supplements_dir / safe_dirname(record.arxiv_id))
    row = {
        "arxiv_id": record.arxiv_id,
        "title": record.title,
        "openreview_id": record.openreview_id,
        "arxiv_id_source": record.arxiv_id_source,
        "paper_source_url": record.source_url,
        "paper_status": (source_row or {}).get("status") or "downloaded",
        "paper_tex_files": tex_files,
        "paper_tex_text": join_tex(tex_files),
        "supplement_source_url": (supplement_row or {}).get("source_url") or "",
        "supplement_status": "downloaded" if supplement_files else "missing",
        "supplement_files": supplement_files,
    }
    logical_bytes = sum(item["file_size"] for item in tex_files + supplement_files)
    return row, logical_bytes


def collect_tex_files(source_dir: Path) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file() or path.suffix.casefold() != ".tex":
            continue
        content = path.read_bytes()
        files.append(
            {
                "relative_path": path.relative_to(source_dir).as_posix(),
                "filename": path.name,
                "file_size": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
                "text": decode_text(content) or "",
            }
        )
    return files


def artifact_files(root: Path) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")) if root.is_dir() else []:
        if not path.is_file():
            continue
        content = path.read_bytes()
        extension = path.suffix.casefold()
        text = decode_text(content) if extension in TEXT_EXTENSIONS else None
        files.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "filename": path.name,
                "extension": extension,
                "file_size": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
                "is_text": text is not None,
                "text": text,
                "content": content,
            }
        )
    return files


def join_tex(files: list[dict[str, Any]]) -> str:
    chunks = []
    for item in files:
        chunks.append(f"% FILE: {item['relative_path']}\n{item['text']}")
    return "\n\n".join(chunks)


def decode_text(content: bytes) -> str | None:
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        pass
    try:
        return content.decode("latin-1")
    except UnicodeDecodeError:
        return None


def update_stats(stats: Stats, row: dict[str, Any], logical_bytes: int) -> None:
    stats.papers += 1
    stats.papers_with_tex += int(bool(row["paper_tex_files"]))
    stats.papers_with_supplement += int(bool(row["supplement_files"]))
    stats.tex_files += len(row["paper_tex_files"])
    stats.supplement_files += len(row["supplement_files"])
    stats.bytes += logical_bytes
