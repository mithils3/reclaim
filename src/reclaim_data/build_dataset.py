#!/usr/bin/env python3
"""Build the NeurIPS 2025 paper dataset: arXiv LaTeX sources + OpenReview supplements.

Single pipeline with five stages:

  index       load pre-matched arxiv ids from the ai-conferences/NeurIPS2025
              dataset and snapshot them to data/neurips2025_index.csv
  sources     download and extract arXiv e-print packages
  supplements download OpenReview supplementary material, matched by note id
  bundle      assemble the one-row-per-paper Parquet dataset
  upload      push the bundle folder to the Hugging Face Hub

Papers without a pre-matched arxiv_id are dropped; nothing in this pipeline
fuzzy-matches titles.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .pipeline.bundle import stage_bundle
from .pipeline.common import BUNDLE_DIRNAME, INDEX_FILENAME, NOTES_CACHE_FILENAME, STAGES
from .pipeline.index import read_index_csv, stage_index
from .pipeline.output import DEFAULT_REPO_ID, stage_upload
from .pipeline.sources import stage_sources
from .pipeline.supplements import DEFAULT_VENUE_ID, OPENREVIEW_API, stage_supplements


def main() -> int:
    args = parse_args()
    stages = resolve_stages(args.stages, args.upload)
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if "index" in stages:
        records = stage_index(data_dir, force=args.force)
    else:
        records = read_index_csv(data_dir / INDEX_FILENAME)
    if args.limit:
        records = records[: args.limit]
    print(f"{len(records)} papers in scope for stages: {', '.join(stages)}.", file=sys.stderr)

    if args.dry_run:
        for record in records[: min(10, len(records))]:
            print(f"{record.arxiv_id}\t{record.openreview_id}\t{record.source_url}")
        return 0

    failures = 0
    if "sources" in stages:
        results = stage_sources(
            records,
            data_dir,
            workers=args.workers,
            delay=args.delay,
            retries=args.retries,
            timeout=args.timeout,
            overwrite=args.force,
            keep_archive=args.keep_archive,
        )
        failures += sum(result.status == "failed" for result in results)
    if "supplements" in stages:
        results = stage_supplements(
            records,
            data_dir,
            notes_json=args.notes_json,
            api_base=args.api_base,
            venue_id=args.venue_id,
            workers=args.workers,
            delay=args.supplement_delay,
            retries=args.retries,
            overwrite=args.force,
        )
        failures += sum(result.status == "failed" for result in results)
    if "bundle" in stages:
        stage_bundle(
            records,
            data_dir,
            shard_size_mb=args.shard_size_mb,
            batch_size_mb=max(1, args.batch_size_mb),
            batch_rows=max(1, args.batch_rows),
            compression=args.compression,
            force=args.force,
        )
    if "upload" in stages:
        stage_upload(
            data_dir / BUNDLE_DIRNAME,
            repo_id=args.repo_id,
            private=args.private,
            commit_message=args.commit_message,
        )
    if failures:
        print(f"{failures} download(s) failed.", file=sys.stderr)
        if not args.allow_failures:
            return 1
    return 0


def resolve_stages(raw: str, upload: bool) -> list[str]:
    requested = {part.strip() for part in raw.split(",") if part.strip()}
    unknown = requested - set(STAGES)
    if unknown:
        raise SystemExit(
            f"Unknown stage(s): {', '.join(sorted(unknown))}. "
            f"Valid stages: {', '.join(STAGES)}"
        )
    if upload:
        requested.add("upload")
    return [stage for stage in STAGES if stage in requested]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--stages",
        default="index,sources,supplements,bundle",
        help=f"Comma-separated subset of: {', '.join(STAGES)}. Always run in that order.",
    )
    parser.add_argument("--data-dir", default="data", help="Root for all pipeline artifacts.")
    parser.add_argument("--limit", type=int, help="Process at most this many papers.")
    parser.add_argument(
        "--force",
        "--overwrite",
        dest="force",
        action="store_true",
        help="Refetch the index, re-download existing dirs, and replace the bundle output.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--delay",
        type=float,
        default=0.25,
        help="Minimum seconds between arXiv requests across all workers.",
    )
    parser.add_argument(
        "--supplement-delay",
        type=float,
        default=0.75,
        help="Minimum seconds between OpenReview attachment requests across all workers.",
    )
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--keep-archive", action="store_true")
    parser.add_argument(
        "--notes-json",
        type=Path,
        help=f"OpenReview notes cache path. Defaults to <data-dir>/{NOTES_CACHE_FILENAME}.",
    )
    parser.add_argument("--venue-id", default=DEFAULT_VENUE_ID)
    parser.add_argument("--api-base", default=OPENREVIEW_API)
    parser.add_argument("--shard-size-mb", type=int, default=512)
    parser.add_argument("--batch-size-mb", type=int, default=64)
    parser.add_argument("--batch-rows", type=int, default=64)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--upload", action="store_true", help="Upload the bundle after building.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--private", action="store_true")
    parser.add_argument(
        "--commit-message",
        default="Upload NeurIPS 2025 paper bundle dataset",
    )
    parser.add_argument("--allow-failures", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
