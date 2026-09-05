"""Parquet schema, sharded writer, dataset card, and Hugging Face upload."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


# Upload target for the built bundle. No default: the operator names their own
# Hugging Face dataset repo with --repo-id or $RECLAIM_BUNDLE_DATASET.
DEFAULT_REPO_ID = os.environ.get("RECLAIM_BUNDLE_DATASET") or None

TEX_STRUCT = pa.struct(
    [
        ("relative_path", pa.string()),
        ("filename", pa.string()),
        ("file_size", pa.int64()),
        ("sha256", pa.string()),
        ("text", pa.string()),
    ]
)

FILE_STRUCT = pa.struct(
    [
        ("relative_path", pa.string()),
        ("filename", pa.string()),
        ("extension", pa.string()),
        ("file_size", pa.int64()),
        ("sha256", pa.string()),
        ("is_text", pa.bool_()),
        ("text", pa.string()),
        ("content", pa.binary()),
    ]
)

SCHEMA = pa.schema(
    [
        ("arxiv_id", pa.string()),
        ("title", pa.string()),
        ("openreview_id", pa.string()),
        ("arxiv_id_source", pa.string()),
        ("paper_source_url", pa.string()),
        ("paper_status", pa.string()),
        ("paper_tex_files", pa.list_(TEX_STRUCT)),
        ("paper_tex_text", pa.string()),
        ("supplement_source_url", pa.string()),
        ("supplement_status", pa.string()),
        ("supplement_files", pa.list_(FILE_STRUCT)),
    ]
)


class ShardedBundleWriter:
    def __init__(
        self,
        output_data_dir: Path,
        target_shard_bytes: int,
        batch_bytes: int,
        batch_rows: int,
        compression: str,
        schema: pa.Schema,
    ) -> None:
        self.output_data_dir = output_data_dir
        self.target_shard_bytes = target_shard_bytes
        self.batch_bytes = batch_bytes
        self.batch_rows = batch_rows
        self.compression = compression
        self.schema = schema
        self.shard_index = -1
        self.shard_logical_bytes = 0
        self.writer: pq.ParquetWriter | None = None
        self.rows: list[dict[str, Any]] = []
        self.rows_logical_bytes = 0

    def write(self, row: dict[str, Any], logical_bytes: int) -> None:
        if (
            self.rows
            and self.shard_logical_bytes
            and self.shard_logical_bytes + self.rows_logical_bytes >= self.target_shard_bytes
        ):
            self.flush()
        self.rows.append(row)
        self.rows_logical_bytes += logical_bytes
        if len(self.rows) >= self.batch_rows or self.rows_logical_bytes >= self.batch_bytes:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        if (
            self.writer is None
            or self.shard_logical_bytes >= self.target_shard_bytes
            or (
                self.shard_logical_bytes
                and self.shard_logical_bytes + self.rows_logical_bytes > self.target_shard_bytes
            )
        ):
            self._open_next_shard()
        assert self.writer is not None
        self.writer.write_table(pa.Table.from_pylist(self.rows, schema=self.schema))
        self.shard_logical_bytes += self.rows_logical_bytes
        self.rows = []
        self.rows_logical_bytes = 0

    def close(self) -> int:
        self.flush()
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        return self.shard_index + 1

    def _open_next_shard(self) -> None:
        if self.writer is not None:
            self.writer.close()
        self.shard_index += 1
        self.shard_logical_bytes = 0
        path = self.output_data_dir / f"train-{self.shard_index:05d}.parquet"
        self.writer = pq.ParquetWriter(path, self.schema, compression=self.compression)


def write_bundle_readme(output_dir: Path, stats: Any, neurips_dataset: str) -> None:
    (output_dir / "README.md").write_text(
        f"""---
license: other
language:
- en
pretty_name: NeurIPS 2025 Paper Sources and OpenReview Supplements
tags:
- arxiv
- openreview
- neurips
- parquet
- scholarly-documents
size_categories:
- 1K<n<10K
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*.parquet
---

# NeurIPS 2025 Paper Sources and OpenReview Supplements

One row per arXiv paper. Each row groups decoded paper `.tex` files with the
OpenReview supplementary material files matched by OpenReview note id.

## Provenance

ArXiv ids and titles come pre-matched from the
[`{neurips_dataset}`](https://huggingface.co/datasets/{neurips_dataset})
dataset. Papers without a pre-matched arxiv id are excluded; no fuzzy title
matching is performed anywhere in the pipeline. `arxiv_id_source` records how
the upstream dataset derived each match, and `openreview_id` is the OpenReview
forum id used to fetch supplements.

## Snapshot

- Papers: {stats.papers:,}
- Papers with `.tex`: {stats.papers_with_tex:,}
- Papers with supplements: {stats.papers_with_supplement:,}
- Papers skipped (no downloaded source): {stats.papers_skipped_no_source:,}
- Paper `.tex` files: {stats.tex_files:,}
- Supplement files: {stats.supplement_files:,}
- Logical bytes: {stats.bytes:,}
- Parquet shards: {stats.shards:,}

## Notes

Licensing varies by paper and supplementary archive. Check the arXiv record,
OpenReview page, and included files before redistribution or reuse.
""",
        encoding="utf-8",
    )


def write_bundle_stats(output_dir: Path, stats: Any, config: dict[str, Any]) -> None:
    payload = stats.__dict__ | config
    (output_dir / "dataset_stats.json").write_text(json.dumps(payload, indent=2) + "\n")


def stage_upload(bundle_dir: Path, *, repo_id: str | None, private: bool,
                 commit_message: str) -> None:
    if not repo_id:
        raise SystemExit(
            "--upload needs a destination: pass --repo-id <owner>/<name> or set "
            "$RECLAIM_BUNDLE_DATASET."
        )
    validate_dataset_folder(bundle_dir)
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True, private=private)
    api.upload_folder(
        folder_path=str(bundle_dir),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit_message,
    )
    print(f"Uploaded {bundle_dir} to https://huggingface.co/datasets/{repo_id}")


def validate_dataset_folder(path: Path) -> None:
    if not path.is_dir():
        raise SystemExit(f"Missing dataset folder: {path}")
    if not (path / "README.md").is_file():
        raise SystemExit(f"Missing dataset card: {path / 'README.md'}")
    shards = sorted((path / "data").glob("*.parquet"))
    if not shards:
        raise SystemExit(f"No Parquet shards found under {path / 'data'}")
    print(f"Found {len(shards)} Parquet shard(s) under {path / 'data'}")
