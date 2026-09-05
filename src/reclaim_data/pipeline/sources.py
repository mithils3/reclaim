"""Sources stage: download and extract arXiv e-print packages."""

from __future__ import annotations

import gzip
import io
import shutil
import sys
import tarfile
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .common import (
    MANIFEST_FILENAME,
    SOURCES_DIRNAME,
    DownloadResult,
    RequestThrottle,
    count_files,
    extract_tar_members,
    http_get_bytes,
    looks_like_html,
    make_result,
    print_stage_summary,
    safe_dirname,
    write_manifest,
    write_single_source_file,
)
from .index import IndexRecord


def stage_sources(
    records: list[IndexRecord],
    data_dir: Path,
    *,
    workers: int,
    delay: float,
    retries: int,
    timeout: float,
    overwrite: bool,
    keep_archive: bool,
) -> list[DownloadResult]:
    output_dir = data_dir / SOURCES_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)
    throttle = RequestThrottle(max(0.0, delay))
    results: list[DownloadResult] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(
                download_source_one,
                index,
                record,
                output_dir,
                max(0, retries),
                max(1.0, timeout),
                overwrite,
                keep_archive,
                throttle,
            ): record
            for index, record in enumerate(records, start=1)
        }
        for done, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            print(f"[{done}/{len(futures)}] {result.arxiv_id}: {result.status}", file=sys.stderr)
    results = sorted(results, key=lambda result: result.index)
    write_manifest(output_dir / MANIFEST_FILENAME, results)
    print_stage_summary("sources", output_dir / MANIFEST_FILENAME, results)
    return results


def download_source_one(
    index: int,
    record: IndexRecord,
    output_dir: Path,
    retries: int,
    timeout: float,
    overwrite: bool,
    keep_archive: bool,
    throttle: RequestThrottle,
) -> DownloadResult:
    target = output_dir / safe_dirname(record.arxiv_id)
    existing = target.exists() and target.is_dir() and any(target.iterdir())
    if existing and not overwrite:
        return make_result(
            index,
            record.arxiv_id,
            record.title,
            record.source_url,
            "skipped_existing",
            target,
            count_files(target),
            0,
        )
    if target.exists() and not target.is_dir() and not overwrite:
        return make_result(
            index,
            record.arxiv_id,
            record.title,
            record.source_url,
            "failed",
            target,
            0,
            0,
            "output path exists and is not a directory",
        )

    tmp_dir = Path(tempfile.mkdtemp(dir=output_dir))
    try:
        throttle.wait()
        data, content_type = http_get_bytes(record.source_url, retries, timeout)
        if looks_like_html(data, content_type):
            raise RuntimeError("arXiv returned HTML instead of a source package")
        if keep_archive:
            (tmp_dir / f"{safe_dirname(record.arxiv_id)}.e-print").write_bytes(data)
        files_written = unpack_source_package(data, tmp_dir, record.arxiv_id)
        replace_with_tmp(target, tmp_dir)
        return make_result(
            index,
            record.arxiv_id,
            record.title,
            record.source_url,
            "downloaded",
            target,
            files_written,
            len(data),
        )
    except Exception as error:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return make_result(
            index,
            record.arxiv_id,
            record.title,
            record.source_url,
            "failed",
            target,
            0,
            0,
            str(error),
        )


def replace_with_tmp(target: Path, tmp_dir: Path) -> None:
    if target.exists():
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    tmp_dir.rename(target)


def unpack_source_package(data: bytes, output_dir: Path, arxiv_id: str) -> int:
    tar_count = try_extract_tar(data, output_dir)
    if tar_count is not None:
        return tar_count

    if data.startswith(b"\x1f\x8b"):
        decompressed = gzip.decompress(data)
        tar_count = try_extract_tar(decompressed, output_dir)
        if tar_count is not None:
            return tar_count
        if looks_like_html(decompressed, ""):
            raise RuntimeError("arXiv returned compressed HTML instead of source")
        return write_single_source_file(decompressed, output_dir, "source.tex")

    return write_single_source_file(data, output_dir, f"{safe_dirname(arxiv_id)}.source")


def try_extract_tar(data: bytes, output_dir: Path) -> int | None:
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as archive:
            return extract_tar_members(archive, output_dir)
    except tarfile.TarError:
        return None
