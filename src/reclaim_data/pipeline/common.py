"""Shared constants, HTTP/archive helpers, and manifest IO for the dataset pipeline."""

from __future__ import annotations

import csv
import shutil
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


STAGES = ("index", "sources", "supplements", "bundle", "upload")
INDEX_FILENAME = "neurips2025_index.csv"
SOURCES_DIRNAME = "arxiv_sources"
SUPPLEMENTS_DIRNAME = "openreview_supplements"
BUNDLE_DIRNAME = "paper_bundle_dataset"
NOTES_CACHE_FILENAME = "openreview_notes.json"
MANIFEST_FILENAME = "manifest.csv"

USER_AGENT = "reclaim-arxiv-source-downloader/0.1"
RETRYABLE_HTTP_CODES = {408, 429, 500, 502, 503, 504}


@dataclass(frozen=True)
class DownloadResult:
    index: int
    arxiv_id: str
    title: str
    source_url: str
    status: str
    output_path: str
    files_written: int
    bytes_downloaded: int
    error: str = ""


class RequestThrottle:
    def __init__(self, interval_seconds: float) -> None:
        self.interval_seconds = interval_seconds
        self.lock = threading.Lock()
        self.next_at = 0.0

    def wait(self) -> None:
        if self.interval_seconds <= 0:
            return
        with self.lock:
            now = time.monotonic()
            wait_for = max(0.0, self.next_at - now)
            self.next_at = max(now, self.next_at) + self.interval_seconds
        if wait_for:
            time.sleep(wait_for)


def http_get_bytes(url: str, retries: int, timeout: float) -> tuple[bytes, str]:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                content_type = header_value(response.headers.get("Content-Type", ""))
                return response.read(), content_type
        except urllib.error.HTTPError as error:
            last_error = error
            if error.code not in RETRYABLE_HTTP_CODES:
                raise RuntimeError(f"HTTP {error.code}") from None
            sleep_seconds = retry_delay(error, attempt)
        except urllib.error.URLError as error:
            last_error = error
            sleep_seconds = min(2**attempt, 30)

        if attempt < retries:
            time.sleep(sleep_seconds)
    raise RuntimeError(f"GET failed after {retries + 1} tries") from last_error


def retry_delay(error: urllib.error.HTTPError, attempt: int) -> int:
    retry_after = error.headers.get("Retry-After")
    if retry_after and retry_after.isdigit():
        return min(int(retry_after), 60)
    return min(2**attempt, 30)


def header_value(value: str | bytes) -> str:
    if isinstance(value, bytes):
        return value.decode("latin-1", errors="replace")
    return value


def looks_like_html(data: bytes, content_type: str) -> bool:
    if "text/html" in content_type.casefold():
        return True
    prefix = data[:512].lstrip().lower()
    return prefix.startswith(b"<!doctype html") or prefix.startswith(b"<html")


def extract_tar_members(archive: tarfile.TarFile, output_dir: Path) -> int:
    files_written = 0
    for member in archive.getmembers():
        target = safe_archive_target(output_dir, member.name)
        if target is None:
            continue
        if member.isdir():
            target.mkdir(parents=True, exist_ok=True)
        elif member.isfile():
            source = archive.extractfile(member)
            if source is None:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with source, target.open("wb") as handle:
                shutil.copyfileobj(source, handle)
            files_written += 1
    return files_written


def safe_archive_target(output_dir: Path, member_name: str) -> Path | None:
    pure_path = PurePosixPath(member_name.replace("\\", "/"))
    if pure_path.is_absolute() or ".." in pure_path.parts:
        return None
    parts = [part for part in pure_path.parts if part not in {"", "."}]
    return output_dir.joinpath(*parts) if parts else None


def write_single_source_file(data: bytes, output_dir: Path, filename: str) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / filename).write_bytes(data)
    return 1


def safe_dirname(arxiv_id: str) -> str:
    return arxiv_id.replace("/", "_")


def count_files(path: Path) -> int:
    return sum(1 for item in path.rglob("*") if item.is_file())


def arxiv_eprint_url(arxiv_id: str) -> str:
    return f"https://arxiv.org/e-print/{urllib.parse.quote(arxiv_id, safe='/')}"


def make_result(
    index: int,
    arxiv_id: str,
    title: str,
    source_url: str,
    status: str,
    target: Path,
    files_written: int,
    bytes_downloaded: int,
    error: str = "",
) -> DownloadResult:
    return DownloadResult(
        index=index,
        arxiv_id=arxiv_id,
        title=title,
        source_url=source_url,
        status=status,
        output_path=str(target),
        files_written=files_written,
        bytes_downloaded=bytes_downloaded,
        error=error,
    )


def write_manifest(path: Path, results: list[DownloadResult]) -> None:
    fields = [
        "index",
        "arxiv_id",
        "title",
        "source_url",
        "status",
        "output_path",
        "files_written",
        "bytes_downloaded",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(result.__dict__)


def load_manifest_map(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["arxiv_id"]: row for row in csv.DictReader(handle) if row.get("arxiv_id")}


def print_stage_summary(stage: str, manifest_path: Path, results: list[DownloadResult]) -> None:
    failures = sum(result.status == "failed" for result in results)
    print(
        f"{stage}: wrote manifest to {manifest_path}. "
        f"{len(results) - failures} succeeded/skipped, {failures} failed.",
        file=sys.stderr,
    )
