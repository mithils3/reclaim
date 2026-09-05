"""Supplements stage: match papers to OpenReview notes by note id and download attachments.

Matching uses the OpenReview forum id captured in the index; there is no
title-based fallback. This module also holds the OpenReview client, the
note/job types, and the supplement attachment downloader.
"""

from __future__ import annotations

import csv
import io
import json
import os
import shutil
import sys
import tarfile
import tempfile
import threading
import time
import urllib.parse
import zipfile
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import (
    MANIFEST_FILENAME,
    NOTES_CACHE_FILENAME,
    SUPPLEMENTS_DIRNAME,
    DownloadResult,
    RequestThrottle,
    count_files,
    extract_tar_members,
    make_result,
    print_stage_summary,
    safe_archive_target,
    safe_dirname,
    write_manifest,
    write_single_source_file,
)
from .index import IndexRecord


def stage_supplements(
    records: list[IndexRecord],
    data_dir: Path,
    *,
    notes_json: Path | None,
    api_base: str,
    venue_id: str,
    workers: int,
    delay: float,
    retries: int,
    overwrite: bool,
) -> list[DownloadResult]:
    output_dir = data_dir / SUPPLEMENTS_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)
    notes_path = notes_json or (data_dir / NOTES_CACHE_FILENAME)
    notes = load_or_fetch_notes(notes_path, api_base, venue_id)
    notes_by_id = index_notes_by_id(notes)
    jobs, missing_id, missing_note, no_supplement = match_supplement_jobs(records, notes_by_id)
    print(
        f"Matched {len(jobs)} OpenReview supplements from {len(records)} papers "
        f"and {len(notes)} notes ({missing_id} without an OpenReview id, "
        f"{missing_note} ids not found, {no_supplement} notes without supplements).",
        file=sys.stderr,
    )
    write_job_csv(output_dir / "openreview_jobs.csv", jobs)
    results = download_supplements(
        jobs=jobs,
        output_dir=output_dir,
        retries=max(0, retries),
        api_base=api_base,
        overwrite=overwrite,
        delay=max(0.0, delay),
        workers=max(1, workers),
    )
    write_manifest(output_dir / MANIFEST_FILENAME, results)
    print_stage_summary("supplements", output_dir / MANIFEST_FILENAME, results)
    return results


def load_or_fetch_notes(path: Path, api_base: str, venue_id: str) -> list[Note]:
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_notes = payload.get("notes", payload) if isinstance(payload, dict) else payload
        print(f"Loaded {len(raw_notes)} cached OpenReview notes from {path}.", file=sys.stderr)
    else:
        raw_notes = fetch_openreview_notes(build_client(api_base), venue_id)
        print(f"Fetched {len(raw_notes)} OpenReview notes for {venue_id}.", file=sys.stderr)
        try:
            path.write_text(
                json.dumps({"notes": [note_payload(note) for note in raw_notes]}),
                encoding="utf-8",
            )
            print(f"Cached notes to {path}.", file=sys.stderr)
        except (TypeError, ValueError, OSError) as error:
            print(f"Could not cache notes to {path}: {error}", file=sys.stderr)
    return [parse_note(raw) for raw in raw_notes]


def note_payload(raw: Any) -> Any:
    if isinstance(raw, dict):
        return raw
    to_json = getattr(raw, "to_json", None)
    if callable(to_json):
        return to_json()
    return raw.__dict__


def parse_note(raw: Any) -> Note:
    if isinstance(raw, dict):
        content = raw.get("content") or {}
        note_id = str(raw.get("id") or "")
        forum = str(raw.get("forum") or "")
    else:
        content = getattr(raw, "content", {}) or {}
        note_id = str(getattr(raw, "id", "") or "")
        forum = str(getattr(raw, "forum", "") or "")
    return Note(note_id=note_id, forum=forum, content=content)


def index_notes_by_id(notes: Iterable[Note]) -> dict[str, Note]:
    notes_by_id: dict[str, Note] = {}
    for note in notes:
        if note.note_id:
            notes_by_id.setdefault(note.note_id, note)
        if note.forum:
            notes_by_id.setdefault(note.forum, note)
    return notes_by_id


def match_supplement_jobs(
    records: list[IndexRecord],
    notes_by_id: dict[str, Note],
) -> tuple[list[SupplementJob], int, int, int]:
    jobs: list[SupplementJob] = []
    missing_id = missing_note = no_supplement = 0
    for index, record in enumerate(records, start=1):
        if not record.openreview_id:
            missing_id += 1
            continue
        note = notes_by_id.get(record.openreview_id)
        if note is None:
            missing_note += 1
            continue
        attachment = supplement_attachment(note)
        if not attachment:
            no_supplement += 1
            continue
        field_name, source_url = attachment
        jobs.append(
            SupplementJob(
                index,
                record.arxiv_id,
                record.title,
                note.note_id,
                field_name,
                source_url,
            )
        )
    return jobs, missing_id, missing_note, no_supplement


def supplement_attachment(note: Note) -> tuple[str, str] | None:
    for key, raw_value in note.content.items():
        if "supplement" not in key.casefold():
            continue
        value = content_value(raw_value)
        if isinstance(value, str) and value:
            return key, absolute_openreview_url(value)
        return key, attachment_url(note.note_id, key)
    return None


def content_value(value: Any) -> Any:
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


OPENREVIEW_API = "https://api2.openreview.net"
OPENREVIEW_SITE = "https://openreview.net"
DEFAULT_VENUE_ID = "NeurIPS.cc/2025/Conference"
THREAD_LOCAL = threading.local()


@dataclass(frozen=True)
class Note:
    note_id: str
    forum: str
    content: dict[str, Any]


@dataclass(frozen=True)
class SupplementJob:
    index: int
    arxiv_id: str
    title: str
    note_id: str
    attachment_name: str
    source_url: str


def build_client(api_base: str) -> Any:
    import openreview

    username = os.environ.get("OPENREVIEW_USERNAME")
    password = os.environ.get("OPENREVIEW_PASSWORD")
    return openreview.api.OpenReviewClient(
        baseurl=api_base,
        username=username,
        password=password,
    )


def thread_client(api_base: str) -> Any:
    client = getattr(THREAD_LOCAL, "openreview_client", None)
    if client is None:
        client = build_client(api_base)
        THREAD_LOCAL.openreview_client = client
    return client


def fetch_openreview_notes(client: Any, venue_id: str) -> list[Any]:
    return client.get_all_notes(content={"venueid": venue_id})


def attachment_url(note_id: str, field_name: str) -> str:
    query = urllib.parse.urlencode({"id": note_id, "name": field_name})
    return f"{OPENREVIEW_SITE}/attachment?{query}"


def absolute_openreview_url(value: str) -> str:
    if value.startswith("http://") or value.startswith("https://"):
        return value
    if value.startswith("/"):
        return f"{OPENREVIEW_SITE}{value}"
    return value


def download_supplements(
    jobs: list[SupplementJob],
    output_dir: Path,
    retries: int,
    api_base: str,
    overwrite: bool,
    delay: float,
    workers: int,
) -> list[DownloadResult]:
    results: list[DownloadResult] = []
    throttle = RequestThrottle(delay)
    # Submit everything: the pool caps concurrency at `workers` and the throttle
    # paces the OpenReview calls, so drip-feeding submissions bought nothing.
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [
            pool.submit(download_supplement_one, api_base, job, output_dir, retries, overwrite, throttle)
            for job in jobs
        ]
        for done, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            print(progress_line(done, len(jobs), result), file=sys.stderr)
    return sorted(results, key=lambda item: item.index)


def progress_line(done: int, total: int, result: DownloadResult) -> str:
    line = f"[{done}/{total}] {result.arxiv_id}: {result.status}"
    if result.error:
        return f"{line}: {result.error}"
    return line


def download_supplement_one(
    api_base: str,
    job: SupplementJob,
    output_dir: Path,
    retries: int,
    overwrite: bool,
    throttle: RequestThrottle,
) -> DownloadResult:
    target = output_dir / safe_dirname(job.arxiv_id)
    if target.is_dir() and any(target.iterdir()) and not overwrite:
        return make_result(
            job.index, job.arxiv_id, job.title, job.source_url, "skipped_existing", target, count_files(target), 0
        )
    tmp_dir = Path(tempfile.mkdtemp(dir=output_dir))
    try:
        client = thread_client(api_base)
        data = get_attachment_bytes(client, job, retries, throttle)
        files_written = unpack_archive(data, tmp_dir)
        if target.exists():
            shutil.rmtree(target)
        tmp_dir.rename(target)
        return make_result(
            job.index, job.arxiv_id, job.title, job.source_url, "downloaded", target, files_written, len(data)
        )
    except Exception as error:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return make_result(
            job.index, job.arxiv_id, job.title, job.source_url, "failed", target, 0, 0, str(error)
        )


def get_attachment_bytes(
    client: Any,
    job: SupplementJob,
    retries: int,
    throttle: RequestThrottle,
) -> bytes:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            throttle.wait()
            data = client.get_attachment(field_name=job.attachment_name, id=job.note_id)
            if isinstance(data, bytes):
                return data
            if isinstance(data, bytearray):
                return bytes(data)
            return str(data).encode("utf-8")
        except Exception as error:
            last_error = error
            message = str(error)
            if "404" in message or "not found" in message.casefold():
                raise RuntimeError(
                    f"OpenReview attachment not found: {job.attachment_name}"
                ) from None
            if attempt < retries:
                time.sleep(min(2**attempt, 60))
    raise RuntimeError(f"OpenReview attachment fetch failed: {last_error}") from last_error


def unpack_archive(data: bytes, output_dir: Path) -> int:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            return extract_zip_members(archive, output_dir)
    except zipfile.BadZipFile:
        pass
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as archive:
            return extract_tar_members(archive, output_dir)
    except tarfile.TarError:
        return write_single_source_file(data, output_dir, "supplementary_material.bin")


def extract_zip_members(archive: zipfile.ZipFile, output_dir: Path) -> int:
    files_written = 0
    for info in archive.infolist():
        target = safe_archive_target(output_dir, info.filename)
        if target is None or info.is_dir():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(info) as source, target.open("wb") as handle:
            shutil.copyfileobj(source, handle)
        files_written += 1
    return files_written


def write_job_csv(path: Path, jobs: list[SupplementJob]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SupplementJob.__dataclass_fields__))
        writer.writeheader()
        for job in jobs:
            writer.writerow(job.__dict__)
