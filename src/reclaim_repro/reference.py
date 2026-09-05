"""Materialize one paper's read-only reference directory from the paper bundle.

The live repro harness calls :func:`materialize_reference` to write a single
paper's sources into::

    <dest>/latex/<relative_path>        # paper .tex sources (UTF-8 text)
    <dest>/supplement/<relative_path>   # OpenReview supplement, raw bytes
    <dest>/info.json                    # ids, urls, statuses, file counts

Supplement files are written from their raw ``content`` bytes, NOT from the
``text`` field: the bundle builder only fills ``text`` for a narrow extension set
(``.py`` and most code files are excluded), but ``content`` holds the bytes for
every file. LaTeX files only carry ``text`` (always decodable), so those are
written as UTF-8.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# The paper-bundle dataset carrying each paper's LaTeX sources and OpenReview
# supplement. It is not shipped with the code (the bundles are large and their
# licences are the authors' own), so the repo id is supplied by the operator.
ENV_BUNDLE_DATASET = "RECLAIM_BUNDLE_DATASET"
DEFAULT_DATASET = os.environ.get(ENV_BUNDLE_DATASET, "")


def write_paper(row: dict, paper_dir: Path) -> dict:
    """Materialize one bundle row into ``paper_dir`` and return its counts."""
    latex_dir = paper_dir / "latex"
    supp_dir = paper_dir / "supplement"
    n_tex = write_tex_files(row.get("paper_tex_files") or [], latex_dir)
    n_supp = write_supplement_files(row.get("supplement_files") or [], supp_dir)
    write_info(row, paper_dir, n_tex, n_supp)
    write_manifest(paper_dir)
    return {"arxiv_id": row.get("arxiv_id"), "latex_files": n_tex, "supplement_files": n_supp}


def write_tex_files(items: list, latex_dir: Path) -> int:
    count = 0
    for item in items:
        rel = item.get("relative_path") or item.get("filename")
        target = safe_target(latex_dir, rel)
        if target is None:
            continue
        text = item.get("text")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text if isinstance(text, str) else "", encoding="utf-8")
        count += 1
    return count


def write_supplement_files(items: list, supp_dir: Path) -> int:
    count = 0
    for item in items:
        rel = item.get("relative_path") or item.get("filename")
        target = safe_target(supp_dir, rel)
        if target is None:
            continue
        data = supplement_bytes(item)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        count += 1
    return count


def supplement_bytes(item: dict) -> bytes:
    content = item.get("content")
    if isinstance(content, (bytes, bytearray)):
        return bytes(content)
    text = item.get("text")
    if isinstance(text, str):
        return text.encode("utf-8")
    return b""


def write_info(row: dict, paper_dir: Path, n_tex: int, n_supp: int) -> None:
    info = {
        "arxiv_id": row.get("arxiv_id"),
        "title": row.get("title"),
        "openreview_id": row.get("openreview_id"),
        "paper_source_url": row.get("paper_source_url"),
        "paper_status": row.get("paper_status"),
        "supplement_source_url": row.get("supplement_source_url"),
        "supplement_status": row.get("supplement_status"),
        "latex_files": n_tex,
        "supplement_files": n_supp,
    }
    paper_dir.mkdir(parents=True, exist_ok=True)
    (paper_dir / "info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")


def write_manifest(paper_dir: Path) -> Path:
    """List every materialized reference file (latex/ + supplement/) with sizes."""
    lines = [f"REFERENCE MANIFEST for {paper_dir.name}", ""]
    total = 0
    for sub in ("latex", "supplement"):
        root = paper_dir / sub
        files = sorted(p for p in root.rglob("*") if p.is_file()) if root.is_dir() else []
        lines.append(f"{sub}/ ({len(files)} file(s)):")
        for path in files:
            size = path.stat().st_size
            total += size
            lines.append(f"  {path.relative_to(paper_dir)}  ({size} bytes)")
        lines.append("")
    lines.append(f"{total} bytes total across latex/ and supplement/.")
    manifest = paper_dir / "MANIFEST.txt"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def materialize_reference(
    arxiv_id: str,
    dest: Path,
    *,
    dataset: str = DEFAULT_DATASET,
    overwrite: bool = False,
    row: dict | None = None,
) -> dict:
    """Write one paper's read-only ``reference/`` dir (latex + supplement + manifest).

    ``row`` lets callers (and tests) skip the network; otherwise the bundle is
    streamed until ``arxiv_id`` is found. Returns the per-paper counts plus an
    ``ok`` flag so the workspace setup can report what landed.
    """
    dest = Path(dest)
    if dest.exists() and any(dest.iterdir()) and not overwrite:
        return {"ok": True, "skipped": True, "reason": "reference already materialized", "dest": str(dest)}
    if row is None and not dataset:
        return {
            "ok": False,
            "error": f"no paper-bundle dataset configured (set ${ENV_BUNDLE_DATASET})",
            "dest": str(dest),
        }
    if row is None:
        row = find_bundle_row(arxiv_id, dataset=dataset)
    if row is None:
        return {"ok": False, "error": f"arxiv_id {arxiv_id!r} not found in bundle {dataset!r}", "dest": str(dest)}
    counts = write_paper(row, dest)
    return {"ok": True, "skipped": False, "dest": str(dest), **counts}


def find_bundle_row(arxiv_id: str, *, dataset: str = DEFAULT_DATASET) -> dict | None:
    """Return the bundle row whose arXiv id matches, from the non-streaming dataset.

    Reads the lightweight ``arxiv_id`` column to locate the row, then materializes
    just that one row's heavy ``content`` bytes — so finding one paper never pulls
    every row's payload into memory. The load is non-streaming: it downloads and
    caches every shard once, shows HF download progress, and serves later runs from
    local memory-mapped reads. Point ``HF_HOME``/``HF_HUB_CACHE`` at scratch rather
    than ``$HOME``, because the full bundle is large.
    """
    from datasets import load_dataset

    ds = load_dataset(dataset, split="train")
    # Match ignoring any trailing version suffix (``2505.11483v2`` is ``2505.11483``).
    wanted = str(arxiv_id or "").strip().split("v")[0]
    for index, row_id in enumerate(ds["arxiv_id"]):
        text = str(row_id or "").strip()
        if text and text.split("v")[0] == wanted:
            return dict(ds[index])
    return None


def safe_target(base: Path, rel: object) -> Path | None:
    """Join ``rel`` under ``base``, rejecting absolute paths and traversal."""
    raw = str(rel or "").strip().replace("\\", "/")
    if not raw or raw.startswith("/"):
        return None
    parts = [p for p in raw.split("/") if p not in ("", ".")]
    if any(p == ".." for p in parts):
        return None
    if not parts:
        return None
    return base.joinpath(*parts)


def safe_component(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_").strip() or "unknown"
