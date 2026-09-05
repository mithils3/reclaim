"""Lockfile loading + per-row field accessors for the reproduce episodes.

The release ships the lockfile as two JSONL files under ``data/splits``:
``eval_100.jsonl`` (the 100-paper frozen benchmark, ``split="eval"`` in-row) and
``dev_14.jsonl`` (the disjoint 14-paper dev split, ``split="dev"`` in-row). Those
files are the default source. Set ``RECLAIM_SPLITS_DIR`` to read them from
somewhere else.

``load_lockfile_rows`` accepts, in priority order:

* nothing (``None``) -- the shipped split file for the requested split;
* a bare Hugging Face dataset repo id (``owner/name``) loaded with
  ``datasets.load_dataset`` at the requested split;
* an ``hf://datasets/<owner>/<name>/<file>`` reference (a loose file on the Hub);
* a local ``.jsonl`` path.

Split names follow the Hugging Face convention (``test`` for the frozen
benchmark, ``validation`` for dev) and the aliases ``eval``/``dev`` are accepted.
The selector applies to the default files and to the bare-repo path; a loose file
or a local ``.jsonl`` is read whole. Everything here is pure row plumbing, with no
episode state.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from reclaim_vllm.audit import h100
from reclaim_vllm.runtime.mre_records import load_mre_records

# The reproduction agent reproduces the frozen benchmark by default; "validation"
# (the 14-paper dev split) is for development. "train" does not exist here.
DEFAULT_LOCKFILE_SPLIT = "test"
_SPLIT_ALIASES = {"eval": "test", "eval100": "test", "dev": "validation"}
_SPLIT_FILES = {"test": "eval_100.jsonl", "validation": "dev_14.jsonl"}
ENV_SPLITS_DIR = "RECLAIM_SPLITS_DIR"


def normalize_split(name: str | None) -> str:
    """Map friendly split aliases (eval/dev) to the dataset's real split names."""
    key = str(name or "").strip().lower()
    return _SPLIT_ALIASES.get(key, key) or DEFAULT_LOCKFILE_SPLIT


def splits_dir() -> Path:
    """Directory holding the shipped split files (``$RECLAIM_SPLITS_DIR`` wins)."""
    override = os.environ.get(ENV_SPLITS_DIR)
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / "data" / "splits"


def default_split_path(split: str = DEFAULT_LOCKFILE_SPLIT) -> Path:
    """Path of the shipped JSONL for ``split``."""
    name = normalize_split(split)
    try:
        return splits_dir() / _SPLIT_FILES[name]
    except KeyError:
        raise SystemExit(
            f"Unknown split {split!r}: expected one of {sorted(_SPLIT_FILES)} "
            "(aliases 'eval'/'dev' are accepted)."
        ) from None


def load_lockfile_rows(
    source: str | None = None, *, split: str = DEFAULT_LOCKFILE_SPLIT
) -> dict[str, dict]:
    """Return ``{arxiv_id: row}`` from the shipped splits, an HF dataset, or JSONL."""
    spec = str(source or "").strip()
    if not spec:
        path = default_split_path(split)
        if not path.exists():
            raise SystemExit(
                f"Lockfile split file not found: {path}. Pass --lockfile, or point "
                f"${ENV_SPLITS_DIR} at the directory holding {sorted(_SPLIT_FILES.values())}."
            )
        return load_mre_records(path)
    if _looks_like_dataset_repo(spec):
        return _index_rows(_iter_hf_dataset(spec, normalize_split(split)))
    # Local .jsonl path or hf://datasets/<owner>/<name>/<file>: reuse the tested
    # MRE-record file loader the auditor already shares.
    return load_mre_records(spec)


def _looks_like_dataset_repo(spec: str) -> bool:
    if spec.startswith("hf://") or Path(spec).exists():
        return False
    if spec.endswith((".jsonl", ".json")):
        return False
    return "/" in spec


def _iter_hf_dataset(repo_id: str, split: str) -> Iterable[dict]:
    from datasets import load_dataset

    return iter(load_dataset(repo_id, split=split))


def _index_rows(rows: Iterable[dict]) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    for row in rows:
        arxiv_id = arxiv_id_of(row)
        if arxiv_id:
            indexed[arxiv_id] = dict(row)
    if not indexed:
        raise SystemExit("No lockfile rows found (every row lacked an arXiv id).")
    return indexed


def arxiv_id_of(row: dict) -> str:
    for key in ("custom_id", "paper_id", "arxiv_id"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def band_of(row: dict) -> str:
    for key in ("selection_band", "h100_band"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "(unspecified)"


def band_max_hours(row: dict) -> float | None:
    """Upper edge of the row's compute band in H100-h (e.g. ``'96-192'`` -> 192.0,
    ``'0-8'`` -> 8.0). Returns ``None`` when the band is unspecified/unparseable.

    The ladder itself lives in ``reclaim_vllm.audit.h100`` (which assigns the
    labels), so the edges are read from it rather than recovered from the label
    text. A label off that ladder still falls back to reading its trailing edge, so
    a lockfile carrying a band this build does not know about keeps working.
    """
    label = band_of(row)
    edge = h100.band_max_hours(label)
    if edge is not None:
        return edge
    try:
        return float(label.split("-")[-1])
    except (ValueError, IndexError):
        return None


def format_hours(hours: float) -> str:
    return f"{float(hours):g}"
