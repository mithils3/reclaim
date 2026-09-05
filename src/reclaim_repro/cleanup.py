"""Post-episode workspace pruning: reclaim accidental large downloads.

The agent's ``workspace/`` (bound to ``/repro/workspace`` in the sandbox) is NVMe
scratch it clones code and builds a venv into. It is also where a mis-issued
``load_dataset`` / ``wget`` / ``git clone`` of a full dataset lands, and those can
be tens of GB. Nothing downstream needs them: the auditor grades
``report.json`` plus the durable ``evidence/`` trajectory, never the workspace
scratch, and the telemetry sink only uploads ``stats.json`` + ``agent.full.log``
from ``evidence/``. So once an episode reaches a terminal state we prune the
workspace's big artifacts to keep the run root (and the shared filesystem) small.

Three steps, all confined to ``workspace/`` (``reference/`` and ``evidence/`` are
never touched) and all skipping symlinks so we only ever delete a link, never the
thing it points at:

  1. delete the agent's Python venv(s) whole — the prompt tells it to build a
     ``.venv`` and install a multi-GB aarch64 CUDA torch into it; that install is
     the single biggest reclaimable artifact and must go as a unit (the size
     passes below would only shred it partially). Venvs are found by their
     definitive ``pyvenv.cfg`` marker, so a repo-nested or oddly-named one counts.
  2. delete any single regular file larger than the threshold, then
  3. delete the *tightest* subdirectory whose total still exceeds the threshold —
     the deepest qualifying dir, so a sharded dataset dir (many sub-threshold
     shards) is removed whole while the enclosing cloned code repo survives.

Deletion is best-effort and logged with a path + size line each; a filesystem
error on one entry is reported and skipped, never raised, so pruning can never
strand an otherwise-finalized run.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import TextIO

# Files/dirs at or above this many MB are pruned from the workspace after an
# episode finishes. Tunable per run via --prune-workspace-threshold-mb (0 = off).
DEFAULT_PRUNE_THRESHOLD_MB = 500


def _human(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TiB"


def _tree_size(root: Path) -> int:
    """Recursive byte size of ``root``'s regular files, not following symlinks."""
    total = 0
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=False):
        for name in filenames:
            fp = os.path.join(dirpath, name)
            if os.path.islink(fp):
                continue
            try:
                total += os.path.getsize(fp)
            except OSError:
                continue
    return total


def _find_venvs(workspace: Path) -> list[Path]:
    """Every Python venv under ``workspace``, identified by its ``pyvenv.cfg``.

    Does not descend into a venv once found, so a venv-inside-a-venv counts once.
    """
    venvs: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(workspace, followlinks=False):
        if os.path.islink(dirpath):
            dirnames[:] = []
            continue
        if "pyvenv.cfg" in filenames:
            venvs.append(Path(dirpath))
            dirnames[:] = []  # a venv's internals are never themselves a venv to prune
    return venvs


def _dir_sizes(
    workspace: Path, threshold: int | None = None
) -> tuple[dict[str, int], list[tuple[Path, int]]]:
    """Recursive byte size of every directory under (and including) ``workspace``, plus
    every regular file larger than ``threshold`` bytes.

    One walk serves both size passes. Totalling the directories has to stat every file
    anyway, so the oversized files are picked up on the way through instead of costing a
    second full stat-walk of the tree on the finalize path. The walk is top-down (the
    order the oversized files are reported and deleted in) and the totals are
    accumulated children-first afterwards, so each dir's total is its own regular files
    plus its already-summed immediate subdirs. Symlinks — files and dirs — are skipped,
    never followed.
    """
    own: dict[str, int] = {}
    children: dict[str, list[str]] = {}
    oversized: list[tuple[Path, int]] = []
    top: str | None = None
    for dirpath, dirnames, filenames in os.walk(workspace, followlinks=False):
        if top is None:
            top = dirpath
        total = 0
        for name in filenames:
            fp = os.path.join(dirpath, name)
            if os.path.islink(fp):
                continue
            try:
                size = os.path.getsize(fp)
            except OSError:
                continue
            total += size
            if threshold is not None and size > threshold:
                oversized.append((Path(fp), size))
        own[dirpath] = total
        children[dirpath] = [
            sub
            for sub in (os.path.join(dirpath, name) for name in dirnames)
            if not os.path.islink(sub)
        ]
    sizes: dict[str, int] = {}
    for dirpath in _children_first(top, children) if top is not None else ():
        sizes[dirpath] = own[dirpath] + sum(sizes.get(sub, 0) for sub in children[dirpath])
    return sizes, oversized


def _children_first(top: str, children: dict[str, list[str]]) -> list[str]:
    """``top``'s subtree with every directory after its own children.

    The order ``os.walk(topdown=False)`` yields, rebuilt from the single top-down walk:
    the size accumulation needs each dir summed before its parent, and the prune log
    reports big directories in this order. Dirs the walk never yielded (unreadable) are
    skipped, exactly as ``os.walk`` skips them.
    """
    order: list[str] = []
    stack: list[tuple[str, bool]] = [(top, False)]
    while stack:
        node, summed = stack.pop()
        if summed:
            order.append(node)
            continue
        stack.append((node, True))
        for sub in reversed(children[node]):
            if sub in children:
                stack.append((sub, False))
    return order


def _discount(sizes: dict[str, int], path: Path, root: Path, size: int) -> None:
    """Take a just-deleted file's bytes off every directory total that counted it.

    Keeps ``sizes`` describing the tree as it stands *now*, which is what the
    deepest-big-dir pass needs: it runs once the oversized files are already gone.
    """
    for parent in path.parents:
        key = str(parent)
        if key in sizes:
            sizes[key] -= size
        if parent == root:
            break


def _deepest_big_dirs(
    sizes: dict[str, int], workspace: Path, threshold: int
) -> list[tuple[Path, int]]:
    """Subdirectories whose total exceeds ``threshold`` and that contain no such
    subdirectory themselves — the tightest big subtree along each path.

    ``sizes`` comes from :func:`_dir_sizes`, discounted for whatever the oversized-file
    pass already deleted. Excludes ``workspace`` itself: we prune its contents, never
    the root.
    """
    root = str(workspace)
    qualifying = [d for d, size in sizes.items() if d != root and size > threshold]
    quals = set(qualifying)
    targets = []
    for d in qualifying:
        prefix = d + os.sep
        if not any(other != d and other.startswith(prefix) for other in quals):
            targets.append((Path(d), sizes[d]))
    return targets


def prune_workspace(
    workspace: Path | str | None,
    threshold_mb: float,
    *,
    remove_venv: bool = True,
    log: TextIO = sys.stderr,
    label: str = "",
) -> int:
    """Delete the venv(s), big files, and big subdirs from ``workspace``; return
    bytes freed.

    A missing/non-directory workspace is a no-op. ``threshold_mb`` of 0 disables the
    size passes but ``remove_venv`` still applies; ``remove_venv=False`` with
    ``threshold_mb=0`` is a full no-op.
    """
    if not workspace:
        return 0
    root = Path(workspace)
    if not root.is_dir():
        return 0
    tag = f"prune {label}: " if label else "prune: "
    deleted: list[tuple[Path, int]] = []

    # Step 1: the agent's venv(s), removed whole (torch + deps are the biggest
    # single reclaim; the size passes would only shred them partially).
    if remove_venv:
        for venv in _find_venvs(root):
            size = _tree_size(venv)
            if _remove(venv, is_dir=True, tag=tag, log=log):
                deleted.append((venv, size))

    if threshold_mb > 0:
        threshold = int(threshold_mb * 1024 * 1024)
        # One walk feeds both size passes (it stats every file either way), and each
        # deleted file is discounted from its parents so step 3 still sees the tree as
        # it stands after step 2.
        sizes, oversized = _dir_sizes(root, threshold)
        # Step 2: individual oversized files (anywhere in the tree, symlinks skipped).
        for fp, size in oversized:
            if _remove(fp, is_dir=False, tag=tag, log=log):
                deleted.append((fp, size))
                _discount(sizes, fp, root, size)

        # Step 3: tightest subdirs still over threshold once big files are gone.
        for path, size in _deepest_big_dirs(sizes, root, threshold):
            if _remove(path, is_dir=True, tag=tag, log=log):
                deleted.append((path, size))

    freed = sum(size for _, size in deleted)
    for path, size in deleted:
        try:
            rel = path.relative_to(root)
        except ValueError:
            rel = path
        print(f"{tag}deleted {rel} ({_human(size)})", file=log)
    if deleted:
        print(f"{tag}reclaimed {_human(freed)} across {len(deleted)} item(s)", file=log)
    return freed


def _remove(path: Path, *, is_dir: bool, tag: str, log: TextIO) -> bool:
    try:
        if is_dir:
            shutil.rmtree(path)
        else:
            path.unlink()
    except OSError as exc:
        print(f"{tag}could not delete {path} ({exc}); skipping", file=log)
        return False
    return True


__all__ = ["DEFAULT_PRUNE_THRESHOLD_MB", "prune_workspace"]
