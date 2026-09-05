"""Path-confined write tools for the reproduction agent.

The agent *reads* files through ``workspace_bash`` (``grep -n`` / ``sed -n`` /
``cat`` -- targeted, line-numbered, and far cheaper than dumping whole files into
context), so this module ships only the mutating ops: ``write_file`` and
``edit_file``.

Adapted from ``run_dir_tools._resolve_within``: every path the agent writes must
resolve inside one of the episode's writable roots -- ``workspace`` (the editable
clone) and ``evidence``; the ``reference/`` copy is never writable. These are a
subset of the roots the Apptainer sandbox (``sandbox.py``) binds read-write — the file
tools stay tight to durable source/evidence, while bulk ``/tmp`` scratch is
shell-driven, not a file-tool path — so the tool-layer check is a strict inner
boundary nested inside the OS sandbox.

The agent works against the short in-container paths the sandbox remaps the episode
dirs onto (``/repro/workspace`` etc.); since these tools do real *host* I/O,
``_to_host`` first translates a leading ``/repro`` back to the host run dir. Relative
paths then resolve against the workspace (the same cwd ``workspace_bash`` runs in), and
host-absolute paths must still fall within an allowed root. ``..`` segments and NUL
bytes are rejected outright, and ``resolve()`` canonicalizes symlinks before the
containment check, so traversal, symlink escapes, and writes outside the bundle are
all rejected before any I/O happens.

``edit_file`` replaces the old diff-application tool (a ported Codex
context-matching patch engine): across sweeps it errored on 68% of its 66
calls -- agents pass repo-relative paths from inside the cloned repo, hunks
mis-locate, and models learned to route around it via whole-file ``write_file``
rewrites instead. ``edit_file`` trades diff hunks for one exact-string
replacement: no context matching to fuzz, and when the path itself is wrong (the
paper's repo cloned into ``workspace/<repo>/...`` but the model addresses it by a
bare repo-relative path) it suffix-searches the workspace tree for an unambiguous
match instead of failing outright. Ambiguity and no-match failures still get a
recovery hint (the suffix-search + mismatch-hint helpers below), same spirit as the
old engine's context-miss diagnostics.
"""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any

from reclaim_vllm.config.config import RUN_FILE_WRITE_MAX_CHARS, function_tool

from reclaim_repro.context import ExecutionContext
from reclaim_repro import evidence
from reclaim_repro.sandbox import CONTAINER_RUN

_SKIP_DIRS = {"venv", ".venv", "node_modules", "__pycache__"}
_MAX_SUFFIX_MATCHES = 10
_HINT_PAD = 3  # file lines quoted on each side of the closest match


def write_file(arguments: dict[str, Any], ctx: ExecutionContext) -> dict[str, Any]:
    raw_path = _first(arguments, "path", "file_path", "filename", "file")
    resolved = _resolve(ctx, raw_path)
    if not resolved["ok"]:
        return resolved
    target: Path = resolved["path"]
    if target.is_dir():
        return {"ok": False, "error": f"Path is a directory: {raw_path}"}
    content = _first(arguments, "content", "text", "contents", "data")
    if not isinstance(content, str):
        return {"ok": False, "error": "Missing 'content' string to write."}
    if len(content) > RUN_FILE_WRITE_MAX_CHARS:
        return {
            "ok": False,
            "error": (
                f"Content is {len(content)} chars, over the {RUN_FILE_WRITE_MAX_CHARS} limit. "
                "Write the file in pieces or generate it with a workspace_bash heredoc instead."
            ),
        }
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    except OSError as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"ok": True, "path": str(target), "bytes_written": len(content.encode("utf-8"))}


def edit_file(arguments: dict[str, Any], ctx: ExecutionContext) -> dict[str, Any]:
    raw_path = _first(arguments, "path", "file_path", "filename", "file")
    resolved = _resolve(ctx, raw_path)
    if not resolved["ok"]:
        return resolved
    target: Path = resolved["path"]
    resolved_note: str | None = None

    if not target.exists():
        located = resolve_by_suffix(ctx, raw_path)
        if located is not None:
            if not located["ok"]:
                if ctx.evidence is not None:
                    evidence.log_command(
                        ctx.evidence, f"edit_file {raw_path} ({located['reason']})",
                        returncode=1, cwd=ctx.workspace,
                    )
                return located["result"]
            # Re-run the containment check on the located file: rglob can walk
            # through a symlinked dir, and only ``_resolve`` canonicalizes before
            # bounding, so the inner boundary stays strict even for found paths.
            checked = _resolve(ctx, str(located["path"]))
            if not checked["ok"]:
                return checked
            target = checked["path"]
            resolved_note = str(target)
        elif not target.exists():
            return {"ok": False, "error": f"File does not exist: {raw_path}."}

    if target.is_dir():
        return {"ok": False, "error": f"Path is a directory: {raw_path}"}

    old_string = _first(arguments, "old_string", "old_str", "old", "find", "search")
    if not isinstance(old_string, str) or old_string == "":
        return {
            "ok": False,
            "error": "Missing non-empty 'old_string' to match against the file text. "
                     "Use write_file to create a new file.",
        }
    new_string = _first_str(arguments, "new_string", "new_str", "new", "replacement", "replace")
    if new_string is None:
        return {"ok": False, "error": "Missing 'new_string' (pass \"\" to delete the matched text)."}
    if old_string == new_string:
        return {"ok": False, "error": "old_string and new_string are identical -- nothing to change."}
    replace_all = bool(arguments.get("replace_all") or arguments.get("all"))

    try:
        before = target.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    count = before.count(old_string)
    if count == 0:
        if ctx.evidence is not None:
            evidence.log_command(ctx.evidence, f"edit_file {raw_path} (no match)", returncode=1, cwd=ctx.workspace)
        hint = mismatch_hint(before, old_string)
        error = f"old_string not found in {raw_path}."
        return {"ok": False, "error": f"{error} {hint}".strip()}
    if count > 1 and not replace_all:
        if ctx.evidence is not None:
            evidence.log_command(ctx.evidence, f"edit_file {raw_path} (ambiguous match)", returncode=1, cwd=ctx.workspace)
        return {
            "ok": False,
            "error": f"old_string occurs {count} times in {raw_path} -- give a larger, "
                     "unique snippet or pass replace_all=true.",
        }

    after = before.replace(old_string, new_string, -1 if replace_all else 1)
    try:
        target.write_text(after, encoding="utf-8")
    except OSError as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    try:  # diff header only, so it never needs to be exact
        rel_display = str(target.relative_to(Path(ctx.workspace)))
    except (ValueError, TypeError):
        rel_display = target.name
    diff_text = "".join(difflib.unified_diff(
        before.splitlines(keepends=True), after.splitlines(keepends=True),
        fromfile=rel_display, tofile=rel_display,
    ))
    saved = evidence.save_patch(ctx.evidence, diff_text, name=target.name) if ctx.evidence and diff_text else None
    if ctx.evidence is not None:
        evidence.log_command(ctx.evidence, f"edit_file {rel_display}", returncode=0, cwd=ctx.workspace)

    result: dict[str, Any] = {
        "ok": True,
        "path": str(target),
        "replacements": count if replace_all else 1,
        "patch_file": str(saved) if saved else None,
    }
    if resolved_note:
        result["resolved_path"] = resolved_note
    return result


def _first(arguments: dict[str, Any], *keys: str) -> Any:
    """First present, non-empty value among ``keys`` -- absorbs models that name
    the argument ``file_path``/``text``/``patch`` instead of the schema's name."""
    for key in keys:
        val = arguments.get(key)
        if val not in (None, ""):
            return val
    return None


def _first_str(arguments: dict[str, Any], *keys: str) -> str | None:
    """Like ``_first``, but keeps an explicit ``""`` -- ``new_string`` may
    legitimately be empty (a pure deletion), unlike every other aliased arg."""
    for key in keys:
        val = arguments.get(key)
        if isinstance(val, str):
            return val
    return None


def _to_host(ctx: ExecutionContext, text: str) -> str:
    """Map an in-container ``/repro/...`` path back to the host run dir for host-side I/O.

    The agent works against the short, remapped container paths (``sandbox.py``), but these
    tools do real host I/O, so ``/repro/workspace/x`` → ``<run_dir>/workspace/x`` (and the
    run dir is the host workspace's parent — see ``inputs.RunPaths``). Host-absolute and
    relative paths are returned unchanged; the containment check below still bounds them.
    """
    if ctx.workspace is None:
        return text
    if text == CONTAINER_RUN or text.startswith(CONTAINER_RUN + "/"):
        run_dir = Path(ctx.workspace).parent
        rel = text[len(CONTAINER_RUN):].lstrip("/")
        return str(run_dir / rel) if rel else str(run_dir)
    return text


def _resolve(ctx: ExecutionContext, raw: Any) -> dict[str, Any]:
    text = str(raw or "").strip()
    if not text:
        return {"ok": False, "error": "Missing path."}
    if "\\" in text or "\x00" in text:
        return {"ok": False, "error": f"Unsafe path: {text!r}"}
    text = _to_host(ctx, text)
    if ".." in Path(text).parts:
        return {"ok": False, "error": f"Path contains a '..' segment: {text}"}
    # ``write_file``/``edit_file`` only ever write durable source/evidence, so they
    # stay tight to workspace + evidence (a subset of the sandbox's rw binds; bulk
    # /tmp scratch is shell-driven, not a file-tool path).
    roots = [Path(root) for root in (ctx.workspace, ctx.evidence) if root is not None]
    if not roots:
        return {"ok": False, "error": "Episode has no workspace directory set."}
    base = Path(text) if text.startswith("/") else Path(ctx.workspace or roots[0]) / text
    try:
        resolved = base.resolve()
    except OSError as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    for root in roots:
        rootr = root.resolve()
        if resolved == rootr or rootr in resolved.parents:
            return {"ok": True, "path": resolved, "root": str(rootr)}
    return {"ok": False, "error": f"Path escapes the episode's writable roots: {text}"}


def mismatch_hint(content: str, old_string: str) -> str:
    """Recovery hint for an ``old_string`` that did not match ``content`` verbatim.

    Cheapest check first: if every line matches once leading/trailing whitespace
    is stripped, the drift is almost certainly just indentation -- say so
    directly. Otherwise anchor on ``old_string``'s first non-blank line and fuzzy-
    match it against the file's lines, quoting the real text ``_HINT_PAD`` lines
    on each side of the closest hit.
    """
    old_lines = old_string.splitlines()
    file_lines = content.splitlines()
    if not old_lines or not file_lines:
        return ""
    stripped_old = "\n".join(line.strip() for line in old_lines)
    stripped_file = "\n".join(line.strip() for line in file_lines)
    if stripped_old and stripped_old in stripped_file:
        return (
            "The text matches once each line's leading/trailing whitespace is "
            "stripped -- only indentation differs. Copy old_string with the "
            "file's exact whitespace."
        )
    anchor = next((line for line in old_lines if line.strip()), None)
    if anchor is None:
        return ""
    close = difflib.get_close_matches(anchor, file_lines, n=1, cutoff=0.5)
    if not close:
        return ""
    idx = file_lines.index(close[0])
    lo, hi = max(0, idx - _HINT_PAD), min(len(file_lines), idx + _HINT_PAD + 1)
    quoted = "\n".join(f"{i + 1:>6}\t{file_lines[i]}" for i in range(lo, hi))
    return f"Closest match is near line {idx + 1} -- the file's actual text there (copy it exactly):\n{quoted}"


def resolve_by_suffix(ctx: ExecutionContext, raw_path: Any) -> dict[str, Any] | None:
    """Suffix-search fallback for a relative path that did not resolve directly.

    Absolute paths join in one case: the prompt teaches the container layout, so a
    miss often arrives as ``/repro/workspace/src/x.py`` (or the host equivalent) --
    the same repo-relative confusion with the workspace prefix stapled on. Strip
    that prefix and search the remainder; any other absolute path that is missing
    is just missing, so return ``None``. Otherwise the result is a dict describing
    the outcome: ``{"ok": True, "path": Path}`` on exactly one hit, or
    ``{"ok": False, "result": <tool error dict>, "reason": <str>}`` on zero or
    multiple hits, so the caller can log the failure before returning it.
    """
    text = str(raw_path or "").strip()
    if not text or ctx.workspace is None:
        return None
    if text.startswith("/"):
        for prefix in (CONTAINER_RUN + "/workspace", str(Path(ctx.workspace))):
            if text.startswith(prefix + "/"):
                text = text[len(prefix) + 1 :]
                break
        else:
            return None
    # Skip ``.git``, any other dot-directory, and common dependency dirs so a
    # vendored copy or a venv's site-packages never shadows the real source file.
    # Capped at ``_MAX_SUFFIX_MATCHES`` hits -- this only runs on the failure path
    # of a direct resolve, but a runaway workspace tree should still bound the cost.
    workspace, rel = Path(ctx.workspace), Path(text)
    hits: list[Path] = []
    for hit in workspace.rglob(rel.name):
        if not hit.is_file():
            continue
        ancestors = hit.relative_to(workspace).parts[:-1]
        if any(part == ".git" or part.startswith(".") or part in _SKIP_DIRS for part in ancestors):
            continue
        if hit.parts[-len(rel.parts):] == rel.parts:
            hits.append(hit)
            if len(hits) >= _MAX_SUFFIX_MATCHES:
                break
    if len(hits) == 1:
        return {"ok": True, "path": hits[0]}
    if len(hits) > 1:
        shown = ", ".join(str(h) for h in hits[:5])
        return {
            "ok": False,
            "reason": "ambiguous",
            "result": {"ok": False, "error": f"Ambiguous path '{raw_path}' -- matches: {shown}."},
        }
    return {
        "ok": False,
        "reason": "not found",
        "result": {
            "ok": False,
            "error": (
                f"File not found: {raw_path}. Paths are relative to the workspace root -- "
                "the repo is likely cloned into a subdirectory; check with workspace_bash `ls`."
            ),
        },
    }


FILE_TOOLS = [
    function_tool(
        "write_file",
        "Write (or overwrite) a UTF-8 text file in the workspace or evidence dir. "
        "The reference copy is read-only; writes outside the bundle are rejected.",
        {
            "path": {"type": "string", "description": "File path (relative to workspace, or absolute within the bundle)."},
            "content": {"type": "string", "description": "Full file contents to write."},
        },
        ["path", "content"],
    ),
    function_tool(
        "edit_file",
        "Replace an exact string in an existing file. old_string must match the file's "
        "text exactly (copy it from a workspace_bash `grep -n` / `sed -n` read, "
        "indentation included) and must be unique in the file unless replace_all is set. "
        "Use write_file to create new files. Paths are relative to the workspace root; if "
        "the repo lives in a subdirectory, a bare repo-relative path is located "
        "automatically when it resolves to exactly one file.",
        {
            "path": {"type": "string", "description": "File path (relative to workspace, or absolute within the bundle)."},
            "old_string": {"type": "string", "description": "Exact text to replace; must match the file verbatim."},
            "new_string": {"type": "string", "description": "Replacement text (may be empty to delete old_string)."},
            "replace_all": {"type": "boolean", "description": "Replace every occurrence instead of requiring a unique match."},
        },
        ["path", "old_string", "new_string"],
    ),
]

FILE_TOOL_HANDLERS = {
    "write_file": write_file,
    "edit_file": edit_file,
}
