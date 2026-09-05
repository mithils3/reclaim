"""Run-directory tools for the LLM reproduction auditor.

The auditor is handed ONE agent reproduction run directory -- a collection of
``*.log`` files, output artifacts, and any code the agent wrote -- and explores
it to trace how every graded number was actually produced. These tools are
scoped to that one directory:

  - ``list_run_files``  -- list files and directories in the run dir,
  - ``read_run_file``   -- read one text file (a log, an output JSON, a script),
  - ``write_run_file``  -- write a NEW text file (e.g. a re-scoring script) into
                           the run dir to run with ``bash`` and cite; never
                           overwrites an existing file,
  - ``bash``            -- run a shell command with the run dir as the cwd (grep
                           logs, inspect outputs, run a written script).

There is deliberately no separate Python interpreter: re-scoring goes through a
``write_run_file`` script + ``bash``, keeping every computation on disk and
citable. ``run_dir_manifest`` builds the file listing injected into the prompt.
"""

from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path
from typing import Any

from reclaim_vllm.config.config import (
    BASH_TIMEOUT,
    RUN_FILE_DEFAULT_CHARS,
    RUN_FILE_MAX_CHARS,
    RUN_FILE_WRITE_MAX_CHARS,
    RUN_MANIFEST_MAX_ENTRIES,
    bounded as _bounded,
    function_tool,
)

SKIP_DIRS = {".git", "__pycache__", ".venv", "node_modules"}


def list_run_files(arguments: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    base = _resolve_within(run_dir, arguments.get("path") or ".")
    if not base["ok"]:
        return base
    target: Path = base["path"]
    if not target.exists():
        return {"ok": False, "error": f"Path does not exist in run dir: {arguments.get('path')}"}
    entries = _walk(target, run_dir, recursive=bool(arguments.get("recursive")))
    return {
        "ok": True,
        "run_dir": str(run_dir),
        "path": _rel(target, run_dir),
        "count": len(entries),
        "entries": entries[:RUN_MANIFEST_MAX_ENTRIES],
        "truncated": len(entries) > RUN_MANIFEST_MAX_ENTRIES,
    }


def read_run_file(arguments: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    resolved = _resolve_within(run_dir, arguments.get("path"))
    if not resolved["ok"]:
        return resolved
    target: Path = resolved["path"]
    if not target.is_file():
        return {"ok": False, "error": f"Not a file in run dir: {arguments.get('path')}"}
    max_chars = _bounded(arguments.get("max_chars"), RUN_FILE_DEFAULT_CHARS, RUN_FILE_MAX_CHARS)
    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "ok": True,
        "path": _rel(target, run_dir),
        "size": _size(target),
        "text": text[:max_chars],
        "truncated": len(text) > max_chars,
    }


def run_bash(arguments: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    command = str(arguments.get("command") or "").strip()
    if not command:
        return {"ok": False, "error": "Missing bash command"}
    if not run_dir.exists():
        return {"ok": False, "error": f"Run dir does not exist: {run_dir}"}
    timeout = _bounded(arguments.get("timeout"), BASH_TIMEOUT, BASH_TIMEOUT)
    try:
        # Own process group, so the timeout can kill the whole tree. `subprocess.run`
        # kills only the direct child: a command that backgrounds or spawns anything
        # (`python eval.py &`, a server, a pool) leaves grandchildren holding the
        # inherited stdout pipe, and the reap after the timeout blocks on a read that
        # never ends -- the timeout expires and the tool hangs anyway, stalling the
        # whole audit on one shell command.
        proc = subprocess.Popen(
            ["bash", "-lc", command],
            cwd=str(run_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
    except OSError as exc:
        return {"ok": False, "command": command, "error": f"{type(exc).__name__}: {exc}"}
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        stdout, stderr = _kill_group(proc)
        return {
            "ok": False,
            "command": command,
            "error": f"bash timed out after {timeout}s (process group killed)",
            "stdout": stdout,
            "stderr": stderr,
        }
    return {
        "ok": proc.returncode == 0,
        "command": command,
        "returncode": proc.returncode,
        "stdout": stdout,
        "stderr": stderr,
    }


def _kill_group(proc: "subprocess.Popen[str]") -> tuple[str, str]:
    """SIGKILL the timed-out command's whole process group; return what it printed.

    The second wait is itself bounded: if something survives the kill (an
    unkillable D-state child on a network filesystem), we abandon the output
    rather than block the auditor forever.
    """
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (OSError, ProcessLookupError):
        proc.kill()
    try:
        return proc.communicate(timeout=5)
    except (subprocess.TimeoutExpired, ValueError):
        return "", ""


def write_run_file(arguments: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    resolved = _resolve_within(run_dir, arguments.get("path"))
    if not resolved["ok"]:
        return resolved
    target: Path = resolved["path"]
    if target == run_dir or target.is_dir():
        return {"ok": False, "error": f"Not a writable file path: {arguments.get('path')}"}
    if target.exists():
        # The run dir is the evidence under audit; never clobber an agent artifact.
        rel = _rel(target, run_dir)
        return {"ok": False, "error": f"Refusing to overwrite {rel}; write to a new path."}
    content = arguments.get("content")
    if not isinstance(content, str):
        return {"ok": False, "error": "Missing 'content' string to write."}
    if len(content) > RUN_FILE_WRITE_MAX_CHARS:
        return {"ok": False, "error": f"Content exceeds {RUN_FILE_WRITE_MAX_CHARS} chars."}
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    except OSError as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "ok": True,
        "path": _rel(target, run_dir),
        "bytes_written": len(content.encode("utf-8")),
    }


AUDIT_TOOL_HANDLERS = {
    "list_run_files": list_run_files,
    "read_run_file": read_run_file,
    "write_run_file": write_run_file,
    "bash": run_bash,
}


AUDIT_TOOLS = [
    function_tool(
        "list_run_files",
        "List files and directories in the agent's reproduction run directory. "
        "Use it to discover the *.log, output, and code files the agent produced.",
        {
            "path": {
                "type": "string",
                "description": "Subdirectory within the run dir; omit for the run-dir root.",
            },
            "recursive": {"type": "boolean", "default": False},
        },
        [],
    ),
    function_tool(
        "read_run_file",
        "Read a text file (a log, an output JSON, or a script the agent wrote) "
        "from the run directory by relative path.",
        {
            "path": {
                "type": "string",
                "description": "File path relative to the run directory.",
            },
            "max_chars": {
                "type": "integer",
                "default": RUN_FILE_DEFAULT_CHARS,
                "minimum": 1,
                "maximum": RUN_FILE_MAX_CHARS,
            },
        },
        ["path"],
    ),
    function_tool(
        "bash",
        "Run a bash command with the agent's run directory as the working "
        "directory. Use it to grep logs, inspect outputs, or run `python3 -c "
        "...`. Returns stdout, stderr, and the exit code.",
        {
            "command": {
                "type": "string",
                "description": "Bash command to run inside the run directory.",
            },
            "timeout": {
                "type": "integer",
                "default": BASH_TIMEOUT,
                "minimum": 1,
                "maximum": BASH_TIMEOUT,
            },
        },
        ["command"],
    ),
    function_tool(
        "write_run_file",
        "Write a NEW text file (e.g. a re-scoring Python script) into the run "
        "directory so you can run it with bash and cite it as evidence. Use this "
        "for multi-line scripts instead of fighting `python3 -c` quoting. Will "
        "not overwrite an existing file -- pick a new path.",
        {
            "path": {
                "type": "string",
                "description": "New file path relative to the run directory; must not already exist.",
            },
            "content": {
                "type": "string",
                "description": "Text content to write (e.g. Python source).",
            },
        },
        ["path", "content"],
    ),
]


def run_dir_manifest(run_dir: Path) -> str:
    """A text listing of the run directory, injected into the audit prompt."""
    if not run_dir.exists():
        return (
            f"(No run directory found at {run_dir}. No agent reproduction run is "
            "available for this paper, so the only defensible verdict is "
            "`unverifiable` with score 0.)"
        )
    files = [e for e in _walk(run_dir, run_dir, recursive=True) if e["type"] == "file"]
    if not files:
        return f"(Run directory {run_dir} is empty; verdict `unverifiable`, score 0.)"
    shown = files[:RUN_MANIFEST_MAX_ENTRIES]
    lines = [f"AGENT RUN DIRECTORY: {run_dir}", f"{len(files)} file(s):"]
    lines += [f"  {entry['path']}  ({entry['size']} bytes)" for entry in shown]
    if len(files) > len(shown):
        lines.append(f"  ... and {len(files) - len(shown)} more")
    lines.append(
        "\nUse list_run_files, read_run_file, and bash (cwd = this run directory) "
        "to open these files and trace how every reported number was produced. "
        "To re-score an artifact, write_run_file a script and run it with bash."
    )
    return "\n".join(lines)


def _resolve_within(run_dir: Path, rel: Any) -> dict[str, Any]:
    raw = str(rel or "").strip()
    if raw in ("", "."):
        return {"ok": True, "path": run_dir}
    if raw.startswith("/") or "\\" in raw or ".." in raw.split("/"):
        return {"ok": False, "error": f"Unsafe path: {raw}"}
    target = run_dir / raw
    try:
        root = run_dir.resolve()
        resolved = target.resolve()
    except OSError as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    if resolved != root and root not in resolved.parents:
        return {"ok": False, "error": f"Path escapes the run directory: {raw}"}
    return {"ok": True, "path": target}


def _walk(target: Path, run_dir: Path, *, recursive: bool) -> list[dict[str, Any]]:
    """List ``target``, recursively or not, with SKIP_DIRS subtrees left out.

    A recursive walk prunes SKIP_DIRS as it descends instead of listing everything
    and filtering afterwards: an unpruned run bundle's ``.venv`` is 50-100k files,
    and this runs once per audit prompt plus on every recursive ``list_run_files``.
    Entries are still sorted by full path, and every entry is still collected, so
    the manifest listing and the caller's count/truncation are unchanged.
    """
    if any(part in SKIP_DIRS for part in _rel_parts(target, run_dir)):
        return []
    if recursive:
        found = _descendants(target)
    else:
        try:
            found = [(child, child.is_dir()) for child in target.iterdir()]
        except OSError:
            return []
        found = [
            item for item in found
            if not any(part in SKIP_DIRS for part in _rel_parts(item[0], run_dir))
        ]
    return [
        {
            "path": _rel(child, run_dir),
            "type": "dir" if is_dir else "file",
            "size": 0 if is_dir else _size(child),
        }
        for child, is_dir in sorted(found, key=lambda item: str(item[0]))
    ]


def _descendants(target: Path) -> list[tuple[Path, bool]]:
    """Every descendant of ``target`` as (path, is_dir), SKIP_DIRS pruned in place."""
    found: list[tuple[Path, bool]] = []
    for dirpath, dirnames, filenames in os.walk(target, topdown=True):
        dirnames[:] = [name for name in dirnames if name not in SKIP_DIRS]
        base = Path(dirpath)
        found.extend((base / name, True) for name in dirnames)
        found.extend((base / name, False) for name in filenames)
    return found


def _rel_parts(path: Path, run_dir: Path) -> tuple[str, ...]:
    try:
        return path.relative_to(run_dir).parts
    except ValueError:
        return ()


def _rel(path: Path, run_dir: Path) -> str:
    try:
        return str(path.relative_to(run_dir))
    except ValueError:
        return str(path)


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


