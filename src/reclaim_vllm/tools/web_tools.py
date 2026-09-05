"""Auditor tool dispatch: routes model tool calls to the run-dir tool handlers.

Every audit tool call enters through ``execute_tool_call(call, paper)``: parse
the arguments, bind the handler to the paper's agent run directory, retry one
transient failure, and trim the result to the shared tool budget.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from reclaim_vllm.config.config import TOOL_RESULT_MAX_CHARS
from reclaim_vllm.papers.papers import Paper
from .result_limits import is_transient_error, truncate_tool_result
from .run_dir_tools import AUDIT_TOOL_HANDLERS
from .web_fetch import parse_tool_arguments


def execute_tool_call(call: dict[str, Any], paper: Paper | None = None) -> dict[str, Any]:
    result = run_tool_call(call, paper)
    if is_transient_error(result):
        result = run_tool_call(call, paper)
        result["retried"] = True
    return truncate_tool_result(result, TOOL_RESULT_MAX_CHARS)


def run_tool_call(call: dict[str, Any], paper: Paper | None) -> dict[str, Any]:
    function = call.get("function") or {}
    name = function.get("name", "")
    try:
        arguments = parse_tool_arguments(function.get("arguments", {}))
        audit_handler = AUDIT_TOOL_HANDLERS.get(name)
        if audit_handler is None:
            available = ", ".join(AUDIT_TOOL_HANDLERS)
            return {
                "ok": False,
                "error": f"Unknown tool: {name}. Available tools: {available}",
            }
        run_dir = str(getattr(paper, "run_dir", "") or "") if paper else ""
        if not run_dir:
            return {
                "ok": False,
                "tool": name,
                "error": "No agent run directory is bound for this paper (set --runs-dir).",
            }
        return audit_handler(arguments, Path(run_dir))
    except Exception as exc:
        return {"ok": False, "tool": name, "error": f"{type(exc).__name__}: {exc}"}
