"""Repro toolset — the agent's hands, assembled and dispatched.

The reproduction agent acts on its episode through these tools:

* ``workspace_bash`` -- cwd-confined shell (clone, venv, installs, run CPU work,
  **and all file reading** via ``grep``/``sed``/``cat`` -- there is no read_file
  tool: targeted shell reads are far cheaper than dumping whole files to context),
* ``write_file`` / ``edit_file`` -- path-confined file writes/exact-match edits,
* ``update_plan`` -- a short, steerable checklist (Codex-style): harness state the
  model resends each call, pinned onto the ``ExecutionContext`` and mirrored to
  ``evidence/plan.md`` for the auditor,
* ``fetch_url`` -- read-only fetch of a public http(s) URL (docs, wheel index, raw
  repo files); there is no general web search, so it fetches URLs the agent knows,
* ``list_partitions`` -- read-only ``sinfo`` of the cluster's partitions (node pools)
  + the known-cluster defaults, so the agent can pick a ``run_gpu`` ``partition``
  instead of the profile's hardcoded default,
* ``run_gpu`` -- the one metered path to a GPU: a ``salloc`` allocation held across
  calls (``srun --jobid`` per step, released on ``release=true``/teardown), billed by
  wall clock (``gpu_session``); ``partition`` (optional) selects the pool to hold.

``build_repro_tools`` returns the schema list advertised to the model (wired onto
``args.tools`` in ``cli_args``); ``REPRO_TOOL_HANDLERS`` maps each name to its
``(arguments, ctx)`` handler. ``execute_repro_tool_call`` is the single entry the
loop dispatches through (``dispatch.append_tool_results``): it parses arguments,
routes to the handler against this episode's ``ExecutionContext``, retries a
transient failure once, and trims the result to the shared tool budget -- the same
posture as the auditor's ``web_tools.execute_tool_call``, minus the read-only
``Paper``.
"""

from __future__ import annotations

from typing import Any

from reclaim_vllm.config.config import TOOL_MAX_CHARS, TOOL_RESULT_MAX_CHARS, function_tool
from reclaim_vllm.tools.result_limits import is_transient_error, truncate_tool_result
from reclaim_vllm.tools.web_fetch import fetch_url_tool, parse_tool_arguments

from reclaim_repro.context import ExecutionContext
from reclaim_repro.tools.files import FILE_TOOL_HANDLERS, FILE_TOOLS
from reclaim_repro.tools.partitions import LIST_PARTITIONS_HANDLERS, LIST_PARTITIONS_TOOLS
from reclaim_repro.tools.plan import UPDATE_PLAN_HANDLERS, UPDATE_PLAN_TOOLS
from reclaim_repro.tools.run_gpu import RUN_GPU_HANDLERS, run_gpu_tool
from reclaim_repro.tools.workspace_bash import WORKSPACE_BASH_HANDLERS, WORKSPACE_BASH_TOOL

_FETCH_TOOL = function_tool(
    "fetch_url",
    "Fetch a direct public http(s) URL and return its status, final URL, and "
    "text (HTML is reduced to text). There is no general web-search tool — you "
    "fetch URLs you already know or can construct (e.g. the PyTorch install page "
    "or wheel index to find the right aarch64/CUDA torch build, a raw README or "
    "requirements file, a docs/release page).",
    {
        "url": {"type": "string", "description": "HTTP or HTTPS URL."},
        "max_chars": {"type": "integer", "default": TOOL_MAX_CHARS},
    },
    ["url"],
)


def build_repro_tools(gpus_per_node: int) -> list[dict]:
    """The tool schemas advertised to the model, with run_gpu's GPU cap = node size."""
    return [
        WORKSPACE_BASH_TOOL,
        *FILE_TOOLS,
        *UPDATE_PLAN_TOOLS,
        _FETCH_TOOL,
        *LIST_PARTITIONS_TOOLS,
        run_gpu_tool(gpus_per_node),
    ]


REPRO_TOOL_HANDLERS: dict[str, Any] = {
    **WORKSPACE_BASH_HANDLERS,
    **FILE_TOOL_HANDLERS,
    **UPDATE_PLAN_HANDLERS,
    "fetch_url": lambda arguments, _ctx: fetch_url_tool(arguments),
    **LIST_PARTITIONS_HANDLERS,
    **RUN_GPU_HANDLERS,
}


def execute_repro_tool_call(call: dict[str, Any], ctx: ExecutionContext) -> dict[str, Any]:
    """Route one tool call through the episode's ``ExecutionContext`` and trim it."""
    result = _run_tool_call(call, ctx)
    if is_transient_error(result):
        result = _run_tool_call(call, ctx)
        result["retried"] = True
    return truncate_tool_result(result, TOOL_RESULT_MAX_CHARS)


def _run_tool_call(call: dict[str, Any], ctx: ExecutionContext) -> dict[str, Any]:
    function = call.get("function") or {}
    name = str(function.get("name") or "")
    handler = REPRO_TOOL_HANDLERS.get(name)
    if handler is None:
        available = ", ".join(REPRO_TOOL_HANDLERS)
        return {"ok": False, "error": f"Unknown tool: {name}. Available tools: {available}"}
    try:
        arguments = parse_tool_arguments(function.get("arguments", {}))
    except Exception as exc:
        return {"ok": False, "tool": name, "error": f"{type(exc).__name__}: {exc}"}
    try:
        return handler(arguments, ctx)
    except Exception as exc:
        return {"ok": False, "tool": name, "error": f"{type(exc).__name__}: {exc}"}


__all__ = ["REPRO_TOOL_HANDLERS", "build_repro_tools", "execute_repro_tool_call"]
