"""Human-readable live transcript for the reproduce loop — a ``tail -f`` view.

The durable evidence sinks are for the auditor: ``commands.log`` is one terse line
per shell command, ``trajectory.jsonl`` is machine-readable and only written for
``run_gpu`` steps, and the model's natural-language reasoning never lands there at
all (it lives in the in-memory conversation and is flushed to ``reproduce.jsonl``
only when the episode *finishes*). So while a run is in flight there is nowhere to
watch *what the agent is thinking and doing*.

This module fills that gap. It writes two append-only files, formatted like an
interactive agent session, that interleave per round: the model's reasoning, its
visible text, each tool call, and each tool result.

- ``<run_dir>/agent.log`` — the skimmable view: each reasoning / text / stdout /
  stderr block is head-truncated to ``HEAD_LINES``. Point ``tail -f`` at it.
- ``<run_dir>/agent.full.log`` — the same transcript with NOTHING truncated by the
  logger: every line of reasoning, visible text, and tool output the model produced
  or received is written verbatim. Use it when you need the complete record (e.g.
  post-hoc analysis). (Any cap a *tool* applies to its own result is upstream of
  this file and still applies — the logger no longer adds one.)

It is best-effort and append-only: a logging failure must never break the loop, so
every writer swallows its own exceptions. Each episode writes its own files (keyed
by run_dir), and rounds within an episode are sequential, so no locking is needed.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from reclaim_repro.context import ExecutionContext

# Max stdout/stderr (or reasoning/text) lines shown per block in ``agent.log``; the
# untruncated copy goes to ``agent.full.log``. Keeps the live log skimmable.
HEAD_LINES = 24
RULE = "─" * 72

# Optional structured sink (e.g. the telemetry uploader). Registered once at
# startup; fed the *raw* per-round data alongside the file writes so a remote
# dashboard sees the same events. Best-effort: a sink failure must never break
# the loop, exactly like the file writers below.
_SINK: "Callable[[str, ExecutionContext, dict[str, Any]], None] | None" = None


def register_sink(fn: "Callable[[str, ExecutionContext, dict[str, Any]], None] | None") -> None:
    """Install (or clear with ``None``) the structured live sink."""
    global _SINK
    _SINK = fn


def _notify(kind: str, ctx: ExecutionContext, payload: dict[str, Any]) -> None:
    if _SINK is None:
        return
    try:
        _SINK(kind, ctx, payload)
    except Exception:  # noqa: BLE001 — best-effort, never break the loop
        return


def _log_path(ctx: ExecutionContext) -> Path | None:
    """``<run_dir>/agent.log`` — sibling of the evidence dir — or None if unknown."""
    if ctx.evidence is None:
        return None
    return Path(ctx.evidence).parent / "agent.log"


def _full_log_path(ctx: ExecutionContext) -> Path | None:
    """``<run_dir>/agent.full.log`` — the untruncated companion, or None if unknown."""
    if ctx.evidence is None:
        return None
    return Path(ctx.evidence).parent / "agent.full.log"


def _stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _head(text: str, prefix: str, *, full: bool = False) -> list[str]:
    body = (text or "").rstrip()
    if not body:
        return []
    lines = body.splitlines()
    if full:
        return [prefix + ln for ln in lines]
    shown = [prefix + ln for ln in lines[:HEAD_LINES]]
    extra = len(lines) - HEAD_LINES
    if extra > 0:
        shown.append(f"{prefix}… (+{extra} more line{'s' if extra != 1 else ''})")
    return shown


def call_arguments(call: dict[str, Any]) -> dict[str, Any]:
    """Decode a tool call's ``function.arguments`` (JSON string or dict) to a dict.

    Public so row builders (``telemetry_rows``) share one parse of the call payload
    instead of reaching into a private helper. A non-JSON string degrades to
    ``{"_raw": <string>}``; anything non-dict-like becomes ``{}``.
    """
    fn = call.get("function") or {}
    args = fn.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (ValueError, TypeError):
            return {"_raw": args}
    return args if isinstance(args, dict) else {}


def _call_summary(call: dict[str, Any], *, full: bool = False) -> tuple[str, str]:
    """(tool name, the single most useful argument rendered for a human)."""
    name = str((call.get("function") or {}).get("name") or "?")
    args = call_arguments(call)
    if "command" in args:
        detail = "$ " + str(args["command"])
    elif "diff" in args:
        path = args.get("path")
        detail = f"(apply diff{' to ' + str(path) if path else ''})"
    elif "path" in args:
        detail = str(args["path"])
    elif args:
        dumped = json.dumps(args)
        detail = dumped if full else dumped[:300]
    else:
        detail = ""
    return name, detail


def _result_lines(result: dict[str, Any], *, full: bool = False) -> list[str]:
    ok = result.get("ok")
    mark = "✓" if ok else "✗"
    parts = [f"   {mark} ok={ok}"]
    if result.get("returncode") is not None:
        parts.append(f"rc={result['returncode']}")
    if result.get("duration_s") is not None:
        parts.append(f"{result['duration_s']}s")
    if result.get("run_seconds") is not None:
        parts.append(f"{result['run_seconds']}s")
    if result.get("retried"):
        parts.append("(retried)")
    lines = [" ".join(parts)]
    if result.get("error"):
        lines.append("   ! " + str(result["error"]).splitlines()[0])
    for key in ("path", "bytes_written", "cost_h100_hours", "remaining_h100_hours"):
        if key in result:
            lines.append(f"   · {key}={result[key]}")
    lines += _head(str(result.get("stdout") or ""), "   │ ", full=full)
    lines += _head(str(result.get("stderr") or ""), "   ┊ ", full=full)
    return lines


def _message_lines(message: dict[str, Any], *, full: bool = False) -> list[str]:
    lines: list[str] = []
    reasoning = message.get("reasoning") or message.get("reasoning_content")
    if reasoning:
        lines.append(" 💭 reasoning")
        lines += _head(str(reasoning), "    ", full=full)
    content = message.get("content")
    if content and str(content).strip():
        lines.append(" 🗒  assistant")
        lines += _head(str(content), "    ", full=full)
    return lines


def _write(path: Path | None, lines: list[str]) -> None:
    """Append ``lines`` to ``path`` and flush immediately so ``tail -f`` sees them.

    Each call opens/appends/closes, so writes stream out incrementally within a
    round rather than batching at round end — the agent's plan, the command it is
    about to run, and that command's result each land the moment they happen.
    """
    if path is None or not lines:
        return
    try:
        with path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
            handle.flush()
    except OSError:
        return


def _emit(ctx: ExecutionContext, build: Callable[[bool], list[str]]) -> None:
    """Render a block twice: head-truncated to agent.log, verbatim to agent.full.log."""
    _write(_log_path(ctx), build(False))
    _write(_full_log_path(ctx), build(True))


def log_round_open(
    ctx: ExecutionContext,
    message: dict[str, Any],
    *,
    round_index: int | None = None,
    finish_reason: str | None = None,
) -> None:
    """Open a round: flush the model's reasoning/text before any tool runs."""
    rnd = f"round {round_index}" if round_index is not None else "round"
    header = f" {rnd} · {getattr(ctx, 'arxiv_id', '?')} · {_stamp()}"
    if finish_reason and finish_reason != "stop":
        header += f" · finish={finish_reason}"
    _emit(ctx, lambda full: [RULE, header, RULE, *_message_lines(message, full=full)])
    _notify("round_open", ctx, {
        "round_index": round_index, "message": message, "finish_reason": finish_reason,
    })


def log_call_start(ctx: ExecutionContext, call: dict[str, Any]) -> None:
    """Flush the tool call + its command *before* it executes (streams long steps)."""

    def build(full: bool) -> list[str]:
        name, detail = _call_summary(call, full=full)
        lines = ["", f" ⏺ {name}"]
        if detail:
            lines.append("   " + detail)
        return lines

    _emit(ctx, build)
    _notify("call_start", ctx, {"call": call})


def log_call_result(ctx: ExecutionContext, result: dict[str, Any]) -> None:
    """Flush a tool result the instant it returns."""
    _emit(ctx, lambda full: _result_lines(result, full=full))
    _notify("call_result", ctx, {"result": result})


def log_final(
    ctx: ExecutionContext,
    message: dict[str, Any],
    *,
    round_index: int | None = None,
    exit_reason: str = "",
    finish_reason: str | None = None,
) -> None:
    """Append the episode's terminal submission (final assistant turn)."""
    rnd = f" (round {round_index})" if round_index is not None else ""
    header = (
        f" ✅ FINAL{rnd} · {getattr(ctx, 'arxiv_id', '?')} · exit={exit_reason} · {_stamp()}"
    )
    if finish_reason and finish_reason != "stop":
        header += f" · finish={finish_reason}"
    _emit(ctx, lambda full: [RULE, header, RULE, *_message_lines(message, full=full), ""])
    _notify("final", ctx, {
        "round_index": round_index, "message": message,
        "exit_reason": exit_reason, "finish_reason": finish_reason,
    })


def log_usage(
    ctx: ExecutionContext,
    usage: dict[str, Any] | None,
    *,
    round_index: int | None = None,
    kind: str = "round",
) -> None:
    """Forward one model response's token ``usage`` to the structured sink.

    No file write — the human transcript stays clean; the usage is summed onto the
    telemetry run row and the uploaded ``stats.json``. Best-effort and a no-op when
    no sink is registered or ``usage`` is empty, exactly like the writers above.
    """
    if not usage:
        return
    _notify("usage", ctx, {"round_index": round_index, "kind": kind, "usage": usage})
