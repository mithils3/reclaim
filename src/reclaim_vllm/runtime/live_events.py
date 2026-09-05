"""Structured live-event seam for the auditor tool loop.

Mirrors ``reclaim_repro.live_log``'s sink hook so an opt-in telemetry sink can
stream each auditor round to the dashboard exactly like a reproduce run. Unlike
``live_log`` this writes no files (the auditor keeps its own ``--trace-output``);
it only forwards raw per-round data to a registered sink. Best-effort: a sink
failure must never break the loop.

The auditor loop is keyed by ``custom_id`` (the arXiv id it is grading), so every
event carries ``{custom_id, round_index}`` as its meta; the sink maps that to one
``audit_runs`` row + a stream of ``audit_events``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

# (kind, meta, payload) -> None. Registered once at startup by the audit sink.
_SINK: "Callable[[str, dict[str, Any], dict[str, Any]], None] | None" = None


def register_sink(fn: "Callable[[str, dict[str, Any], dict[str, Any]], None] | None") -> None:
    """Install (or clear with ``None``) the structured live sink."""
    global _SINK
    _SINK = fn


def _notify(kind: str, meta: dict[str, Any], payload: dict[str, Any]) -> None:
    if _SINK is None:
        return
    try:
        _SINK(kind, meta, payload)
    except Exception:  # noqa: BLE001 — best-effort, never break the loop
        return


def _meta(custom_id: str, round_index: int | None) -> dict[str, Any]:
    return {"custom_id": custom_id, "round_index": round_index}


def round_open(custom_id: str, round_index: int, message: dict[str, Any]) -> None:
    """A working round: the auditor's reasoning/text before its tools run."""
    _notify("round_open", _meta(custom_id, round_index), {"message": message})


def call_start(custom_id: str, round_index: int, call: dict[str, Any]) -> None:
    """A tool call (list_run_files / read_run_file / bash) about to execute."""
    _notify("call_start", _meta(custom_id, round_index), {"call": call})


def call_result(custom_id: str, round_index: int, result: dict[str, Any]) -> None:
    """That tool's result the instant it returns."""
    _notify("call_result", _meta(custom_id, round_index), {"result": result})


def final(custom_id: str, round_index: int, message: dict[str, Any], exit_reason: str,
          verdict: dict[str, Any]) -> None:
    """The terminal submission + the finalized verdict row (score/verdict/flags)."""
    _notify("final", _meta(custom_id, round_index),
            {"message": message, "exit_reason": exit_reason, "verdict": verdict})
