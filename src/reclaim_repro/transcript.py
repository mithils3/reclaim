"""Conversation shaping and incremental output writing for the reproduce loop.

Split out of ``loop.py`` to keep the driver focused. These are forked analogs of
the shared final-message and output helpers, minus the auditor-specific
extracted/audit rows: repro writes the raw response JSONL (and an optional round
trace). The agent's terminal output is its ``report.json``, an account
of what it ran and measured; the *verdict* is the auditor's, not written here.
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from typing import Any

from reclaim_vllm.runtime.trace_io import append_trace_row, assistant_message
from reclaim_vllm.vllm.io import append_jsonl_row, truncate_output_file

# Repro-specific loop text; the shared strings stay in reclaim_vllm.
CONTEXT_BUDGET_NOTE = (
    "The conversation hit its context budget, so the tool phase ended early. "
    "Finalize from the evidence already written to the run directory rather "
    "than re-running tools. "
)

# Serializes incremental output writes across the tool-pool threads.
OUTPUT_WRITE_LOCK = threading.Lock()

# Exit reasons that prepend the budget note to the final tools-off turn.
EARLY_EXIT_REASONS = ("context_budget", "budget_exhausted")

# Max times a single episode may be nudged after a length-truncated turn before the
# loop gives up and force-finals it. A truncated turn carries no tool call, so left
# unbounded a model that keeps overrunning the output cap would loop forever; two
# retries let it recover the plan and act without stranding the round budget.
LENGTH_RETRY_LIMIT = 2

# How much of a truncated turn's reasoning to keep when re-issuing. The tail was cut
# mid-stream by the output-token cap, so the useful plan is at the head; the rest is
# dropped so the trimmed turn does not itself re-blow the budget on the next request.
TRUNCATED_REASONING_KEEP_CHARS = 2000
TRUNCATED_REASONING_MARKER = " ...[cut off at the output-token limit]"

# Absolute unix epoch (seconds) at which SLURM kills the sweep job hosting this
# run's vLLM server. Set by the sweep sbatch scripts (scripts/reproduce/*); unset
# for local runs and OpenRouter-backed drivers that carry no SLURM allocation.
WALL_DEADLINE_ENV = "REPRO_WALL_DEADLINE_EPOCH"


def prepare_incremental_outputs(args: argparse.Namespace) -> None:
    truncate_output_file(args.output)
    if args.save_round_jsonl:
        truncate_output_file(args.trace_output)


def append_completed_outputs(
    custom_id: str,
    row: dict[str, Any],
    messages: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    with OUTPUT_WRITE_LOCK:
        append_jsonl_row(args.output, row, truncate=False)
        if args.save_round_jsonl:
            append_trace_row(args.trace_output, custom_id, messages, row)


def noop() -> None:
    return None


def trim_truncated_reasoning(message: dict[str, Any]) -> dict[str, Any]:
    """Shallow copy of ``message`` with over-long ``reasoning``/``reasoning_content`` cut.

    A turn that hit the output-token cap carries a bloated, mid-stream reasoning
    trace. We keep the head (the plan) and mark the cut so the re-issued turn stays
    cheap. The input dict is never mutated; short fields pass through untouched.
    """
    trimmed = dict(message)
    for key in ("reasoning", "reasoning_content"):
        value = trimmed.get(key)
        if isinstance(value, str) and len(value) > TRUNCATED_REASONING_KEEP_CHARS:
            trimmed[key] = value[:TRUNCATED_REASONING_KEEP_CHARS] + TRUNCATED_REASONING_MARKER
    return trimmed


def length_nudge_message() -> dict[str, Any]:
    """User turn that follows a length-truncated turn: act now, keep reasoning short."""
    return {
        "role": "user",
        "content": (
            "Your previous turn hit the output-token limit before emitting any tool "
            "call, so it was cut off mid-stream and trimmed above. Do not re-plan from "
            "scratch. Take the next concrete action with a tool call now, and keep "
            "reasoning brief."
        ),
    }


def final_user_message(budget_note: bool, final_message: str) -> dict[str, Any]:
    content = CONTEXT_BUDGET_NOTE + final_message if budget_note else final_message
    return {"role": "user", "content": content}


def sweep_wall_note(now: float | None = None) -> str | None:
    """Countdown to the outer sweep job's SLURM wall, or ``None`` if not applicable.

    Reads ``WALL_DEADLINE_ENV``, an absolute unix epoch set by the sweep sbatch
    scripts. Missing, empty, or unparseable values return ``None`` rather than
    raising, since most drivers (local runs, OpenRouter-backed runs) never set it.
    """
    raw = os.environ.get(WALL_DEADLINE_ENV)
    if not raw:
        return None
    try:
        deadline = float(raw)
    except ValueError:
        return None
    remaining = deadline - (now if now is not None else time.time())
    if remaining <= 0:
        return "sweep wall EXPIRED — the job hosting this run is being killed; finalize immediately"
    h = int(remaining // 3600)
    m = int((remaining % 3600) // 60)
    return f"sweep wall ~{h}h{m:02d}m left"


def round_status_message(round_index: int, max_rounds: int) -> dict[str, Any]:
    """Ephemeral per-request line telling the model which tool round it is on.

    Built fresh for each request and never stored in the conversation, so stale
    counters don't accumulate. Surfaces the turn budget — capped independently of the
    H100-hour budget — which the model otherwise has no way to see. When the sweep
    sbatch scripts have set ``WALL_DEADLINE_ENV``, also appends a countdown to the
    outer SLURM job's wall clock; that countdown is recomputed fresh every call, so
    it never goes stale, and is omitted (byte-identical to the old line) when unset.
    """
    used = round_index + 1
    left = max(0, max_rounds - used)
    content = f"Tool round {used}/{max_rounds} · {left} left"
    note = sweep_wall_note()
    if note is not None:
        content = f"{content} · {note}"
    return {"role": "user", "content": content}


def conversation_for_round(
    messages: list[dict],
    include_tools: bool,
    *,
    budget_note: bool = False,
    final_message: str = "",
) -> list[dict]:
    if include_tools:
        return messages
    return [*messages, final_user_message(budget_note, final_message)]


def append_final_message(
    messages: list[dict[str, Any]],
    message: dict[str, Any],
    tool_calls: list[dict[str, Any]],
    include_tools: bool,
    *,
    budget_note: bool = False,
    final_message: str = "",
) -> None:
    if not include_tools:
        messages.append(final_user_message(budget_note, final_message))
    messages.append(assistant_message(message, tool_calls))
