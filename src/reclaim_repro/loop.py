"""Forked tool loop for the reproduction agent.

A *diverged* fork of ``reclaim_vllm.runtime.tool_loop.run_tool_loop``: it began
by mirroring that loop's structure (two thread pools, ``wait(FIRST_COMPLETED)``,
``handle_request_done``) but the two have since drifted and are no longer kept in
sync — this loop swaps the body for the reproduction seams:

* tool calls dispatch through a per-episode ``ExecutionContext`` via
  ``append_tool_results`` (workspace + budget + allocation + evidence)
  instead of ``execute_tool_call(call, paper=paper)``;
* the post-round seam (the analog of ``handle_request_done`` in ``tool_loop.py``)
  calls ``apply_guardrails`` — the compute-budget force-final plus the
  ``elide-compact`` context tier that keeps the loop going.

Conversation shaping and output writing live in ``transcript.py``; the tool seam
(``append_tool_results``), the between-round budget + context guardrails
(``apply_guardrails``), and the end-of-episode seam (``finalize_episode`` — release
GPU, emit ``report.json``, flip the run terminal) live in this module. The loop emits
the agent's ``report.json`` but writes **no verdict**; the auditor grades
the run bundle.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from reclaim_vllm.config.config import REQUEST_TIMEOUT
from reclaim_vllm.runtime.run_health import loop_telemetry
from reclaim_vllm.runtime.trace_io import assistant_message
from reclaim_vllm.vllm.client import post_chat_completion_row, response_row
from reclaim_vllm.vllm.io import (
    build_chat_completion_request,
    initial_messages,
    normalize_tool_calls,
    response_finish_reason,
    response_message,
    tool_result_message,
)

from reclaim_repro import cleanup, compact, gpu_session, live_log, report
from reclaim_repro import evidence as evidence_mod
from reclaim_repro.context import ExecutionContext
from reclaim_repro.tools import execute_repro_tool_call
from reclaim_repro.transcript import (
    EARLY_EXIT_REASONS,
    LENGTH_RETRY_LIMIT,
    append_completed_outputs,
    append_final_message,
    conversation_for_round,
    length_nudge_message,
    noop,
    prepare_incremental_outputs,
    round_status_message,
    trim_truncated_reasoning,
)


def run_reproduce_loop(
    args: argparse.Namespace,
    contexts: list[ExecutionContext],
    prompts: list[str],
    server_url: str,
    model_id: str | None = None,
) -> None:
    request_model = model_id or args.model
    conversations = {
        ctx.arxiv_id: initial_messages(prompt, args.system_message)
        for ctx, prompt in zip(contexts, prompts, strict=True)
    }
    contexts_by_id = {ctx.arxiv_id: ctx for ctx in contexts}
    original_ids = [ctx.arxiv_id for ctx in contexts]
    final_rows: dict[str, dict] = {}
    exit_reasons: dict[str, str] = {}
    length_retries = {custom_id: 0 for custom_id in original_ids}
    tool_rounds_used = {custom_id: 0 for custom_id in original_ids}
    workers = max(1, min(args.request_workers, len(original_ids)))
    base_url = server_url.rstrip("/")
    print(
        f"Running reproduce loop for {len(original_ids)} episode(s) "
        f"with {workers} worker(s)",
        file=sys.stderr,
    )
    prepare_incremental_outputs(args)

    with ThreadPoolExecutor(max_workers=workers) as requests, ThreadPoolExecutor(
        max_workers=workers
    ) as tools:
        request_futures: dict[Future, dict[str, Any]] = {}
        tool_futures: dict[Future, dict[str, Any]] = {}

        def submit_request(custom_id: str, round_index: int, include_tools: bool) -> None:
            messages = conversation_for_round(
                conversations[custom_id],
                include_tools,
                budget_note=exit_reasons.get(custom_id) in EARLY_EXIT_REASONS,
                final_message=args.final_no_tools_message,
            )
            if include_tools:
                # Ephemeral, request-only: append the turn-budget counter without
                # persisting it into the stored conversation (no stale copies pile up).
                messages = [*messages, round_status_message(round_index, args.tool_rounds)]
            request = build_chat_completion_request(
                request_model,
                custom_id,
                messages,
                args,
                include_tools=include_tools,
                tool_choice="auto",
            )
            future = requests.submit(post_chat_completion_row, base_url, request, REQUEST_TIMEOUT)
            request_futures[future] = {
                "custom_id": custom_id,
                "round_index": round_index,
                "include_tools": include_tools,
            }

        for custom_id in original_ids:
            submit_request(custom_id, 0, include_tools=args.use_tools)

        while request_futures or tool_futures:
            done, _ = wait(set(request_futures) | set(tool_futures), return_when=FIRST_COMPLETED)
            for future in done:
                if future in request_futures:
                    handle_request_done(
                        future,
                        request_futures,
                        tool_futures,
                        tools,
                        conversations,
                        final_rows,
                        tool_rounds_used,
                        exit_reasons,
                        length_retries,
                        contexts_by_id,
                        args,
                    )
                    continue
                state = tool_futures.pop(future)
                future.result()
                custom_id = str(state["custom_id"])
                next_round = int(state["round_index"]) + 1
                include_tools = not state.get("force_final") and next_round < args.tool_rounds
                include_tools = apply_guardrails(
                    custom_id,
                    contexts_by_id[custom_id],
                    conversations[custom_id],
                    args,
                    exit_reasons,
                    include_tools,
                )
                submit_request(custom_id, next_round, include_tools)

    missing = [custom_id for custom_id in original_ids if custom_id not in final_rows]
    if missing:
        raise SystemExit(f"Missing final responses for: {', '.join(missing)}")


def handle_request_done(
    future: Future,
    request_futures: dict[Future, dict[str, Any]],
    tool_futures: dict[Future, dict[str, Any]],
    tools: ThreadPoolExecutor,
    conversations: dict[str, list[dict]],
    final_rows: dict[str, dict],
    tool_rounds_used: dict[str, int],
    exit_reasons: dict[str, str],
    length_retries: dict[str, int],
    contexts_by_id: dict[str, ExecutionContext],
    args: argparse.Namespace,
) -> None:
    state = request_futures.pop(future)
    custom_id = str(state["custom_id"])
    round_index = int(state["round_index"])
    try:
        result = future.result()
    except Exception as exc:  # noqa: BLE001 — a failed model call must not strand the run
        # The round's model call failed after the retry budget (a non-retryable 4xx, or a
        # hang that exhausted timeout+retries). Letting it propagate unwinds the whole loop
        # and leaves this run's row stuck at status="running" with no report.json. Finalize
        # just this episode as an error terminal instead: it reaches a terminal status and
        # still gets a (degraded) report for the auditor, and sibling episodes keep running.
        print(
            f"request {custom_id}: model call failed at round {round_index}: {exc}",
            file=sys.stderr,
        )
        finalize_episode(
            custom_id,
            {"custom_id": custom_id, "response": {"status_code": 0, "body": None, "error": str(exc)}},
            {},
            [],
            round_index,
            "error",
            conversations,
            final_rows,
            tool_rounds_used,
            contexts_by_id,
            args,
        )
        return
    row = response_row(custom_id, result)
    message = response_message(row)
    finish_reason = response_finish_reason(row)
    # Capture the model's real token usage for this response (every round, the
    # intermediate force-final pass, and the final turn each pass through here once).
    body = row.get("response", {}).get("body")
    usage = body.get("usage") if isinstance(body, dict) else None
    live_log.log_usage(contexts_by_id[custom_id], usage, round_index=round_index)
    # Stash the exact prompt-token count so the between-round context guardrails gate on
    # the server's own number instead of a chars-per-token estimate.
    if isinstance(usage, dict) and usage.get("prompt_tokens") is not None:
        contexts_by_id[custom_id].last_prompt_tokens = int(usage["prompt_tokens"])
    tool_calls = normalize_tool_calls(message.get("tool_calls") or [])
    if state["include_tools"] and tool_calls:
        # The episode acted, so whatever made an earlier turn overrun is behind it.
        # LENGTH_RETRY_LIMIT guards against a model stuck overrunning; counting
        # recovered truncations against it instead force-finals a healthy run at its
        # third unlucky turn, hours of budget unspent.
        length_retries[custom_id] = 0
        tool_rounds_used[custom_id] = max(tool_rounds_used[custom_id], round_index + 1)
        if round_index + 1 >= args.tool_rounds:
            exit_reasons[custom_id] = "round_limit"
        tool_future = tools.submit(
            append_tool_results,
            conversations[custom_id],
            message,
            tool_calls,
            contexts_by_id[custom_id],
            round_index,
        )
        tool_futures[tool_future] = state
        return
    if state["include_tools"]:
        if finish_reason == "length" and length_retries[custom_id] < LENGTH_RETRY_LIMIT:
            # The turn hit the output-token cap before emitting a tool call, so it was
            # cut off mid-reasoning. Don't read that as "the model is done" -- trim the
            # bloated reasoning, nudge it to act, and re-issue with tools still on (via
            # the noop tool-future path, so guardrails + the round limit still apply).
            length_retries[custom_id] += 1
            trimmed = trim_truncated_reasoning(message)
            conversations[custom_id].append(assistant_message(trimmed, tool_calls))
            conversations[custom_id].append(length_nudge_message())
            live_log.log_round_open(
                contexts_by_id[custom_id], trimmed,
                round_index=round_index, finish_reason=finish_reason,
            )
            tool_futures[tools.submit(noop)] = state
            return
        # Model stopped without a tool call while tools were live; re-issue one
        # tools-off pass to get the schema-constrained final submission. Trim first:
        # arriving here on a length-truncated turn (the retry budget is spent) would
        # otherwise carry the full cut-off reasoning into the report request, and the
        # model writes its report against a mid-stream ramble instead of the run.
        final_turn = trim_truncated_reasoning(message) if finish_reason == "length" else message
        conversations[custom_id].append(assistant_message(final_turn, tool_calls))
        live_log.log_round_open(
            contexts_by_id[custom_id], final_turn,
            round_index=round_index, finish_reason=finish_reason,
        )
        tool_futures[tools.submit(noop)] = {**state, "force_final": True}
        return
    exit_reason = exit_reasons.get(custom_id, "natural")
    finalize_episode(
        custom_id,
        row,
        message,
        tool_calls,
        round_index,
        exit_reason,
        conversations,
        final_rows,
        tool_rounds_used,
        contexts_by_id,
        args,
    )


# Tool-call dispatch — the key fork from the shared core.
#
# Where the auditor runs ``execute_tool_call(call, paper=paper)``, the repro
# loop runs ``execute_repro_tool_call(call, ctx)`` so each call acts on this
# episode's ``ExecutionContext`` (workspace, budget meter, allocation,
# evidence) rather than a read-only ``Paper``. Handler routing lives in
# ``tools/__init__.py`` (``build_repro_tools`` / ``execute_repro_tool_call``);
# this seam owns only recording each call's result into the transcript.
def append_tool_results(
    messages: list[dict],
    message: dict,
    tool_calls: list[dict],
    ctx: ExecutionContext,
    round_index: int | None = None,
) -> None:
    messages.append(assistant_message(message, tool_calls))
    # Stream to <run_dir>/agent.log: the round's reasoning first, then each call's
    # command before it runs and its result the moment it returns (see live_log).
    live_log.log_round_open(ctx, message, round_index=round_index)
    for call in tool_calls:
        live_log.log_call_start(ctx, call)
        result = execute_repro_tool_call(call, ctx)
        messages.append(tool_result_message(call, result))
        live_log.log_call_result(ctx, result)


# Between-round guardrails: compute budget + context tier.
#
# ``apply_guardrails`` runs once between tool rounds and decides whether the
# next request still offers tools: force-final when the compute budget is
# spent, then ``elide-compact`` — shrink the old span's bulky tool stdout in
# place so the loop keeps going (the tier that *replaces* the old hard context
# stop). Compaction elides only ``role:"tool"`` result contents (see
# ``compact.py``); the agent's own reasoning and its ``tool_calls`` stay
# verbatim, so the intent that was live going into compaction survives it. An
# earlier tier summarized the whole span with a model call, which flattened
# that intent and let agents finalize prematurely; an even earlier microcompact
# elided results with no on-disk pointer, which sent agents re-running whole
# GPU evals. Elision-with-pointer keeps both the reasoning and a path back to
# the full output (durable in ``evidence/``). The hard context cutoff no
# longer ends episodes; it survives only as a degraded backstop for when
# elision frees nothing and we are already past the real ceiling, so we never
# silently front-truncate the pinned system prompt + task head.
def apply_guardrails(
    custom_id: str,
    ctx: ExecutionContext,
    messages: list[dict[str, Any]],
    args: argparse.Namespace,
    exit_reasons: dict[str, str],
    include_tools: bool,
) -> bool:
    """Budget guardrail + context management, run between tool rounds.

    Returns whether the next request should still offer tools. Force-finals first
    when the compute budget is spent, then lets ``elide-compact`` shrink the old
    span's bulky tool stdout so the loop keeps going. The hard context cutoff no
    longer ends episodes; it survives only as a degraded backstop for when elision
    frees nothing and we are past the real ceiling (so we never silently
    front-truncate the prompt).
    """
    if not include_tools:
        gpu_session.release(ctx, "tools_off")
        return False
    # Bill any GPU wall held since the last charge so a held node depletes the budget
    # even across a long reasoning gap, and the ceiling is enforced mid-hold.
    gpu_session.charge_accrued(ctx)
    if ctx.budget is not None and ctx.budget.exhausted():
        gpu_session.release(ctx, "budget_exhausted")
        exit_reasons[custom_id] = "budget_exhausted"
        print(f"Stopping reproduce loop for {custom_id}: compute budget exhausted", file=sys.stderr)
        return False
    # The context tier gates on the model's own usage.prompt_tokens from the last
    # response (exact, free, backend-agnostic) — no chars-per-token estimate. That count
    # lags by this round's just-appended tool results, which is fine for a soft trigger:
    # the next round's real count catches up. When no usage has been recorded yet we skip
    # rather than estimate.
    if args.compact_enabled and _over(ctx, args, args.compact_threshold):
        elide_compaction(custom_id, ctx, messages, args, exit_reasons)
        if exit_reasons.get(custom_id) == "context_budget":
            return False
    return True


def elide_compaction(
    custom_id: str,
    ctx: ExecutionContext,
    messages: list[dict[str, Any]],
    args: argparse.Namespace,
    exit_reasons: dict[str, str],
) -> None:
    """Elide the old span's bulky tool stdout and keep going; degrade only on failure."""
    full_log = None if ctx.evidence is None else str(Path(ctx.evidence).parent / "agent.full.log")
    stats = compact.elide_compact(
        messages,
        keep_recent_tokens=args.compact_keep_tokens,
        full_log_path=full_log,
    )
    if stats["compacted"]:
        print(
            f"elide-compact {custom_id}: elided {stats['elided_messages']} tool result(s), "
            f"dropped reasoning on {stats['dropped_reasoning']} turn(s), "
            f"{stats['chars_before']}->{stats['chars_after']} chars",
            file=sys.stderr,
        )
        if ctx.evidence is not None:
            evidence_mod.append_trajectory(
                ctx.evidence,
                {
                    "type": "compaction",
                    "mode": "elide",
                    "custom_id": custom_id,
                    "elided_messages": stats["elided_messages"],
                    "dropped_reasoning": stats["dropped_reasoning"],
                    "chars_before": stats["chars_before"],
                    "chars_after": stats["chars_after"],
                },
            )
        return
    # Elision freed nothing (no old, bulky tool output). Keep going (retry next round)
    # unless the last real prompt-token count is already past the hard ceiling, in which
    # case end gracefully rather than let vLLM front-truncate the pinned system + task head.
    if _over(ctx, args, 1.0):
        gpu_session.release(ctx, "context_overflow")
        exit_reasons[custom_id] = "context_budget"
        print(
            f"Stopping reproduce loop for {custom_id}: context over ceiling and "
            f"elide-compact freed nothing ({stats.get('reason')})",
            file=sys.stderr,
        )


def _over(ctx: ExecutionContext, args: argparse.Namespace, threshold: float) -> bool:
    """True once the last measured prompt_tokens crosses ``threshold`` of the input budget.

    Uses the model's own ``usage.prompt_tokens`` (exact) — no chars-per-token estimate.
    Returns False when no usage has been recorded yet, so the caller skips compaction that
    round rather than guessing.
    """
    tokens = ctx.last_prompt_tokens
    return tokens is not None and tokens >= threshold * args.max_input_tokens


# Terminal finalization for one reproduce episode.
#
# Release the GPU, persist ``report.json``, flip the run row to a terminal
# status via ``log_final``, then write the completed response row. Reached on
# both a clean tools-off final pass and a failed model call
# (``exit_reason="error"``), so a run always reaches a terminal state with an
# account for the auditor rather than stranding as a permanently "running"
# row. ``report.json`` is written *before* ``log_final`` on purpose: the
# ``final`` event is what flips the run row terminal and the auditor's
# manifest expects the report to already exist. The write is best-effort —
# even a filesystem error still lets the run reach a terminal status rather
# than stranding it live with no account to grade.
def finalize_episode(
    custom_id: str,
    row: dict[str, Any],
    message: dict[str, Any],
    tool_calls: list[dict],
    round_index: int,
    exit_reason: str,
    conversations: dict[str, list[dict]],
    final_rows: dict[str, dict],
    tool_rounds_used: dict[str, int],
    contexts_by_id: dict[str, ExecutionContext],
    args: argparse.Namespace,
) -> None:
    """End one episode: release its GPU, persist ``report.json``, then flip it terminal.

    Reached from the tools-off final pass (``exit_reason`` natural / round_limit / …) and
    from a failed model call (``exit_reason="error"``). The report is persisted before the
    ``log_final`` event so a finished run always has an account on disk for the auditor.
    """
    ctx = contexts_by_id[custom_id]
    # Free any GPU allocation still held at the end of the episode (the model is told
    # to release itself, but never leak a node if it didn't).
    gpu_session.release(ctx, exit_reason)
    row["tool_loop"] = {
        "tool_rounds_used": tool_rounds_used[custom_id],
        "max_tool_rounds": args.tool_rounds,
        "hit_tool_round_limit": exit_reason == "round_limit",
        "exit_reason": exit_reason,
        "telemetry": loop_telemetry(conversations[custom_id], args.max_input_tokens),
    }
    # The terminal branch is only reached with tools off, so the final message always
    # carries the tools-off user turn.
    append_final_message(
        conversations[custom_id],
        message,
        tool_calls,
        False,
        budget_note=exit_reason in EARLY_EXIT_REASONS,
        final_message=args.final_no_tools_message,
    )
    final_rows[custom_id] = row
    # Persist the agent's account as report.json for the auditor, BEFORE the
    # terminal event below announces the run finished.
    _write_episode_report(ctx, message, exit_reason)
    live_log.log_final(
        ctx,
        message,
        round_index=round_index,
        exit_reason=exit_reason,
        finish_reason=response_finish_reason(row),
    )
    append_completed_outputs(custom_id, row, conversations[custom_id], args)
    # Reclaim NVMe scratch: drop the agent's venv plus any oversized files/dirs it
    # left in the workspace (e.g. an accidental full-dataset download). The auditor
    # grades report.json + evidence/, never the workspace, so this is safe here.
    cleanup.prune_workspace(
        ctx.workspace,
        getattr(args, "prune_workspace_threshold_mb", cleanup.DEFAULT_PRUNE_THRESHOLD_MB),
        remove_venv=getattr(args, "prune_workspace_venv", True),
        label=ctx.arxiv_id,
    )


def _write_episode_report(ctx: ExecutionContext, message: dict[str, Any], exit_reason: str) -> None:
    """Best-effort ``report.json`` write; a filesystem error must not block finalization."""
    try:
        report_path = report.write_episode_report(ctx, message.get("content") or "", exit_reason)
    except OSError as exc:  # disk full / read-only run dir — degrade, don't strand the run
        print(
            f"report {ctx.arxiv_id}: write failed ({exc}); finalizing without report.json",
            file=sys.stderr,
        )
        return
    if report_path is not None:
        print(f"report {ctx.arxiv_id}: wrote {report_path}", file=sys.stderr)
