from __future__ import annotations

import argparse
import sys
import threading
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import Any

from reclaim_vllm.vllm.io import (
    append_jsonl_row,
    build_chat_completion_request,
    extracted_response,
    initial_messages,
    normalize_tool_calls,
    response_message,
    truncate_output_file,
    tool_result_message,
)
from reclaim_vllm.config.config import CONTEXT_BUDGET_NOTE, REQUEST_TIMEOUT
from reclaim_vllm.runtime.loop_guards import context_budget_exceeded, record_tool_call, repeated_tool_call
from reclaim_vllm.runtime import live_events
from reclaim_vllm.papers.papers import Paper
from reclaim_vllm.runtime.run_health import loop_telemetry
from reclaim_vllm.tools.web_tools import execute_tool_call
from reclaim_vllm.runtime.trace_io import append_trace_row, assistant_message
from reclaim_vllm.vllm.client import post_chat_completion_row, response_row

# Serializes output appends so concurrent workers never interleave half-written lines.
OUTPUT_WRITE_LOCK = threading.Lock()


def run_tool_loop(
    args: argparse.Namespace,
    papers: list[Paper],
    prompts: list[str],
    server_url: str,
    model_id: str,
) -> None:
    # The attached server is addressed by the id it advertises, resolved by the
    # caller (resolve_served_model returns a non-empty id or raises).
    conversations = {
        paper.arxiv_id: initial_messages(prompt, args.system_message)
        for paper, prompt in zip(papers, prompts, strict=True)
    }
    papers_by_id = {paper.arxiv_id: paper for paper in papers}
    original_ids = [paper.arxiv_id for paper in papers]
    final_rows: dict[str, dict] = {}
    exit_reasons: dict[str, str] = {}
    tool_call_counts = {custom_id: Counter() for custom_id in original_ids}
    tool_rounds_used = {custom_id: 0 for custom_id in original_ids}
    workers = max(1, min(args.request_workers, len(original_ids)))
    base_url = server_url.rstrip("/")
    print(
        f"Running async tool loop for {len(original_ids)} request(s) "
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
                budget_note=exit_reasons.get(custom_id) == "context_budget",
                final_message=args.final_no_tools_message,
            )
            request = build_chat_completion_request(
                model_id,
                custom_id,
                messages,
                args,
                include_tools=include_tools,
                tool_choice="auto",
            )
            future = requests.submit(
                post_chat_completion_row,
                base_url,
                request,
                REQUEST_TIMEOUT,
            )
            request_futures[future] = {
                "custom_id": custom_id,
                "round_index": round_index,
                "include_tools": include_tools,
            }

        for custom_id in original_ids:
            submit_request(custom_id, 0, include_tools=args.use_tools)

        while request_futures or tool_futures:
            done, _ = wait(
                set(request_futures) | set(tool_futures),
                return_when=FIRST_COMPLETED,
            )
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
                        tool_call_counts,
                        papers_by_id,
                        args,
                    )
                    continue
                state = tool_futures.pop(future)
                future.result()
                custom_id = str(state["custom_id"])
                next_round = int(state["round_index"]) + 1
                include_tools = not state.get("force_final") and next_round < args.tool_rounds
                if include_tools and context_budget_exceeded(
                    conversations[custom_id], args.max_input_tokens
                ):
                    exit_reasons[custom_id] = "context_budget"
                    include_tools = False
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
    tool_call_counts: dict[str, Counter],
    papers_by_id: dict[str, Paper],
    args: argparse.Namespace,
) -> None:
    state = request_futures.pop(future)
    custom_id = str(state["custom_id"])
    round_index = int(state["round_index"])
    try:
        completion = future.result()
    except Exception as exc:  # noqa: BLE001 — a failed model call must not strand the run
        # The round's model call failed after the retry budget (a non-retryable 4xx, or
        # a hang that exhausted timeout+retries). Letting it propagate unwinds the whole
        # loop: the paper never reaches final_rows, the run exits non-zero with no
        # verdict written, and its telemetry row is left at status="running" forever.
        # Mirror reclaim_repro.loop and finalize just this paper as an error terminal,
        # so it still gets a (degraded) verdict row and its siblings keep running.
        finalize_failed_request(
            custom_id, round_index, exc, conversations, final_rows,
            tool_rounds_used, exit_reasons, args,
        )
        return
    row = response_row(custom_id, completion)
    message = response_message(row)
    tool_calls = normalize_tool_calls(message.get("tool_calls") or [])
    if state["include_tools"] and tool_calls:
        tool_rounds_used[custom_id] = max(tool_rounds_used[custom_id], round_index + 1)
        repeated = repeated_tool_call(
            custom_id,
            tool_calls,
            tool_call_counts,
            args.max_repeated_tool_calls,
        )
        if repeated:
            print(
                f"Stopping tool loop for {custom_id}: repeated tool call {repeated}",
                file=sys.stderr,
            )
            exit_reasons[custom_id] = "repeated_call_cutoff"
            tool_future = tools.submit(noop)
            tool_futures[tool_future] = {**state, "force_final": True}
            return
        if round_index + 1 >= args.tool_rounds:
            exit_reasons[custom_id] = "round_limit"
        live_events.round_open(custom_id, round_index, message)
        tool_future = tools.submit(
            append_tool_results,
            conversations[custom_id],
            message,
            tool_calls,
            papers_by_id[custom_id],
            tool_call_counts[custom_id],
            custom_id,
            round_index,
        )
        tool_futures[tool_future] = state
        return
    if state["include_tools"]:
        # The model stopped without a tool call while tools were still enabled,
        # so re-issue one tools-off pass to get schema-constrained final JSON.
        conversations[custom_id].append(assistant_message(message, tool_calls))
        tool_future = tools.submit(noop)
        tool_futures[tool_future] = {**state, "force_final": True}
        return
    exit_reason = exit_reasons.get(custom_id, "natural")
    row["tool_loop"] = {
        "tool_rounds_used": tool_rounds_used[custom_id],
        "max_tool_rounds": args.tool_rounds,
        "hit_tool_round_limit": exit_reason == "round_limit",
        "exit_reason": exit_reason,
        "telemetry": loop_telemetry(conversations[custom_id], args.max_input_tokens),
    }
    append_final_message(
        conversations[custom_id],
        message,
        tool_calls,
        bool(state["include_tools"]),
        budget_note=exit_reason == "context_budget",
        final_message=args.final_no_tools_message,
    )
    final_rows[custom_id] = row
    append_completed_outputs(custom_id, row, conversations[custom_id], args)
    live_events.final(custom_id, round_index, message, exit_reason,
                      extracted_response(custom_id, row))


def finalize_failed_request(
    custom_id: str,
    round_index: int,
    exc: BaseException,
    conversations: dict[str, list[dict]],
    final_rows: dict[str, dict],
    tool_rounds_used: dict[str, int],
    exit_reasons: dict[str, str],
    args: argparse.Namespace,
) -> None:
    """Terminate one paper on a dead model call, degraded but accounted for.

    Writes the same row shape reclaim_repro.loop uses for its error terminal
    (``status_code`` 0, no body, the error string), which ``extracted_response``
    already funnels to ``degraded_row``. Landing in ``final_rows`` is what keeps the
    end-of-loop ``missing`` check from turning one paper's failure into a SystemExit
    for the whole batch.
    """
    print(
        f"request {custom_id}: model call failed at round {round_index}: {exc}",
        file=sys.stderr,
        flush=True,
    )
    exit_reasons[custom_id] = "error"
    row: dict[str, Any] = {
        "custom_id": custom_id,
        "response": {"status_code": 0, "body": None, "error": str(exc)},
        "tool_loop": {
            "tool_rounds_used": tool_rounds_used[custom_id],
            "max_tool_rounds": args.tool_rounds,
            "hit_tool_round_limit": False,
            "exit_reason": "error",
            "telemetry": loop_telemetry(conversations[custom_id], args.max_input_tokens),
        },
    }
    final_rows[custom_id] = row
    append_completed_outputs(custom_id, row, conversations[custom_id], args)
    live_events.final(
        custom_id, round_index, {}, "error",
        extracted_response(custom_id, row),
    )


def prepare_incremental_outputs(args: argparse.Namespace) -> None:
    truncate_output_file(args.output)
    truncate_output_file(args.extracted_output)
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
        append_jsonl_row(
            args.extracted_output,
            extracted_response(custom_id, row),
            truncate=False,
        )
        if args.save_round_jsonl:
            append_trace_row(args.trace_output, custom_id, messages, row)


def noop() -> None:
    return None


def final_user_message(budget_note: bool, final_message: str) -> dict[str, Any]:
    content = final_message
    if budget_note:
        content = CONTEXT_BUDGET_NOTE + content
    return {"role": "user", "content": content}


def conversation_for_round(
    messages: list[dict],
    include_tools: bool,
    *,
    budget_note: bool = False,
    final_message: str,
) -> list[dict]:
    if include_tools:
        return messages
    return [*messages, final_user_message(budget_note, final_message)]


def append_tool_results(
    messages: list[dict],
    message: dict,
    tool_calls: list[dict],
    paper: Paper,
    counts: Counter,
    custom_id: str = "",
    round_index: int = 0,
) -> None:
    messages.append(assistant_message(message, tool_calls))
    for call in tool_calls:
        live_events.call_start(custom_id, round_index, call)
        result = execute_tool_call(call, paper=paper)
        live_events.call_result(custom_id, round_index, result)
        record_tool_call(counts, call, result)
        messages.append(tool_result_message(call, result))


def append_final_message(
    messages: list[dict[str, Any]],
    message: dict[str, Any],
    tool_calls: list[dict[str, Any]],
    include_tools: bool,
    *,
    budget_note: bool = False,
    final_message: str,
) -> None:
    if not include_tools:
        messages.append(final_user_message(budget_note, final_message))
    messages.append(assistant_message(message, tool_calls))
