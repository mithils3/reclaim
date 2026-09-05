from __future__ import annotations

import argparse
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from reclaim_repro.context import ExecutionContext
from reclaim_repro.loop import handle_request_done


def _length_body():
    # A reasoning-only turn cut off at the output-token cap: no tool_calls.
    return {
        "choices": [
            {
                "message": {"role": "assistant", "reasoning": "y" * 5000},
                "finish_reason": "length",
            }
        ]
    }


def _tool_call_body():
    # An ordinary productive turn: the episode emitted a tool call and acted.
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "c1", "type": "function",
                        "function": {"name": "workspace_bash", "arguments": '{"command": "ls"}'},
                    }],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }


def _done_future(body):
    fut: "Future" = Future()
    fut.set_result(body)
    return fut


def _harness(length_retries_start: int, body=None):
    custom_id = "2501.00001"
    future = _done_future(body if body is not None else _length_body())
    request_futures = {future: {"custom_id": custom_id, "round_index": 0, "include_tools": True}}
    tool_futures: dict = {}
    conversations = {custom_id: [{"role": "user", "content": "go"}]}
    final_rows: dict = {}
    tool_rounds_used = {custom_id: 0}
    exit_reasons: dict = {}
    length_retries = {custom_id: length_retries_start}
    contexts_by_id = {custom_id: ExecutionContext(arxiv_id=custom_id, evidence=None)}
    args = argparse.Namespace(tool_rounds=10, max_input_tokens=128000)
    with ThreadPoolExecutor(max_workers=1) as tools:
        handle_request_done(
            future, request_futures, tool_futures, tools, conversations, final_rows,
            tool_rounds_used, exit_reasons, length_retries, contexts_by_id, args,
        )
        for f in list(tool_futures):
            f.result()
    return {
        "custom_id": custom_id, "conversation": conversations[custom_id],
        "final_rows": final_rows, "length_retries": length_retries,
        "tool_futures": tool_futures,
    }


def test_length_retry_appends_trimmed_turn_and_nudge():
    out = _harness(length_retries_start=0)
    conv = out["conversation"]
    # user + trimmed assistant + user nudge
    assert conv[-2]["role"] == "assistant"
    assert conv[-1]["role"] == "user"
    assert "output-token limit" in conv[-1]["content"]
    # reasoning was trimmed, not left at 5000 chars
    assert len(conv[-2]["reasoning"]) < 5000
    assert out["length_retries"][out["custom_id"]] == 1
    # no terminal row landed; a follow-up tool-future was scheduled (tools still on)
    assert out["custom_id"] not in out["final_rows"]
    states = list(out["tool_futures"].values())
    assert states and not states[0].get("force_final")


def test_length_retry_exhausted_falls_back_to_force_final():
    out = _harness(length_retries_start=2)  # already at LENGTH_RETRY_LIMIT
    assert out["length_retries"][out["custom_id"]] == 2  # not incremented
    assert out["custom_id"] not in out["final_rows"]
    states = list(out["tool_futures"].values())
    assert states and states[0].get("force_final") is True


def test_force_final_after_truncation_carries_a_trimmed_turn():
    # The give-up path must not hand the report request the full cut-off ramble.
    out = _harness(length_retries_start=2)
    assert len(out["conversation"][-1]["reasoning"]) < 5000


def test_a_productive_round_clears_the_length_retry_budget():
    # The limit exists to stop a model stuck overrunning, not to retire a run that
    # overran, recovered, and kept working -- so acting resets the counter.
    out = _harness(length_retries_start=2, body=_tool_call_body())
    assert out["length_retries"][out["custom_id"]] == 0
    assert out["custom_id"] not in out["final_rows"]
