"""Deterministic run-health: the model reports evidence, this module computes status."""

from __future__ import annotations

import json
from typing import Any

from reclaim_vllm.runtime.loop_guards import conversation_chars

DEGRADED = "degraded"
INCOMPLETE_EXIT_REASONS = ("round_limit", "repeated_call_cutoff", "context_budget")


def loop_telemetry(messages: list[dict[str, Any]], max_input_tokens: int | None) -> dict[str, Any]:
    tool_calls = 0
    tool_errors = 0
    for message in messages:
        if message.get("role") != "tool":
            continue
        tool_calls += 1
        if tool_result_failed(str(message.get("content") or "")):
            tool_errors += 1
    chars = conversation_chars(messages)
    estimated_tokens = chars // 4
    return {
        "tool_calls": tool_calls,
        "tool_errors": tool_errors,
        "conversation_chars": chars,
        "estimated_input_tokens": estimated_tokens,
        "input_overflow": bool(max_input_tokens) and estimated_tokens >= max_input_tokens,
    }


def tool_result_failed(content: str) -> bool:
    try:
        envelope = json.loads(content)
    except json.JSONDecodeError:
        return False
    return isinstance(envelope, dict) and envelope.get("ok") is False


def loop_exit_reason(tool_loop: dict[str, Any]) -> str:
    reason = tool_loop.get("exit_reason")
    if reason:
        return str(reason)
    return "round_limit" if tool_loop.get("hit_tool_round_limit") else "natural"


def degraded_row(custom_id: str, content: str, parsed: Any, tool_loop: dict[str, Any]) -> dict[str, Any]:
    return {
        "custom_id": custom_id,
        "extracted_json": parsed if not isinstance(parsed, dict) else None,
        "raw_content": content,
        "verification_status": DEGRADED,
        "web_verification": "unavailable",
        "score": None,
        "tier": None,
        "exit_reason": loop_exit_reason(tool_loop),
    }
