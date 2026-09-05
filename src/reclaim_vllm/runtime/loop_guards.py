"""Guards that keep the tool loop inside its repeat and context budgets."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

# Conservative chars-per-token so the loop stops before vLLM would silently
# front-truncate the prompt at max_input_tokens.
BUDGET_CHARS_PER_TOKEN = 3


def repeated_tool_call(
    custom_id: str,
    tool_calls: list[dict],
    tool_call_counts: dict[str, Counter],
    max_repeats: int,
) -> str | None:
    counts = tool_call_counts[custom_id]
    seen_in_response: Counter = Counter()
    for call in tool_calls:
        signature = tool_call_signature(call)
        if counts[signature] + seen_in_response[signature] >= max_repeats:
            return f"{signature[0]}({signature[1]})"
        seen_in_response[signature] += 1
    return None


def record_tool_call(counts: Counter, call: dict, result: dict[str, Any]) -> None:
    # Failed calls are not counted, so retrying an errored call is never
    # treated as a repeat.
    if isinstance(result, dict) and result.get("ok") is False:
        return
    counts[tool_call_signature(call)] += 1


def tool_call_signature(call: dict) -> tuple[str, str]:
    function = call.get("function") or {}
    name = str(function.get("name") or "unknown_tool")
    arguments = function.get("arguments") or ""
    try:
        parsed = json.loads(arguments)
    except (TypeError, json.JSONDecodeError):
        return name, str(arguments)
    return name, json.dumps(parsed, ensure_ascii=False, sort_keys=True)


def conversation_chars(messages: list[dict[str, Any]]) -> int:
    chars = 0
    for message in messages:
        # ``reasoning`` counts: it is replayed to the server in the assistant message
        # (trace_io.assistant_message) and occupies real context there, so leaving it
        # out under-measured a reasoning model's prompt by more than half.
        chars += len(str(message.get("content") or ""))
        chars += len(str(message.get("reasoning") or ""))
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            chars += len(str(function.get("arguments") or ""))
    return chars


def context_budget_exceeded(messages: list[dict[str, Any]], max_input_tokens: int | None) -> bool:
    if not max_input_tokens:
        return False
    return conversation_chars(messages) >= max_input_tokens * BUDGET_CHARS_PER_TOKEN
