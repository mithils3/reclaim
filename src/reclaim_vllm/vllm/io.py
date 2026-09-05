from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from reclaim_vllm.audit.audit import finalize_audit_row
from reclaim_vllm.runtime.run_health import degraded_row


def initial_messages(prompt: str, system_message: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]


def build_chat_completion_request(
    model: str,
    custom_id: str,
    messages: list[dict[str, Any]],
    args: argparse.Namespace,
    *,
    include_tools: bool,
    tool_choice: str = "auto",
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": args.max_tokens,
        "truncate_prompt_tokens": args.max_input_tokens,
    }
    # Sampling fields are sent only when set; an unset field is omitted so the
    # served model's own generation_config defaults apply (vLLM recipe style).
    if args.temperature is not None:
        body["temperature"] = args.temperature
    if args.top_p is not None:
        body["top_p"] = args.top_p
    if args.top_k is not None:
        body["top_k"] = args.top_k
    if include_tools:
        body["tools"] = args.tools
        body["tool_choice"] = tool_choice
    else:
        body["response_format"] = args.response_format
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }


def append_jsonl_row(path: Path, row: dict[str, Any], *, truncate: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if truncate else "a"
    with path.open(mode, encoding="utf-8") as handle:
        json.dump(row, handle, ensure_ascii=False)
        handle.write("\n")


def truncate_output_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def extracted_response(custom_id: str, row: dict[str, Any]) -> dict[str, Any]:
    message = response_message(row)
    content = message.get("content") or ""
    tool_loop = row.get("tool_loop") or {}
    parsed = parse_json_content(content)
    if not isinstance(parsed, dict):
        return degraded_row(custom_id, content, parsed, tool_loop)
    result: dict[str, Any] = {"custom_id": custom_id}
    result.update(finalize_audit_row(parsed, tool_loop))
    return result


def parse_json_content(content: str) -> Any | None:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(content[start : end + 1])
    except json.JSONDecodeError:
        return None


def response_message(row: dict[str, Any]) -> dict[str, Any]:
    # ``body`` is explicitly None on a failed-model-call row, so default-to-{} is not
    # enough: .get("body", {}) returns the None that is actually there.
    body = (row.get("response") or {}).get("body") or {}
    choices = body.get("choices") or []
    if not choices:
        return {}
    return choices[0].get("message") or {}


def response_finish_reason(row: dict[str, Any]) -> str | None:
    body = row.get("response", {}).get("body") or {}
    choices = body.get("choices") or []
    if not choices:
        return None
    return choices[0].get("finish_reason")


def normalize_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for index, call in enumerate(tool_calls):
        item = dict(call)
        if not item.get("id"):
            item["id"] = f"call_{index}"
        item.setdefault("function", {})
        normalized.append(item)
    return normalized


def tool_result_message(call: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    function = call.get("function") or {}
    return {
        "role": "tool",
        "tool_call_id": call.get("id", ""),
        "name": function.get("name", "unknown_tool"),
        "content": json.dumps(result, ensure_ascii=False),
    }
