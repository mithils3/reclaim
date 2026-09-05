from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def trace_output_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}_trace{output.suffix}")


def append_trace_row(
    path: Path,
    custom_id: str,
    messages: list[dict[str, Any]],
    final_row: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "custom_id": custom_id,
        "messages": messages,
        "final_response": final_row,
        "tool_loop": final_row.get("tool_loop", {}),
    }
    with path.open("a", encoding="utf-8") as handle:
        json.dump(row, handle, ensure_ascii=False)
        handle.write("\n")


def assistant_message(message: dict[str, Any], tool_calls: list[dict[str, Any]]) -> dict[str, Any]:
    item: dict[str, Any] = {"role": "assistant"}
    if message.get("content") is not None:
        item["content"] = message["content"]
    if message.get("reasoning"):
        item["reasoning"] = message["reasoning"]
    if tool_calls:
        item["tool_calls"] = tool_calls
    return item
