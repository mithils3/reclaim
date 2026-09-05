"""Single chokepoint that keeps every tool result within the context budget."""

from __future__ import annotations

import json
from typing import Any

TRUNCATION_NOTE = (
    "Result truncated to fit the tool budget; request a specific file path, "
    "path_filter, or smaller range for more."
)
MIN_KEPT_CHARS = 200
TRANSIENT_ERROR_MARKERS = (
    "timeouterror",
    "timed out",
    "urlerror",
    "connectionerror",
    "connectionreseterror",
    "remotedisconnected",
    "rate limit",
    "429",
)


def truncate_tool_result(result: dict[str, Any], max_chars: int) -> dict[str, Any]:
    if serialized_length(result) <= max_chars:
        return result
    clamped = json.loads(json.dumps(result, ensure_ascii=False))
    for _ in range(64):
        if serialized_length(clamped) <= max_chars:
            break
        path, length = longest_string_leaf(clamped)
        if path is None or length <= MIN_KEPT_CHARS:
            break
        shrink_string_leaf(clamped, path, max(MIN_KEPT_CHARS, length // 2))
    clamped["truncated"] = True
    clamped["truncation_note"] = TRUNCATION_NOTE
    return clamped


def serialized_length(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False))


def longest_string_leaf(value: Any, path: tuple = ()) -> tuple[tuple | None, int]:
    if isinstance(value, str):
        return path, len(value)
    items: Any = ()
    if isinstance(value, dict):
        items = value.items()
    elif isinstance(value, list):
        items = enumerate(value)
    best_path, best_length = None, 0
    for key, child in items:
        child_path, child_length = longest_string_leaf(child, (*path, key))
        if child_length > best_length:
            best_path, best_length = child_path, child_length
    return best_path, best_length


def shrink_string_leaf(value: Any, path: tuple, keep_chars: int) -> None:
    parent = value
    for key in path[:-1]:
        parent = parent[key]
    leaf = str(parent[path[-1]])
    parent[path[-1]] = leaf[:keep_chars] + "…[truncated]"


def is_transient_error(result: dict[str, Any]) -> bool:
    if not isinstance(result, dict) or result.get("ok") is not False:
        return False
    text = str(result.get("error") or "").lower()
    return any(marker in text for marker in TRANSIENT_ERROR_MARKERS)
