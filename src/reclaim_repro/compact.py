"""Context compaction by tool-stdout elision — the tier that keeps the loop going.

When the conversation nears ``max_input_tokens`` we recover space by replacing the
*contents* of old, bulky tool results with a short ``[elided N chars — on disk at
…]`` placeholder, and by dropping the chain of thought from old assistant turns.
Every assistant turn keeps its ``content`` and ``tool_calls``, every small tool
result, and the pinned head (system + task prompt) are kept **verbatim**.

Why elision rather than a model-call summary (the prior tier): most of a full
context is tool stdout — pip/download logs, GPU eval logs, file dumps. Eliding that
stdout leaves the agent's own words intact, so its stated intent (what it just
learned, what it planned to do next) survives the compaction boundary instead of
being flattened into a lossy third-person summary.

Old chain of thought goes too, and it has to: on a model whose template replays
prior-turn reasoning (DeepSeek-V4 forces this whenever tools are in the request)
the CoT is not a minor term. It measured most of the input ceiling on one sweep and
killed the majority of that sweep's runs on ``context_budget``, an un-elidable floor
that no amount of stdout elision could reach, because this tier used to keep
assistant turns whole. Dropping it costs the agent its verbatim recall of *how* it reached a
past conclusion while keeping *what* it concluded (``content``) and *what it did*
(``tool_calls``).

Nothing is lost to disk: both the elided stdout and the dropped CoT live in the
durable evidence store (``trajectory.jsonl`` + ``agent.full.log``), which the
placeholder points at, so an elided number can be re-read instead of re-computed
and the full reasoning trace survives for analysis.

Cut discipline (kept from the prior tier): keep ~``keep_recent_tokens`` of the
newest messages verbatim so fresh results — and the reasoning the agent is
actively working from — are never touched, and never compact behind the pinned
system + task head. We shrink ``role:"tool"`` contents in place and strip the
``reasoning`` key, but never drop or reorder a message, so no tool result is ever
stranded from the assistant ``tool_calls`` that produced it.

vLLM prefix-cache caveat: shrinking an old message invalidates the KV cache from
that message on. We fire rarely (only near the ceiling) and never touch the recent
tail, so the cache rebuilds a single time.
"""

from __future__ import annotations

from typing import Any

from reclaim_vllm.runtime.loop_guards import BUDGET_CHARS_PER_TOKEN, conversation_chars

# Tool results shorter than this stay verbatim — short confirmations ("ok", a path,
# a small count) cost little and reading them is cheaper than a disk round-trip.
ELIDE_MIN_CHARS = 1000
_ELIDED_PREFIX = "[elided "


def _message_chars(message: dict[str, Any]) -> int:
    chars = len(str(message.get("content") or "")) + len(str(message.get("reasoning") or ""))
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        chars += len(str(function.get("arguments") or ""))
    return chars


def choose_cut(messages: list[dict[str, Any]], keep_recent_tokens: int) -> int:
    """Index where the kept recent tail starts (always after the system message).

    Walks back from the newest message accumulating estimated tokens until
    ``keep_recent_tokens`` is reached. A return of ``<= 1`` means there is nothing
    old enough to compact (only the system message, or the whole conversation fits
    inside the keep window). No boundary snap is needed here — we shrink tool
    contents in place rather than slicing, so the tail can begin anywhere.
    """
    budget_chars = max(0, keep_recent_tokens) * BUDGET_CHARS_PER_TOKEN
    acc = 0
    cut = 1
    for index in range(len(messages) - 1, 0, -1):  # never cross index 0 (system)
        acc += _message_chars(messages[index])
        cut = index
        if acc >= budget_chars:
            break
    return cut


def _placeholder(n_chars: int, pointer: str) -> str:
    return f"{_ELIDED_PREFIX}{n_chars} chars — on disk at {pointer}]"


def elide_compact(
    messages: list[dict[str, Any]],
    *,
    keep_recent_tokens: int,
    full_log_path: str | None = None,
) -> dict[str, Any]:
    """Elide bulky old tool results and drop old chain of thought, in place.

    Mutates ``messages`` over the old span (index 2 up to the recent-tail cut): each
    ``role:"tool"`` result whose content exceeds ``ELIDE_MIN_CHARS`` has its content
    replaced with a pointer to the durable on-disk copy, and every message drops its
    ``reasoning`` key. The pinned head (``messages[0]`` system, ``messages[1]`` task
    prompt) and the recent tail are never touched, so the CoT the agent is actively
    working from survives. Returns a stats dict; ``compacted`` is False (with a ``reason``)
    when there was nothing old and bulky enough to free, leaving ``messages``
    unchanged so the caller can fall back to the hard-ceiling backstop.
    """
    cut = choose_cut(messages, keep_recent_tokens)
    # Span starts at index 2: messages[0] is system, messages[1] is the pinned task
    # prompt. A cut of <= 2 means there is nothing old enough behind the pinned head.
    if cut <= 2:
        return {"compacted": False, "reason": "nothing-old-enough"}
    pointer = full_log_path or "evidence/"
    chars_before = conversation_chars(messages)
    elided = 0
    freed = 0
    dropped_reasoning = 0
    for index in range(2, cut):
        message = messages[index]
        # Old assistant turns keep their content and tool_calls (the stated intent the
        # agent is still acting on) but lose their chain of thought. A reasoning model
        # whose template replays prior-turn CoT -- DeepSeek-V4 forces this whenever
        # tools are in the request -- otherwise accumulates an un-elidable floor that
        # no amount of tool-stdout elision can reach.
        reasoning = str(message.get("reasoning") or "")
        if reasoning:
            del message["reasoning"]
            dropped_reasoning += 1
            freed += len(reasoning)
        if message.get("role") != "tool":
            continue
        content = str(message.get("content") or "")
        if len(content) <= ELIDE_MIN_CHARS or content.startswith(_ELIDED_PREFIX):
            continue
        message["content"] = _placeholder(len(content), pointer)
        elided += 1
        freed += len(content)
    if elided == 0 and dropped_reasoning == 0:
        return {"compacted": False, "reason": "no-bulk-tool-output-or-reasoning"}
    return {
        "compacted": True,
        "elided_messages": elided,
        "dropped_reasoning": dropped_reasoning,
        "freed_chars": freed,
        "chars_before": chars_before,
        "chars_after": conversation_chars(messages),
    }
