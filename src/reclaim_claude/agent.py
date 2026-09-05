"""Grade ONE agent reproduction run with Claude, on the native Messages API.

Same rubric, same prompt, same path-confined run-dir tools, and the same
deterministic ``finalize_audit_row`` as the vLLM auditor -- only the transport
differs, and the transport is the whole point: the Messages API supports prompt
caching, which an audit badly needs. A tool loop re-sends the entire conversation
every round, and the front of it (tool definitions, system prompt, rubric, central
claim, run manifest) never changes. Automatic caching -- one ``cache_control`` at
the top level of the request -- keeps the breakpoint on the last block as the
conversation grows, so each round re-reads the prefix at 0.1x the input price
instead of paying full freight for it.

Two other native-only features the auditor gets here: adaptive thinking at a
chosen effort (the grader reasons before it grades), and a schema-bound final turn
(``output_config.format``) so the verdict JSON validates instead of being coaxed
out of prose.

The Anthropic client is injected rather than constructed here, so the loop is
testable against a stub and the SDK import stays in the entry point.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from reclaim_vllm.config.config import (
    AUDIT_FINAL_NO_TOOLS_MESSAGE,
    AUDIT_SYSTEM_MESSAGE,
    TOOL_RESULT_MAX_CHARS,
)
from reclaim_vllm.schema.audit import AUDIT_JSON_SCHEMA
from reclaim_vllm.tools.run_dir_tools import AUDIT_TOOL_HANDLERS, AUDIT_TOOLS

MODEL = "claude-opus-4-8"
DEFAULT_EFFORT = "high"
DEFAULT_MAX_TOKENS = 32_000
DEFAULT_TOOL_ROUNDS = 25
CACHE_CONTROL = {"type": "ephemeral"}
SONNET_5_INTRO_UNTIL = "2026-08-31"  # $2/$10 per MTok through this date, $3/$15 after


def _tier(base_in: float, base_out: float) -> tuple[float, float, float, float]:
    """(input, output, 5m cache write, cache read) $/token from the base pair.

    Cache writes are 1.25x base input and reads 0.1x, uniformly across models.
    """
    return (base_in, base_out, base_in * 1.25, base_in * 0.1)


PRICES = {
    "claude-opus-4-8": _tier(5e-6, 25e-6),
    "claude-fable-5": _tier(10e-6, 50e-6),
    "claude-haiku-4-5": _tier(1e-6, 5e-6),
}


def prices_for(model: str | None, today: str | None = None) -> tuple[float, float, float, float]:
    """Per-token prices for one grader model, defaulting to the Opus tier.

    Sonnet 5 is date-dependent (introductory pricing runs out), so the cost line
    doesn't quietly halve or double when the promotion ends.
    """
    name = (model or "").rsplit("/", 1)[-1]
    if name.startswith("claude-sonnet-5"):
        stamp = today or time.strftime("%Y-%m-%d")
        return _tier(2e-6, 10e-6) if stamp <= SONNET_5_INTRO_UNTIL else _tier(3e-6, 15e-6)
    return PRICES.get(name, PRICES["claude-opus-4-8"])


@dataclass
class Usage:
    """Token counters summed across every request of one audit, priced by model."""

    model: str | None = None
    input: int = 0
    output: int = 0
    cache_write: int = 0
    cache_read: int = 0

    def add(self, usage: Any) -> None:
        self.input += int(getattr(usage, "input_tokens", 0) or 0)
        self.output += int(getattr(usage, "output_tokens", 0) or 0)
        self.cache_write += int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
        self.cache_read += int(getattr(usage, "cache_read_input_tokens", 0) or 0)

    def merge(self, other: "Usage") -> None:
        self.input += other.input
        self.output += other.output
        self.cache_write += other.cache_write
        self.cache_read += other.cache_read

    @property
    def cost(self) -> float:
        rate_in, rate_out, rate_write, rate_read = prices_for(self.model)
        return (
            self.input * rate_in
            + self.output * rate_out
            + self.cache_write * rate_write
            + self.cache_read * rate_read
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input,
            "output_tokens": self.output,
            "cache_creation_input_tokens": self.cache_write,
            "cache_read_input_tokens": self.cache_read,
            "cost_usd": round(self.cost, 4),
        }


@dataclass
class AuditResult:
    """One graded run: the model's final submission and how it was produced.

    Deliberately not parsed here. The caller feeds ``text`` + ``tool_loop`` to
    ``io.extracted_response``, the same finalizer the vLLM auditor uses, so both
    graders emit byte-identical verdict rows and the anti-cheat caps stay in one
    place.
    """

    text: str
    tool_loop: dict[str, Any]
    usage: Usage = field(default_factory=Usage)


def anthropic_tools(tools: list[dict[str, Any]] = AUDIT_TOOLS) -> list[dict[str, Any]]:
    """The shared run-dir tools, restated in Messages-API shape.

    ``AUDIT_TOOLS`` is written in OpenAI function-calling shape (the dialect the
    vLLM/OpenRouter auditor speaks). Rather than duplicate four tool definitions
    that must stay identical across graders, translate them: one source of truth
    for what the auditor may do to a run directory.
    """
    converted = []
    for tool in tools:
        function = tool.get("function") or {}
        converted.append(
            {
                "name": function.get("name", ""),
                "description": function.get("description", ""),
                "input_schema": function.get("parameters") or {"type": "object"},
            }
        )
    return converted


def verdict_schema(schema: dict[str, Any] = AUDIT_JSON_SCHEMA) -> dict[str, Any]:
    """``AUDIT_JSON_SCHEMA`` reduced to the subset structured outputs accepts.

    Structured outputs reject numeric bounds (``minimum``/``maximum``) and type
    unions written as a list. Both appear in the shared audit schema, so strip the
    bounds (``finalize_audit_row`` re-checks the score range in code anyway, where
    it is enforceable) and rewrite ``["number", "null"]`` as an ``anyOf``.
    """
    if not isinstance(schema, dict):
        return schema
    out: dict[str, Any] = {}
    for key, value in schema.items():
        if key in ("minimum", "maximum"):
            continue
        if key == "type" and isinstance(value, list):
            out["anyOf"] = [{"type": member} for member in value]
            continue
        if isinstance(value, dict):
            out[key] = verdict_schema(value)
        elif isinstance(value, list):
            out[key] = [verdict_schema(item) if isinstance(item, dict) else item for item in value]
        else:
            out[key] = value
    return out


def execute_tool(name: str, arguments: Any, run_dir: Path) -> dict[str, Any]:
    """Run one run-dir tool, turning any failure into a result the model can read."""
    handler = AUDIT_TOOL_HANDLERS.get(name)
    if handler is None:
        return {"ok": False, "error": f"Unknown tool: {name}"}
    try:
        return handler(dict(arguments or {}), Path(run_dir))
    except Exception as exc:  # noqa: BLE001 -- a tool crash is evidence, not a run-killer
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def run_audit(
    client: Any,
    *,
    prompt: str,
    run_dir: Path,
    model: str = MODEL,
    effort: str = DEFAULT_EFFORT,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    tool_rounds: int = DEFAULT_TOOL_ROUNDS,
    on_event: Callable[[str, int, dict[str, Any]], None] | None = None,
) -> AuditResult:
    """Investigate ``run_dir`` with the tools, then return the schema-bound verdict."""
    tools = anthropic_tools()
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    usage, round_index = Usage(model=model), 0
    rounds_used = tool_calls = tool_errors = 0
    exit_reason = "natural"

    for round_index in range(tool_rounds):
        message = _send(client, model=model, messages=messages, tools=tools,
                        effort=effort, max_tokens=max_tokens)
        usage.add(message.usage)
        messages.append({"role": "assistant", "content": message.content})
        _emit(on_event, "round_open", round_index, {"message": _round_message(message)})
        if getattr(message, "stop_reason", None) == "refusal":
            exit_reason = "refusal"
            break
        calls = [block for block in message.content if getattr(block, "type", "") == "tool_use"]
        if not calls:
            break
        rounds_used = round_index + 1
        results = []
        for call in calls:
            _emit(on_event, "call_start", round_index, {"call": _openai_call(call)})
            result = execute_tool(call.name, call.input, run_dir)
            tool_calls += 1
            tool_errors += 0 if result.get("ok") else 1
            _emit(on_event, "call_result", round_index, {"result": result})
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": _result_text(result),
                    "is_error": not result.get("ok", False),
                }
            )
        messages.append({"role": "user", "content": results})
        if round_index + 1 >= tool_rounds:
            exit_reason = "round_limit"

    tool_loop = {
        "tool_rounds_used": rounds_used,
        "max_tool_rounds": tool_rounds,
        "hit_tool_round_limit": exit_reason == "round_limit",
        "exit_reason": exit_reason,
        "telemetry": {"tool_calls": tool_calls, "tool_errors": tool_errors},
    }
    if exit_reason == "refusal":
        return AuditResult("", tool_loop, usage)

    # Tools off, schema on: the verdict is graded from the evidence just gathered.
    # Announced first because this turn is silent -- it makes no tool calls, so
    # without a marker a slow verdict looks like a hung process.
    _emit(on_event, "verdict_turn", round_index, {"rounds_used": rounds_used})
    final = _send(
        client,
        model=model,
        messages=[*messages, {"role": "user", "content": AUDIT_FINAL_NO_TOOLS_MESSAGE}],
        tools=tools,
        effort=effort,
        max_tokens=max_tokens,
        tool_choice={"type": "none"},
        schema=verdict_schema(),
    )
    usage.add(final.usage)
    # The caller emits the "final" event: only it holds the finalized verdict row
    # (score/verdict/flags) that belongs on that event alongside the submission.
    return AuditResult(_text_of(final), tool_loop, usage)


def _send(
    client: Any,
    *,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    effort: str,
    max_tokens: int,
    tool_choice: dict[str, Any] | None = None,
    schema: dict[str, Any] | None = None,
) -> Any:
    """One Messages request: adaptive thinking, automatic caching, streamed.

    ``cache_control`` at the top level is automatic caching: the API puts the
    breakpoint on the last cacheable block and moves it forward as the loop grows,
    so every round after the first reads the whole prior conversation from cache.
    Streaming is for the long tail -- a high-effort round with a big ``max_tokens``
    can outlive the SDK's non-streaming timeout guard.
    """
    output_config: dict[str, Any] = {"effort": effort}
    if schema is not None:
        output_config["format"] = {"type": "json_schema", "schema": schema}
    request: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": AUDIT_SYSTEM_MESSAGE,
        "messages": messages,
        "tools": tools,
        "thinking": {"type": "adaptive"},
        "output_config": output_config,
        "cache_control": CACHE_CONTROL,
    }
    if tool_choice is not None:
        request["tool_choice"] = tool_choice
    with client.messages.stream(**request) as stream:
        return stream.get_final_message()


def _result_text(result: dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False)[:TOOL_RESULT_MAX_CHARS]


def _text_of(message: Any) -> str:
    return "".join(
        block.text for block in message.content if getattr(block, "type", "") == "text"
    )


def _thinking_of(message: Any) -> str:
    return "".join(
        getattr(block, "thinking", "") or ""
        for block in message.content
        if getattr(block, "type", "") == "thinking"
    )


def _round_message(message: Any) -> dict[str, Any]:
    """The round's assistant turn in the shape the audit sink's row builders read."""
    return {"content": _text_of(message), "reasoning": _thinking_of(message)}


def _openai_call(call: Any) -> dict[str, Any]:
    """A tool_use block as the audit sink's ``call_row`` expects it."""
    return {
        "id": call.id,
        "type": "function",
        "function": {"name": call.name, "arguments": json.dumps(call.input, ensure_ascii=False)},
    }


def _emit(
    on_event: Callable[[str, int, dict[str, Any]], None] | None,
    kind: str,
    round_index: int,
    payload: dict[str, Any],
) -> None:
    if on_event is None:
        return
    try:
        on_event(kind, round_index, payload)
    except Exception:  # noqa: BLE001 -- telemetry must never break a paid audit
        return
