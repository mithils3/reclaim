"""``update_plan`` — a tiny, steerable checklist the agent keeps as it works.

Borrowed from OpenAI Codex's ``update_plan`` tool: the plan is *harness state*,
not free-text prose. The model resends the whole ordered checklist on every call;
the handler validates it (known statuses, at most one ``in_progress``), pins the
latest version onto the episode's :class:`ExecutionContext` (so it survives
context compaction), and snapshots a human-readable ``plan.md`` into the evidence
dir for the auditor to read.

Keeping it minimal is the point — a short spine the agent commits to and ticks
through, which curbs the "burn turns re-planning every turn" failure the prompt
warns against, and leaves a legible trace of intent next to the evidence.
"""

from __future__ import annotations

from typing import Any

from reclaim_vllm.config.config import function_tool

from reclaim_repro import evidence
from reclaim_repro.context import ExecutionContext

_STATUSES = ("pending", "in_progress", "completed")
_MARKS = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}


def update_plan(arguments: dict[str, Any], ctx: ExecutionContext) -> dict[str, Any]:
    """Validate and record the agent's plan; mirror it to ``evidence/plan.md``."""
    raw = arguments.get("plan")
    if not isinstance(raw, list) or not raw:
        return {"ok": False, "error": "Provide a non-empty 'plan' array of {step, status} items."}

    plan: list[dict[str, str]] = []
    in_progress = 0
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            return {"ok": False, "error": f"plan[{i}] must be an object with 'step' and 'status'."}
        step = item.get("step")
        status = item.get("status", "pending")
        if not isinstance(step, str) or not step.strip():
            return {"ok": False, "error": f"plan[{i}].step must be a non-empty string."}
        if status not in _STATUSES:
            return {"ok": False, "error": f"plan[{i}].status must be one of {list(_STATUSES)}; got {status!r}."}
        if status == "in_progress":
            in_progress += 1
        plan.append({"step": step.strip(), "status": status})

    if in_progress > 1:
        return {"ok": False, "error": "At most one step may be 'in_progress' at a time."}

    ctx.plan = plan
    rendered = _render(plan)
    if ctx.evidence is not None:
        evidence.save_plan(ctx.evidence, rendered)

    result: dict[str, Any] = {"ok": True, "tool": "update_plan", "plan": plan, "rendered": rendered}
    explanation = arguments.get("explanation")
    if isinstance(explanation, str) and explanation.strip():
        result["explanation"] = explanation.strip()
    return result


def _render(plan: list[dict[str, str]]) -> str:
    return "\n".join(f"{_MARKS[item['status']]} {item['step']}" for item in plan)


UPDATE_PLAN_TOOLS = [
    function_tool(
        "update_plan",
        "Record or update your step-by-step plan as a short checklist (the auditor and "
        "live viewer can see it). Call it once right after you scope the task, then again "
        "whenever a step's status changes: mark a step 'in_progress' before you start it "
        "and 'completed' once its evidence is written. Keep AT MOST ONE step 'in_progress'. "
        "Each call REPLACES the whole plan, so always resend the full ordered list. Keep it "
        "to a few concrete steps (~5-7), not a turn-by-turn play-by-play.",
        {
            "plan": {
                "type": "array",
                "description": "The full ordered checklist, resent in its entirety on every call.",
                "items": {
                    "type": "object",
                    "properties": {
                        "step": {"type": "string", "description": "One concrete step, short and imperative."},
                        "status": {
                            "type": "string",
                            "enum": list(_STATUSES),
                            "description": "pending | in_progress | completed.",
                        },
                    },
                    "required": ["step", "status"],
                },
            },
            "explanation": {
                "type": "string",
                "description": "Optional one-line note on what changed or why.",
            },
        },
        ["plan"],
    ),
]

UPDATE_PLAN_HANDLERS = {"update_plan": update_plan}
