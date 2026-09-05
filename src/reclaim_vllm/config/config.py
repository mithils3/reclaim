from __future__ import annotations

from pathlib import Path
from typing import Any


MINIMAX_M2_MODEL = "MiniMaxAI/MiniMax-M2.7"
DEFAULT_MODEL = MINIMAX_M2_MODEL
PLACEHOLDER = "{PAPER_TEXT}"
MAX_MODEL_LEN = 196608
TOOL_TIMEOUT = 20.0
TOOL_MAX_CHARS = 24_000
TOOL_RESULT_MAX_CHARS = 40_000
# Identical repeated tool calls before the loop forces the final turn
# (read off the args namespace by runtime/tool_loop.py).
MAX_REPEATED_TOOL_CALLS = 2
REQUEST_TIMEOUT = 1800.0
CONTEXT_BUDGET_NOTE = (
    "The conversation hit its context budget, so the tool phase ended early. "
    "Mark any category you could not finish checking as tool_failed or "
    "paper_text_only instead of guessing. "
)

# --- Audit mode (LLM reproduction auditor) ---------------------------------
# The auditor grades one agent reproduction attempt per paper against the rubric,
# using the central_claim (from the audit pool) plus path-confined run-dir tools
# scoped to the agent's run directory. Replaces the deterministic
# verification-target curator.
CLAIM_PLACEHOLDER = "{CENTRAL_CLAIM}"
BUNDLE_PLACEHOLDER = "{RUN_BUNDLE}"
RUBRIC_PLACEHOLDER = "{RUBRIC}"
AUDIT_PROMPT_FILE = Path("prompts/prompt_audit.txt")
AUDIT_RUBRIC_FILE = Path("rubric_audit.md")
AUDIT_DEFAULT_OUTPUT = Path("outputs/v5/audit_pool_audit_verdicts.jsonl")
AUDIT_DEFAULT_EXTRACTED = Path("outputs/v5/audit_pool_audit_verdicts_extracted.jsonl")
# Source of per-paper central claims (the audit pool).
AUDIT_CLAIMS_DEFAULT = Path("outputs/v5/audit_pool_extracted.jsonl")
# One agent reproduction run directory per paper, at AUDIT_RUNS_DIR_DEFAULT/<id>.
AUDIT_RUNS_DIR_DEFAULT = Path("outputs/v5/agent_runs")
RUN_FILE_DEFAULT_CHARS = 40_000
RUN_FILE_MAX_CHARS = 200_000
RUN_FILE_WRITE_MAX_CHARS = 200_000
RUN_MANIFEST_MAX_ENTRIES = 200
BASH_TIMEOUT = 60


def bounded(value: Any, default: int, maximum: int) -> int:
    """Clamp an agent-supplied integer argument, falling back on junk.

    Every tool that takes a ``timeout`` / ``max_chars`` / ``gpus`` from the model
    runs it through here, so the floor, the ceiling and the "missing or
    unparseable means default" rule are one decision rather than one per tool.
    It lives in config because both agents' tool modules already import config —
    homing it in either one's tool package would drag that whole package into the
    other's import graph.
    """
    if value in (None, ""):
        return default
    try:
        return max(1, min(int(value), maximum))
    except (TypeError, ValueError):
        return default
AUDIT_SYSTEM_MESSAGE = (
    "You are an adversarial reproduction auditor for an ML reproduction "
    "benchmark. You grade ONE agent's attempt to reproduce ONE paper's central "
    "claim by investigating the agent's run directory: its *.log files, output "
    "artifacts, and any code it wrote. You have tools scoped to that "
    "directory: list_run_files (list files and folders), read_run_file (read one "
    "file), write_run_file (write a new file, e.g. a re-scoring script, into the "
    "run directory so you can run it and cite it), and bash (run a shell command "
    "with the run directory as the working directory, e.g. grep the logs or run "
    "python3). For a multi-line re-score, write_run_file a .py script and run it "
    "with bash rather than fighting python3 -c quoting. You never trust a number "
    "because the agent printed it: open the file it came from and trace how it "
    "was produced, and flag hardcoded constants, echoed prose numbers, "
    "self-scored or fabricated predictions, wrong split/scale/dataset, and "
    "cherry-picked metrics. Default to a low score; absence of evidence is a low "
    "score, not a pass. When you have gathered enough evidence, grade the attempt "
    "with an integer 0-10 score per the rubric's score scale and return only the "
    "JSON object matching the schema."
)
AUDIT_FINAL_NO_TOOLS_MESSAGE = (
    "Write the final audit verdict JSON now from the run bundle above. Apply "
    "every rubric criterion and cite evidence from the bundle: restate the "
    "target (metric, reference value, op, tolerance); give execution evidence; "
    "report the measured value with an exact file/line or log citation; list "
    "every anti-cheat flag with its evidence and severity; give the op/tolerance "
    "comparison and methodology notes; end with an integer 0-10 score per the "
    "rubric's score scale and a one-paragraph rationale. "
    "Return only the JSON object: the first output character must be { and the "
    "last must be }. No prose, no markdown."
)


def function_tool(
    name: str,
    description: str,
    properties: dict,
    required: list[str],
) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }
