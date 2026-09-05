"""Command-line argument definition, validation, and audit-mode defaults.

Split out of run_arxiv_prompt_vllm.py to keep that entry point focused on the
run flow. ``parse_args`` returns a fully-resolved argparse Namespace: model
profile defaults applied, audit-mode settings filled in, and all
cross-argument validation enforced.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from reclaim_vllm.config.config import (
    AUDIT_CLAIMS_DEFAULT,
    AUDIT_DEFAULT_EXTRACTED,
    AUDIT_DEFAULT_OUTPUT,
    AUDIT_FINAL_NO_TOOLS_MESSAGE,
    AUDIT_PROMPT_FILE,
    AUDIT_RUBRIC_FILE,
    AUDIT_RUNS_DIR_DEFAULT,
    AUDIT_SYSTEM_MESSAGE,
    DEFAULT_MODEL,
    MAX_MODEL_LEN,
    MAX_REPEATED_TOOL_CALLS,
)
from reclaim_vllm.schema.audit import AUDIT_RESPONSE_FORMAT
from reclaim_vllm.tools.run_dir_tools import AUDIT_TOOLS
from reclaim_vllm.runtime.trace_io import trace_output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("audit",),
        default="audit",
        help="audit grades an agent reproduction attempt against the rubric (the only mode).",
    )
    parser.add_argument(
        "--claims",
        type=Path,
        help=(
            "Audit-pool rows (classifier extracted output) carrying the "
            "central_claim per paper, injected into the audit prompt. A local "
            "JSONL path or an hf://datasets/<owner>/<name>/<file> reference. "
            f"Default: {AUDIT_CLAIMS_DEFAULT}."
        ),
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        help=(
            "Root directory of agent reproduction runs; the auditor reads one "
            "run dir per paper at <runs-dir>/<arxiv_id> via the path-confined "
            f"run-dir tools (default: {AUDIT_RUNS_DIR_DEFAULT})."
        ),
    )
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--extracted-output", type=Path)
    parser.add_argument("--trace-output", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--vllm-server-url",
        help=(
            "Served vLLM chat-completions base URL. If omitted, the runner also "
            "checks $RECLAIM_SERVER_URL and the base_url in the file named by "
            "$RECLAIM_ENDPOINT_FILE (published by reclaim_serve). With no endpoint "
            "the auditor exits with an error — it does not self-host a model."
        ),
    )
    parser.add_argument(
        "--served-model-name",
        help=(
            "Model id to send in requests when attached to an existing server. "
            "Defaults to the single model the server advertises at /v1/models "
            "(also reads $RECLAIM_SERVED_MODEL)."
        ),
    )
    parser.add_argument(
        "--paper-ids-file",
        type=Path,
        help="Run only the arXiv ids listed in this file (one per line).",
    )
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--max-input-tokens", type=int, default=128000)
    parser.add_argument("--tool-rounds", type=int, default=10)
    parser.add_argument("--request-workers", type=int, default=8)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--save-round-jsonl", action="store_true")
    args = parser.parse_args()
    apply_model_defaults(args)
    resolve_mode_settings(args)
    if args.tool_rounds < 1:
        parser.error("--tool-rounds must be >= 1")
    if args.request_workers < 1:
        parser.error("--request-workers must be >= 1")
    if args.max_input_tokens < 1:
        parser.error("--max-input-tokens must be >= 1")
    if args.top_k is not None and args.top_k < 1:
        parser.error("--top-k must be >= 1")
    if args.max_input_tokens + args.max_tokens > args.max_model_len:
        parser.error("--max-input-tokens + --max-tokens must fit within model context")
    if args.trace_output is None:
        args.trace_output = trace_output_path(args.output)
    return args


def apply_model_defaults(args: argparse.Namespace) -> None:
    """Fill the request-time length default.

    The auditor is a URL-only client: it only fills the request-time fields it
    POSTs to an already-served model. The vLLM engine/serve-launch flags (tensor
    parallel, tool/reasoning parsers, compilation config, kv-cache dtype, ...)
    live in ``reclaim_serve/profiles.py``, the single source of truth for how a
    model is served, not here.

    Sampling is left unset for every model unless the user set it explicitly:
    the request builder omits unset fields, so the served model's own
    generation_config defaults apply (vLLM recipe style).
    """
    args.max_model_len = args.max_model_len or MAX_MODEL_LEN


def resolve_mode_settings(args: argparse.Namespace) -> None:
    """Fill prompt/output/schema/message defaults for the audit mode."""
    args.prompt_file = args.prompt_file or AUDIT_PROMPT_FILE
    # The audit rubric is fixed; run_arxiv_prompt_vllm reads this attribute.
    args.rubric_file = AUDIT_RUBRIC_FILE
    args.output = args.output or AUDIT_DEFAULT_OUTPUT
    args.extracted_output = args.extracted_output or AUDIT_DEFAULT_EXTRACTED
    args.claims = args.claims or AUDIT_CLAIMS_DEFAULT
    args.runs_dir = args.runs_dir or AUDIT_RUNS_DIR_DEFAULT
    args.response_format = AUDIT_RESPONSE_FORMAT
    args.system_message = AUDIT_SYSTEM_MESSAGE
    args.final_no_tools_message = AUDIT_FINAL_NO_TOOLS_MESSAGE
    # The auditor explores the agent's run directory with the run-dir tools.
    args.tools = AUDIT_TOOLS
    args.use_tools = True
    # Fixed loop guard (was --max-repeated-tool-calls, never varied).
    args.max_repeated_tool_calls = MAX_REPEATED_TOOL_CALLS

