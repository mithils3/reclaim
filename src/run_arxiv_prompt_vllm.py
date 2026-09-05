#!/usr/bin/env python3
"""Auditor entry point: grade agent reproduction runs with vLLM."""

from __future__ import annotations

import sys

from reclaim_vllm.config.config import CLAIM_PLACEHOLDER
from reclaim_vllm.audit.inputs import build_audit_prompt, load_audit_rubric
from reclaim_vllm.config.cli_args import parse_args
from reclaim_vllm.runtime.mre_records import load_mre_records
from reclaim_vllm.runtime.audit_sink import SinkConfig as AuditSinkConfig, install as install_audit_sink
from reclaim_vllm.papers.papers import Paper
from reclaim_vllm.runtime.tool_loop import run_tool_loop
from reclaim_vllm.vllm.endpoint import resolve_served_model, resolve_server_url


def main() -> int:
    args = parse_args()
    prompt_template = args.prompt_file.read_text(encoding="utf-8")
    if CLAIM_PLACEHOLDER not in prompt_template:
        raise SystemExit(f"{args.prompt_file} must contain {CLAIM_PLACEHOLDER}.")

    # Claims-only: the audit-pool rows ARE the paper list; the auditor reads
    # the agent's run directory (one per paper) with the run-dir tools, not
    # the paper text.
    claim_records = load_mre_records(args.claims)
    rubric = load_audit_rubric(args.rubric_file)
    papers = [
        Paper(arxiv_id=arxiv_id, run_dir=run_dir_for(args.runs_dir, arxiv_id))
        for arxiv_id in claim_records
    ]
    if args.paper_ids_file:
        papers = filter_papers_by_ids(papers, args.paper_ids_file)
    prompts = [
        build_audit_prompt(
            prompt_template,
            rubric,
            claim_records.get(paper.arxiv_id),
            paper.arxiv_id,
            args.runs_dir,
        )
        for paper in papers
    ]

    print(f"Running {len(prompts)} prompts", file=sys.stderr)
    server_url = resolve_server_url(args.vllm_server_url)
    if not server_url:
        raise SystemExit(
            "No model endpoint configured. The auditor is a URL-only client and does "
            "not self-host a model: serve one with reclaim_serve (python -m "
            "reclaim_serve ...) and pass --vllm-server-url, or set $RECLAIM_SERVER_URL "
            "/ $RECLAIM_ENDPOINT_FILE."
        )
    model_id = resolve_served_model(server_url, args.served_model_name)
    print(
        f"Using vLLM server at {server_url} (model={model_id})",
        file=sys.stderr,
    )
    # The auditor streams each round to the telemetry service's Audits page, exactly
    # like a reproduce run. Opt-in: a no-op unless RECLAIM_TELEMETRY_URL and
    # RECLAIM_TELEMETRY_KEY are set.
    audit_sink = None
    try:
        audit_sink = install_audit_sink(AuditSinkConfig.from_env(model_id))
        run_tool_loop(args, papers, prompts, server_url, model_id=model_id)
    finally:
        if audit_sink:
            audit_sink.close()

    print(f"Finished writing {len(prompts)} responses to {args.output}", file=sys.stderr)
    print(f"Finished writing extracted JSONL to {args.extracted_output}", file=sys.stderr)
    return 0


def run_dir_for(runs_dir, arxiv_id: str) -> str:
    return str(runs_dir / arxiv_id) if runs_dir else ""


def filter_papers_by_ids(papers: list, ids_file) -> list:
    wanted = {
        line.strip()
        for line in ids_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    selected = [paper for paper in papers if paper.arxiv_id in wanted]
    missing = wanted - {paper.arxiv_id for paper in selected}
    if missing:
        print(
            f"Warning: {len(missing)} requested id(s) not in dataset: "
            f"{', '.join(sorted(missing)[:10])}",
            file=sys.stderr,
        )
    return selected


if __name__ == "__main__":
    raise SystemExit(main())
