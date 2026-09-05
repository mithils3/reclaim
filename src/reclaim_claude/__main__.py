"""Grade a sweep (or a list of papers) with Claude, and file the verdicts.

The auditor with an Anthropic transport. Everything except the transport
is the existing machinery: the same prompt template + rubric + pinned success bar,
the same run-dir tools, the same ``extracted_response`` finalizer (so the verdict
rows are byte-identical to the vLLM auditor's and the anti-cheat caps live in one
place), the same ``audit_upload`` column mapping, and the same ``audit_sink``
stream to the run viewer's Audits page.

What the native API buys: prompt caching. An audit is a long tool loop over a
frozen prefix (tools, system, rubric, claim, run manifest), so automatic caching
turns nearly every re-read into a 0.1x cache hit.

    PYTHONPATH=src python -m reclaim_claude --batch slurm-123456 \
      --runs-dir $RECLAIM_SCRATCH/agent_runs --upload

Needs ANTHROPIC_API_KEY; --batch and --upload additionally need RECLAIM_TELEMETRY_URL +
RECLAIM_TELEMETRY_KEY (the batch lookup is read-only, the upload is not).
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from reclaim_claude import agent
from reclaim_repro import batch_runs, postgrest
from reclaim_repro.audit_upload import audit_fields
from reclaim_repro.dataset import default_split_path
from reclaim_repro.event_sink import service_key_from_env
from reclaim_vllm.audit.inputs import build_audit_prompt_for_dir, load_audit_rubric
from reclaim_vllm.config.config import AUDIT_PROMPT_FILE, AUDIT_RUBRIC_FILE, CLAIM_PLACEHOLDER
from reclaim_vllm.runtime.audit_sink import AuditSink, SinkConfig
from reclaim_vllm.runtime.mre_records import load_mre_records
from reclaim_vllm.vllm.client import response_row
from reclaim_vllm.vllm.io import append_jsonl_row, extracted_response, truncate_output_file

SPLIT_CLAIMS = ("dev", "validation", "eval", "test")
REQUEST_TIMEOUT = 900.0
WRITE_LOCK = threading.Lock()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="reclaim_claude", description=__doc__)
    select = parser.add_argument_group("what to grade")
    select.add_argument("--batch", help="Grade every run of this sbatch batch, e.g. slurm-123456.")
    select.add_argument("--paper-ids", nargs="+", help="Grade these papers' newest bundles instead.")
    select.add_argument("--paper-ids-file", type=Path, help="File of paper ids, one per line.")
    select.add_argument("--runs-dir", type=Path,
                        default=Path(os.environ.get("RECLAIM_SCRATCH")
                                     or os.environ.get("REPRO_WORK_ROOT")
                                     or "scratch") / "agent_runs",
                        help="Root of the agent run bundles (default: $RECLAIM_SCRATCH/agent_runs).")
    select.add_argument("--split", default="eval", choices=sorted(SPLIT_CLAIMS),
                        help="Which lockfile split the runs came from (picks the claims file).")
    select.add_argument("--claims",
                        help="Claims JSONL override (default: the shipped split file).")
    select.add_argument("--limit", type=int, help="Grade at most this many runs (smoke tests).")
    select.add_argument("--skip-audited", action="store_true",
                        help="Skip runs whose row already carries an audit_score.")
    select.add_argument("--include-running", action="store_true",
                        help="Also grade runs still in progress (their bundles are incomplete).")

    grader = parser.add_argument_group("grader")
    grader.add_argument("--model", default=agent.MODEL)
    grader.add_argument("--effort", default=agent.DEFAULT_EFFORT,
                        choices=("low", "medium", "high", "xhigh", "max"),
                        help="Thinking/effort level for the grader (default: high).")
    grader.add_argument("--tool-rounds", type=int, default=agent.DEFAULT_TOOL_ROUNDS)
    grader.add_argument("--max-tokens", type=int, default=agent.DEFAULT_MAX_TOKENS)
    grader.add_argument("--workers", type=int, default=4, help="Runs graded concurrently.")

    out = parser.add_argument_group("outputs")
    out.add_argument("--output", type=Path, help="Raw responses JSONL.")
    out.add_argument("--extracted-output", type=Path, help="Finalized verdict rows JSONL.")
    out.add_argument("--upload", action="store_true",
                     help="PATCH each verdict onto its repro_runs row (OVERWRITES any prior audit).")
    out.add_argument("--no-stream", action="store_true",
                     help="Don't stream the audit transcript to the Audits page.")
    out.add_argument("--dry-run", action="store_true",
                     help="Resolve runs and print what would be graded, then stop.")
    args = parser.parse_args(argv)
    if not (args.batch or args.paper_ids or args.paper_ids_file):
        parser.error("pick what to grade: --batch, --paper-ids, or --paper-ids-file")
    args.claims = args.claims or str(default_split_path(args.split))
    tag = args.batch or "papers"
    args.output = args.output or Path(f"outputs/claude_audit/audit_{tag}.jsonl")
    args.extracted_output = args.extracted_output or Path(
        f"outputs/claude_audit/audit_{tag}_extracted.jsonl"
    )
    return args


def select_runs(args: argparse.Namespace) -> list[batch_runs.Run]:
    """The runs to grade, each already bound to the bundle it wrote."""
    if args.batch:
        base, key = (os.environ.get("RECLAIM_TELEMETRY_URL") or "").rstrip("/"), service_key_from_env()
        if not base or not key:
            raise SystemExit("--batch needs RECLAIM_TELEMETRY_URL + "
                             "RECLAIM_TELEMETRY_KEY to look up the sweep")
        runs = batch_runs.select_runs(
            batch_runs.fetch_runs(base, key, args.batch),
            include_running=args.include_running,
            skip_audited=args.skip_audited,
        )
    else:
        runs = [
            batch_runs.Run(run_id="", arxiv_id=paper_id, status="?", budget=None, audit_score=None)
            for paper_id in _paper_ids(args)
        ]
    if args.limit is not None:
        runs = runs[: max(0, args.limit)]

    selected: list[batch_runs.Run] = []
    for run in runs:
        run.bundle = (
            batch_runs.bundle_for(args.runs_dir, run.arxiv_id, run.run_id)
            if run.run_id
            else batch_runs.newest_bundle(args.runs_dir, run.arxiv_id)
        )
        if run.bundle is None:
            print(f"  {run.arxiv_id}: no bundle under {args.runs_dir} -- skipped", file=sys.stderr)
            continue
        selected.append(run)
    return selected


def _paper_ids(args: argparse.Namespace) -> list[str]:
    ids = list(args.paper_ids or [])
    if args.paper_ids_file:
        ids += [line.strip() for line in args.paper_ids_file.read_text().splitlines() if line.strip()]
    return ids


def audit_run(
    client: Any,
    run: batch_runs.Run,
    prompt: str,
    args: argparse.Namespace,
    sink_config: SinkConfig | None,
) -> dict[str, Any]:
    """Grade one run: audit, finalize, persist, stream. Returns the verdict row."""
    # One sink per run so each audit links to the run it graded, not to $env.
    sink = None if sink_config is None else AuditSink(
        dataclasses.replace(sink_config, graded_run_id=run.run_id or None)
    )
    on_event = _sink_bridge(sink, run.arxiv_id)
    try:
        result = agent.run_audit(
            client,
            prompt=prompt,
            run_dir=run.bundle,
            model=args.model,
            effort=args.effort,
            max_tokens=args.max_tokens,
            tool_rounds=args.tool_rounds,
            on_event=on_event,
        )
        # Reuse the vLLM auditor's finalizer by handing it a chat-completion-shaped
        # row: same parsing, same deterministic caps, same verdict derivation.
        raw = response_row(
            run.arxiv_id,
            {"choices": [{"index": 0, "message": {"role": "assistant", "content": result.text}}]},
        )
        raw.update(tool_loop=result.tool_loop, model=args.model, run_id=run.run_id,
                   bundle=str(run.bundle), usage=result.usage.as_dict())
        verdict = extracted_response(run.arxiv_id, raw)
        with WRITE_LOCK:
            append_jsonl_row(args.output, raw, truncate=False)
            append_jsonl_row(args.extracted_output, verdict, truncate=False)
        on_event("final", result.tool_loop["tool_rounds_used"],
                 {"message": {"content": result.text}, "verdict": verdict,
                  "exit_reason": result.tool_loop["exit_reason"]})
        if args.upload and run.run_id:
            _upload(run.run_id, verdict, args.model)
        print(f"  {run.arxiv_id}: score={verdict.get('score')} verdict={verdict.get('verdict')} "
              f"(${result.usage.cost:.2f}, {result.usage.cache_read:,} cached tokens read)",
              file=sys.stderr, flush=True)
        return {**verdict, "usage": result.usage}
    finally:
        if sink:
            sink.close()


def attempt_tag(model: str, stamp: str) -> str:
    """Identity for this grading pass: ``<model>-<stamp>``.

    Without it a second grader of the same run inherits the first audit's row --
    same id, so the run row shows the older verdict and every event insert
    collides with rows already at those seq numbers and is dropped. The stamp
    keeps re-grades by the SAME model distinct too.
    """
    return f"{model.rsplit('/', 1)[-1]}-{stamp}"


def _sink_bridge(sink: AuditSink | None, paper_id: str):
    """Forward loop events to the Audits page, and narrate them on stderr.

    The narration is not decoration: an audit spends minutes per round inside one
    request, and a terminal that prints nothing between "grading" and the verdict
    is indistinguishable from a hung process.
    """

    def _emit(kind: str, round_index: int, payload: dict[str, Any]) -> None:
        _narrate(paper_id, kind, round_index, payload)
        if sink is not None:
            sink.on_event(kind, {"custom_id": paper_id, "round_index": round_index}, payload)

    return _emit


def _narrate(paper_id: str, kind: str, round_index: int, payload: dict[str, Any]) -> None:
    if kind == "call_start":
        function = payload["call"]["function"]
        arguments = str(function.get("arguments") or "")
        print(f"  [{paper_id}] round {round_index + 1}: {function['name']} {arguments[:90]}",
              file=sys.stderr, flush=True)
    elif kind == "verdict_turn":
        print(f"  [{paper_id}] writing the verdict after {payload['rounds_used']} round(s) "
              f"-- this turn makes no tool calls, so it is silent",
              file=sys.stderr, flush=True)


def _upload(run_id: str, verdict: dict[str, Any], model: str) -> None:
    base, key = (os.environ.get("RECLAIM_TELEMETRY_URL") or "").rstrip("/"), service_key_from_env()
    if not base or not key:
        print("  (upload skipped: no RECLAIM_TELEMETRY_URL / RECLAIM_TELEMETRY_KEY)",
              file=sys.stderr)
        return
    query = urllib.parse.urlencode({"run_id": f"eq.{run_id}"})
    code, text = postgrest.request(
        f"{base}/rest/v1/repro_runs?{query}", service_key=key, method="PATCH",
        body=audit_fields(verdict, model), prefer="return=minimal", timeout=15.0, max_attempts=3,
    )
    if not code or code >= 300:
        print(f"  upload failed for {run_id}: HTTP {code} {text[:160]}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runs = select_runs(args)
    if not runs:
        print("nothing to grade", file=sys.stderr)
        return 1
    if args.dry_run:
        for run in runs:
            print(f"{run.arxiv_id}\t{run.run_id or '(newest)'}\t{run.bundle}")
        return 0

    template = AUDIT_PROMPT_FILE.read_text(encoding="utf-8")
    if CLAIM_PLACEHOLDER not in template:
        raise SystemExit(f"{AUDIT_PROMPT_FILE} must contain {CLAIM_PLACEHOLDER}.")
    rubric = load_audit_rubric(AUDIT_RUBRIC_FILE)
    claims = load_mre_records(args.claims)
    prompts = {
        run.arxiv_id: build_audit_prompt_for_dir(
            template, rubric, claims.get(run.arxiv_id), run.bundle
        )
        for run in runs
    }
    missing = [run.arxiv_id for run in runs if not claims.get(run.arxiv_id)]
    if missing:
        print(f"warning: no central claim for {len(missing)} paper(s): {', '.join(missing[:5])}",
              file=sys.stderr)

    import anthropic  # imported here so --dry-run works without the SDK installed

    # An explicit ceiling: a wedged request should fail and be retried, not hang
    # a sweep forever on a host with flaky outbound DNS.
    client = anthropic.Anthropic(max_retries=5, timeout=REQUEST_TIMEOUT)
    sink_config = None if args.no_stream else SinkConfig.from_env(args.model)
    if sink_config is not None:
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        sink_config = dataclasses.replace(sink_config, attempt=attempt_tag(args.model, stamp))
        print(f"streaming to the Audits page as <run>-<paper>-{sink_config.attempt}-audit",
              file=sys.stderr)
    truncate_output_file(args.output)
    truncate_output_file(args.extracted_output)
    workers = max(1, min(args.workers, len(runs)))  # one worker per run; never idle threads
    print(f"Grading {len(runs)} run(s) with {args.model} (effort={args.effort}, "
          f"{args.tool_rounds} tool rounds, {workers} concurrent)", file=sys.stderr)

    def grade(run: batch_runs.Run) -> dict[str, Any]:
        # One run's failure (API error, a bundle that explodes a tool) must not
        # abandon the rest of a paid sweep: record it and carry on.
        try:
            return audit_run(client, run, prompts[run.arxiv_id], args, sink_config)
        except Exception as exc:  # noqa: BLE001
            print(f"  {run.arxiv_id}: audit failed -- {type(exc).__name__}: {exc}",
                  file=sys.stderr, flush=True)
            return {"custom_id": run.arxiv_id, "score": None, "error": str(exc),
                    "usage": agent.Usage(model=args.model)}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(grade, runs))
    _summarize(rows, args)
    return 0


def _summarize(rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    scored = [row for row in rows if isinstance(row.get("score"), int)]
    total = agent.Usage(model=args.model)
    for row in rows:
        total.merge(row["usage"])
    if scored:
        mean = sum(row["score"] for row in scored) / len(scored)
        reproduced = sum(1 for row in scored if row.get("reproduced"))
        print(f"\n{len(scored)}/{len(rows)} graded  mean {mean:.2f}/10  reproduced {reproduced}",
              file=sys.stderr)
    cached = total.cache_read + total.cache_write + total.input
    hit_rate = (total.cache_read / cached * 100) if cached else 0.0
    print(f"tokens: {total.input:,} in, {total.output:,} out, {total.cache_write:,} cache write, "
          f"{total.cache_read:,} cache read ({hit_rate:.0f}% of input served from cache)",
          file=sys.stderr)
    print(f"cost: ${total.cost:.2f}   verdicts: {args.extracted_output}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
