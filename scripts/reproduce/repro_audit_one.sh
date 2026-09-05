#!/usr/bin/env bash
# Reproduce one paper, grade that exact run bundle, and upload the verdict onto its row.
# Assumes a reachable model endpoint and the split files under data/splits.
# Usage: scripts/reproduce/repro_audit_one.sh PAPER_ID [SPLIT]
# Env overrides: MODEL AUDIT_MODEL ENDPOINT RUNS_DIR TOOL_ROUNDS BUDGET CLAIMS RUN_ID
set -euo pipefail

PAPER_ID="${1:?usage: repro_audit_one.sh PAPER_ID [SPLIT]}"
SPLIT="${2:-dev}"

MODEL="${MODEL:?served model id of the agent}"
AUDIT_MODEL="${AUDIT_MODEL:-$MODEL}"   # grader; set a different model to keep it independent
ENDPOINT="${ENDPOINT:-https://openrouter.ai/api/v1}"
# <scratch>: a scratch/work root with room for run bundles and per-paper venvs.
RUNS_DIR="${RUNS_DIR:-${REPRO_WORK_ROOT:-<scratch>/reclaim}/agent_runs}"
TOOL_ROUNDS="${TOOL_ROUNDS:-25}"
BUDGET="${BUDGET:-}"   # empty = auto (derive the ceiling from the paper's selection band)

# The auditor grades against each paper's central claim and pinned success bar, read
# from the same split file the agent took its target from.
SPLITS_DIR="${SPLITS_DIR:-data/splits}"
case "$SPLIT" in
  dev|validation) SPLIT_FILE=dev_14.jsonl ;;
  *)              SPLIT_FILE=eval_100.jsonl ;;
esac
CLAIMS="${CLAIMS:-${SPLITS_DIR}/${SPLIT_FILE}}"

# Telemetry upload is optional and off unless both are exported:
#   export RECLAIM_TELEMETRY_URL=https://your-endpoint
#   export RECLAIM_TELEMETRY_KEY=your-key
# Stage 3 patches an existing run row, so stage 1 must have uploaded its telemetry.
if [[ -z "${RECLAIM_TELEMETRY_KEY:-}" ]]; then
  echo "note: RECLAIM_TELEMETRY_KEY is unset. The run will not upload, so stage 3" >&2
  echo "      will have no row to patch. The verdict still lands on disk." >&2
fi

# Resolve the repo root (two levels up) so src/ and data/ resolve wherever this runs.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

# Pin a unique run id so we can find the exact bundle afterward and patch the exact row.
RID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$(od -An -N3 -tx1 /dev/urandom | tr -d ' \n')}"
VERDICTS="audit_${PAPER_ID}_extracted.jsonl"
GRADE_ROOT="$(mktemp -d)"
IDS_FILE="$(mktemp -t "audit_ids_${PAPER_ID}.XXXXXX")"
echo "$PAPER_ID" > "$IDS_FILE"
trap 'rm -rf "$GRADE_ROOT" "$IDS_FILE"' EXIT

echo ">>> [1/3] reproduce $PAPER_ID (run_id=$RID, split=$SPLIT, budget=${BUDGET:-auto}, model=$MODEL)"
repro_args=(--paper-id "$PAPER_ID" --split "$SPLIT" --run-id "$RID"
            --vllm-server-url "$ENDPOINT" --served-model-name "$MODEL")
[[ -n "$BUDGET" ]] && repro_args+=(--budget-h100-hours "$BUDGET")
python -m reclaim_repro "${repro_args[@]}"

# Locate the exact bundle this reproduction wrote (unique by run id, any budget dir).
RUN_DIR="$(find "$RUNS_DIR/$PAPER_ID" -type d -name "$RID" -print -quit 2>/dev/null || true)"
if [[ -z "$RUN_DIR" ]]; then
  echo ">>> reproduction wrote no bundle for $RID (it failed hard); nothing to audit" >&2
  exit 1
fi
echo ">>> reproduced bundle: $RUN_DIR"

# Bind a clean grade root that maps the paper id to THIS bundle, so the auditor grades
# exactly this run rather than every past attempt for the paper.
ln -s "$RUN_DIR" "$GRADE_ROOT/$PAPER_ID"

echo ">>> [2/3] audit $PAPER_ID (grader=$AUDIT_MODEL, tool-rounds=$TOOL_ROUNDS)"
export RECLAIM_GRADED_RUN_ID="$RID"
PYTHONPATH=src python3 src/run_arxiv_prompt_vllm.py \
  --mode audit \
  --vllm-server-url "$ENDPOINT" \
  --served-model-name "$AUDIT_MODEL" \
  --model "$AUDIT_MODEL" \
  --runs-dir "$GRADE_ROOT" \
  --claims "$CLAIMS" \
  --paper-ids-file "$IDS_FILE" \
  --tool-rounds "$TOOL_ROUNDS" \
  --output "audit_${PAPER_ID}.jsonl" \
  --extracted-output "$VERDICTS" \
  --trace-output "audit_${PAPER_ID}_trace.jsonl" \
  --save-round-jsonl

echo ">>> [3/3] upload verdict for $PAPER_ID (run $RID)"
PYTHONPATH=src python3 -m reclaim_repro.audit_upload \
  --verdicts "$VERDICTS" \
  --runs-dir "$RUNS_DIR" \
  --run-id "$RID" \
  --audit-model "$AUDIT_MODEL"

echo ">>> how the auditor graded $PAPER_ID:"
python3 - "$VERDICTS" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
if not rows:
    print("  (auditor wrote no rows)"); raise SystemExit
r = rows[0]
print(f"  score: {r.get('score')}/5   verdict: {r.get('verdict')}   reproduced: {r.get('reproduced')}")
if r.get("reported_score") is not None:
    print(f"  (a high cheat flag capped this from {r['reported_score']}/5)")
flags = r.get("cheat_flags") or []
if flags:
    print("  cheat_flags: " + ", ".join(f"{f.get('kind')}({f.get('severity')})" for f in flags))
rationale = r.get("rationale") or ""
if rationale:
    print("  rationale: " + str(rationale)[:600])
PY
echo ">>> done: $PAPER_ID  (auditor transcript: audit_${PAPER_ID}_trace.jsonl; raw responses: audit_${PAPER_ID}.jsonl)"
