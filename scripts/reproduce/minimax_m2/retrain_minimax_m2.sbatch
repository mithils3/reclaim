#!/bin/bash
# Serve MiniMax-M2.7 (4-bit AWQ) on this job's GPUs, then reproduce and grade every Retrain-tier paper.
# Assumes Slurm with GPU nodes, Apptainer, and a virtualenv holding vLLM and this repo's requirements.
# Retrain-tier sibling of run_minimax_m2.sbatch; see that file for the serve rationale.
# Fill in the bracketed placeholders, then: sbatch scripts/reproduce/minimax_m2/retrain_minimax_m2.sbatch
#SBATCH -J repro_retrain_minimax
# <account>: the Slurm account to charge.
#SBATCH -A <account>
# <partition>: the GPU partition to run on.
#SBATCH -p <partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Node share of the deployment these ran on. Resize for your nodes.
#SBATCH --cpus-per-task=144
#SBATCH --gpus-per-node=2
#SBATCH --gpu-bind=none
#SBATCH --mem=220G
#SBATCH --time=48:00:00
#SBATCH -o slurm-repro-retrain-%j.out
#SBATCH -e slurm-repro-retrain-%j.err
#SBATCH --export=ALL

set -euo pipefail

# Config (override from the environment)
# <home>: the home root holding the checkout.
RECLAIM="${RECLAIM:-<home>/reclaim}"
# <venv>: the virtualenv root; the script sources <venv>/bin/activate.
VENV="${VENV:-<venv>}"
# your site's batch walltime; matches the #SBATCH --time above
WALL_HOURS="${WALL_HOURS:-48}"
MODEL="${MODEL:-cyankiwi/MiniMax-M2.7-AWQ-4bit}"   # 4-bit AWQ W4A16, fits 2xGH200
SERVED_NAME="${SERVED_NAME:-MiniMaxAI/MiniMax-M2.7}"
AUDIT_MODEL="${AUDIT_MODEL:-${SERVED_NAME}}"  # grader; this job serves one model, so it defaults to it
PORT="${PORT:-8000}"
SPLIT="${SPLIT:-eval}"                    # eval -> the frozen 100-paper benchmark
# stored spelling of the Run|Retrain|Reimplement tier in the split files
TIER="${TIER:-Medium}"   # only reproduce+grade rows of this tier
SPLITS_DIR="${SPLITS_DIR:-${RECLAIM}/data/splits}"   # frozen split files shipped with the repo
BUDGET_H100_HOURS="${BUDGET_H100_HOURS:-}"   # empty -> per-paper budget from the selection band
TOOL_ROUNDS="${TOOL_ROUNDS:-300}"        # reproduction agent tool rounds
AUDIT_TOOL_ROUNDS="${AUDIT_TOOL_ROUNDS:-40}"  # auditor's run-dir exploration rounds
MAX_PARALLEL="${MAX_PARALLEL:-10}"       # papers in flight at once; each takes out one GPU allocation
RESUME="${RESUME:-1}"                    # 1 -> skip papers that already have a verdict on disk

# Audit against the same lockfile row, and so the same claim, the agent targeted.
case "${SPLIT}" in
  dev|validation) SPLIT_FILE="dev_14.jsonl" ;;
  *)              SPLIT_FILE="eval_100.jsonl" ;;
esac
LOCKFILE="${LOCKFILE:-${SPLITS_DIR}/${SPLIT_FILE}}"
CLAIMS="${CLAIMS:-${LOCKFILE}}"

# Keep the agent's heavy I/O (run bundles, per-paper venvs) off the home filesystem.
# <scratch>: a scratch/work root with room for run bundles.
export REPRO_WORK_ROOT="${REPRO_WORK_ROOT:-<scratch>/reclaim}"
# <hf-cache>: the Hugging Face cache directory.
export HF_HOME="${HF_HOME:-<hf-cache>}"
# The agent runs inside the Apptainer sandbox. Send only its in-container Hugging Face
# pulls to node-local /tmp, which clears per job, so the served model and the
# orchestrator keep the persistent cache above.
export APPTAINERENV_HF_HOME="${APPTAINERENV_HF_HOME:-/tmp/hf_cache}"
RUNS_DIR="${RUNS_DIR:-${REPRO_WORK_ROOT}/agent_runs}"
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARNING: HF_TOKEN is unset; downloading a paper bundle from the Hub may fail." >&2
fi

# Every GPU step the agent takes out allocates on the same account and partition as
# this job, and runs inside the image scripts/cluster/build_cuda_sandbox.sh builds.
export RECLAIM_SLURM_ACCOUNT="${RECLAIM_SLURM_ACCOUNT:-<account>}"
export RECLAIM_SLURM_PARTITION="${RECLAIM_SLURM_PARTITION:-<partition>}"
# <sif>: the Apptainer image every agent shell step runs inside.
export RECLAIM_APPTAINER_IMAGE="${RECLAIM_APPTAINER_IMAGE:-<sif>}"

# Telemetry upload is optional and off unless both are exported:
#   export RECLAIM_TELEMETRY_URL=https://your-endpoint
#   export RECLAIM_TELEMETRY_KEY=your-key
# The run sink writes the run row, then audit_upload patches the verdict onto it.
if [[ -z "${RECLAIM_TELEMETRY_KEY:-}" ]]; then
  echo "note: RECLAIM_TELEMETRY_KEY is unset; runs and verdicts stay on disk only." >&2
fi

OUTER_JOB="${SLURM_JOB_ID:-local}"
# One batch id per launch groups this sweep's runs. The REPRO_* vars survive the
# SLURM_* env scrub below, so override them to join an earlier batch.
export REPRO_BATCH_ID="${REPRO_BATCH_ID:-slurm-${OUTER_JOB}}"
export REPRO_BATCH_LABEL="${REPRO_BATCH_LABEL:-${SLURM_JOB_NAME:-repro_sweep} #${OUTER_JOB}}"
SWEEP_DIR="${SWEEP_DIR:-${REPRO_WORK_ROOT}/retrain_sweeps/${OUTER_JOB}}"
ENDPOINT_FILE="${ENDPOINT_FILE:-${SWEEP_DIR}/server_endpoint.${OUTER_JOB}.json}"
IDS_FILE="${SWEEP_DIR}/retrain_ids.txt"
mkdir -p "${SWEEP_DIR}"

# The epoch at which Slurm kills this job, surfaced to the agent every tool round.
# squeue %e is the scheduler's own end time; fall back to the #SBATCH --time cap.
if end_ts="$(squeue -h -j "${SLURM_JOB_ID}" -o '%e' 2>/dev/null)" \
   && [ -n "${end_ts}" ] \
   && REPRO_WALL_DEADLINE_EPOCH="$(date -d "${end_ts}" +%s 2>/dev/null)"; then
  :
else
  REPRO_WALL_DEADLINE_EPOCH="$(( $(date +%s) + WALL_HOURS * 3600 ))"
fi
export REPRO_WALL_DEADLINE_EPOCH

# Serve environment, shared with scripts/serve/*.
export TORCHINDUCTOR_CACHE_DIR="<scratch>/.cache/torchinductor"
export TRITON_CACHE_DIR="<scratch>/.cache/triton"
export VLLM_CACHE_ROOT="<scratch>/.cache/vllm"
export SAFETENSORS_FAST_GPU=1
export CUDA_MODULE_LOADING=LAZY
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONFAULTHANDLER=1
export MASTER_ADDR=127.0.0.1
export VLLM_HOST_IP=127.0.0.1
export NCCL_CUMEM_ENABLE=0
export NCCL_CUMEM_HOST_ENABLE=0
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_NET_PLUGIN="${NCCL_NET_PLUGIN_OVERRIDE:-none}"
export OMP_NUM_THREADS=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1200}"

ulimit -l unlimited || true
ulimit -s unlimited || true

module load "<module>"
source "${VENV}/bin/activate"
export PYTHONPATH="${RECLAIM}/src:${PYTHONPATH:-}"
cd "${RECLAIM}"

echo "host=$(hostname)  job=${OUTER_JOB}  endpoint_file=${ENDPOINT_FILE}"
# Empty budget means the CLI derives the paper ceiling from its selection band.
if [[ -n "${BUDGET_H100_HOURS}" ]]; then
  BUDGET_ARG=(--budget-h100-hours "${BUDGET_H100_HOURS}")
  BUDGET_LABEL="${BUDGET_H100_HOURS}h (flat)"
else
  BUDGET_ARG=()
  BUDGET_LABEL="per-band"
fi
echo "sweep_dir=${SWEEP_DIR}  split=${SPLIT}  tier=${TIER}  budget=${BUDGET_LABEL}  rounds=${TOOL_ROUNDS}  parallel=${MAX_PARALLEL}"
echo "runs_dir=${RUNS_DIR}  resume=${RESUME}  audit_model=${AUDIT_MODEL}  audit_rounds=${AUDIT_TOOL_ROUNDS}"
echo "claims=${CLAIMS}"
nvidia-smi topo -m || true

# Step 1: serve the model in the background, then wait for /health.
# Never inherit an endpoint from a requeue of this same job id.
rm -f "${ENDPOINT_FILE}"
python -m reclaim_serve \
  --model "${MODEL}" \
  --served-model-name "${SERVED_NAME}" \
  --port "${PORT}" \
  --tensor-parallel-size 2 \
  --max-model-len 196608 \
  --distributed-executor-backend mp \
  --compilation-config '{"mode":3,"pass_config":{"fuse_minimax_qk_norm":true}}' \
  --endpoint-file "${ENDPOINT_FILE}" \
  -- --kv-offloading-backend native --kv-offloading-size 64 &   # 64GB CPU RAM as KV-cache overflow (V1 replaces --swap-space)
SERVER_PID=$!
trap 'kill "${SERVER_PID}" 2>/dev/null || true' EXIT

# reclaim_serve writes ENDPOINT_FILE once /health is green.
for _ in $(seq 1 720); do
  [[ -f "${ENDPOINT_FILE}" ]] && break
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "server exited before publishing an endpoint" >&2; exit 1
  fi
  sleep 5
done
[[ -f "${ENDPOINT_FILE}" ]] || { echo "timed out waiting for ${ENDPOINT_FILE}" >&2; exit 1; }

# A resume points SWEEP_DIR at a predecessor's directory. Refuse an endpoint this job
# did not publish: a dead node's address burns every paper dispatched against it.
BASE_URL="$(python3 - "${ENDPOINT_FILE}" "${OUTER_JOB}" <<'PY'
import json, sys

payload = json.load(open(sys.argv[1]))
owner = str(payload.get("job_id", ""))
if owner != sys.argv[2]:
    raise SystemExit(f"endpoint {sys.argv[1]} was published by job {owner!r}, not {sys.argv[2]!r}")
print(payload["base_url"])
PY
)" || exit 1
curl -fsS "${BASE_URL}/health" >/dev/null || { echo "server unreachable at ${BASE_URL}" >&2; exit 1; }
echo "server up at ${BASE_URL}"

# Host telemetry, optional and off with the upload: one beacon samples this node and
# tails this job's stdout. Per-run beacons ride each on-demand GPU allocation.
BEACON_PID=""
if [[ -n "${RECLAIM_TELEMETRY_KEY:-}" ]]; then
  LOGFILE="$(scontrol show job "${OUTER_JOB}" 2>/dev/null | sed -n 's/.*StdOut=\([^ ]*\).*/\1/p' | head -1)"
  python -m reclaim_repro.metrics_beacon --role master --log-file "${LOGFILE:-/dev/null}" --interval 15 &
  BEACON_PID=$!
fi
trap 'kill "${SERVER_PID}" ${BEACON_PID:+"${BEACON_PID}"} 2>/dev/null || true' EXIT

# The orchestrator + auditor auto-discover the URL + served model from this file.
export RECLAIM_ENDPOINT_FILE="${ENDPOINT_FILE}"

# Step 2: enumerate the Retrain tier of the split
mapfile -t PAPER_IDS < <(
  python - "${LOCKFILE}" "${SPLIT}" "${TIER}" <<'PY'
import sys
from reclaim_repro.inputs import (
    DEFAULT_UNBANDED_BUDGET_H100_HOURS,
    band_max_hours,
    load_lockfile_rows,
)
rows = load_lockfile_rows(sys.argv[1], split=sys.argv[2])
want = sys.argv[3]
tier = [(pid, row) for pid, row in rows.items() if (row.get("tier") or "") == want]
# Smallest compute ceiling first: cheap papers land results early, and a
# long-budget paper hitting the sweep's wall clock wastes less of the queue.
tier.sort(key=lambda pr: band_max_hours(pr[1]) or DEFAULT_UNBANDED_BUDGET_H100_HOURS)
for pid, _ in tier:
    print(pid)
PY
)
if [[ "${#PAPER_IDS[@]}" -eq 0 ]]; then
  echo "no ${TIER}-tier papers found in ${LOCKFILE} split=${SPLIT}" >&2; exit 1
fi
printf '%s\n' "${PAPER_IDS[@]}" > "${IDS_FILE}"
echo "split '${SPLIT}' tier '${TIER}' has ${#PAPER_IDS[@]} papers: ${PAPER_IDS[*]}"
echo "ids -> ${IDS_FILE}"

# Step 3: scrub inherited SLURM_* so each run_gpu salloc/srun is clean
# Child srun inherits this allocation's env; scrub it for fresh GPU steps.
for v in $(compgen -v | grep -E '^SLURM_' || true); do unset "$v"; done
export SLURM_EXPORT_ENV=ALL

# Step 4: reproduce + audit every Retrain paper, PAIRED on one bundle
# Per paper: reproduce with a pinned run id, symlink the grade root at that exact
# bundle, audit it, then patch the verdict onto the same run row.
STATUS_FILE="${SWEEP_DIR}/status.tsv"

# The server can be reachable at sweep start and gone later. A paper dispatched at a
# dead address dies in seconds and every path below returns 0, so it would be filed
# as a model failure and never retried. Probe /models, the exact call the worker
# makes, and wait rather than spend the paper. Normalize the /v1 suffix first.
server_ready() {
  curl -fsS --max-time 10 "${BASE_URL%/v1}/v1/models" >/dev/null 2>&1
}

wait_for_server() {
  local tries="${1:-60}"
  for _ in $(seq 1 "${tries}"); do
    server_ready && return 0
    sleep 10
  done
  return 1
}

run_one() {
  local pid="$1"
  local log="${SWEEP_DIR}/repro_${pid}.log"
  local verdicts="${SWEEP_DIR}/audit_${pid}_extracted.jsonl"

  # Resume only unfinished papers after a wall-time cap.
  if [[ "${RESUME}" == "1" && -s "${verdicts}" ]]; then
    printf '%s\t%s\t%s\t%s\t%s\n' "${pid}" "-" "resume-skip" "-" "-" >> "${STATUS_FILE}"
    echo "=== [$(date '+%F %T')] ${pid} already graded -> skip (${verdicts})"
    return 0
  fi

  if ! wait_for_server 60; then
    printf '%s\t%s\t%s\t%s\t%s\n' "${pid}" "-" "server-down" "-" "-" >> "${STATUS_FILE}"
    echo "!!! [$(date '+%F %T')] ${pid} NOT dispatched: server unreachable at ${BASE_URL}"
    return 0
  fi

  local rid="${OUTER_JOB}-${pid}-$(od -An -N3 -tx1 /dev/urandom | tr -d ' \n')"
  echo ">>> [$(date '+%F %T')] start ${pid} (run_id=${rid})"

  # (1) Reproduce this paper; the server URL comes from RECLAIM_ENDPOINT_FILE.
  python -m reclaim_repro \
    --paper-id "${pid}" \
    --split "${SPLIT}" \
    --lockfile "${LOCKFILE}" \
    --run-id "${rid}" \
    "${BUDGET_ARG[@]}" \
    --tool-rounds "${TOOL_ROUNDS}" \
    --served-model-name "${SERVED_NAME}" \
    --output "${SWEEP_DIR}/reproduce_${pid}.jsonl" \
    --save-round-jsonl \
    >"${log}" 2>&1
  local rc=$?

  # (2) Locate this run's bundle, unique by run id across budget dirs.
  local run_dir
  run_dir="$(find "${RUNS_DIR}/${pid}" -type d -name "${rid}" -print -quit 2>/dev/null || true)"
  if [[ -z "${run_dir}" ]]; then
    printf '%s\t%s\t%s\t%s\t%s\n' "${pid}" "${rc}" "no-bundle" "-" "${rid}" >> "${STATUS_FILE}"
    echo "<<< [$(date '+%F %T')] ${pid} repro_rc=${rc} wrote NO bundle -> skip audit (log: ${log})"
    return 0
  fi

  # (3) Make the auditor's runs-dir contract resolve to this bundle only.
  local grade_root ids_file
  grade_root="$(mktemp -d "${SWEEP_DIR}/grade_${pid}.XXXXXX")"
  ln -s "${run_dir}" "${grade_root}/${pid}"
  ids_file="${grade_root}/ids.txt"
  printf '%s\n' "${pid}" > "${ids_file}"

  # (4) Audit exactly this bundle and link the live audit to the graded run.
  RECLAIM_GRADED_RUN_ID="${rid}" \
  python3 src/run_arxiv_prompt_vllm.py \
    --mode audit \
    --vllm-server-url "${BASE_URL}" \
    --served-model-name "${AUDIT_MODEL}" \
    --model "${AUDIT_MODEL}" \
    --runs-dir "${grade_root}" \
    --claims "${CLAIMS}" \
    --paper-ids-file "${ids_file}" \
    --tool-rounds "${AUDIT_TOOL_ROUNDS}" \
    --max-input-tokens 128000 \
    --max-tokens 32768 \
    --max-model-len 196608 \
    --output "${SWEEP_DIR}/audit_${pid}.jsonl" \
    --extracted-output "${verdicts}" \
    --trace-output "${SWEEP_DIR}/audit_${pid}_trace.jsonl" \
    --save-round-jsonl \
    >>"${log}" 2>&1
  local arc=$?
  rm -rf "${grade_root}"

  # (5) Upload only successful, nonempty audit verdicts to the pinned run_id.
  local uploaded="off"
  if [[ "${arc}" != "0" ]]; then
    uploaded="audit-fail"
  elif [[ ! -s "${verdicts}" ]]; then
    uploaded="no-verdict"
  elif [[ -n "${RECLAIM_TELEMETRY_KEY:-}" ]]; then
    if python3 -m reclaim_repro.audit_upload \
         --verdicts "${verdicts}" \
         --runs-dir "${RUNS_DIR}" \
         --run-id "${rid}" \
         --audit-model "${AUDIT_MODEL}" >>"${log}" 2>&1; then
      uploaded="ok"
    else
      uploaded="upload-fail"
    fi
  fi

  printf '%s\t%s\t%s\t%s\t%s\n' "${pid}" "${rc}" "${arc}" "${uploaded}" "${rid}" >> "${STATUS_FILE}"
  echo "<<< [$(date '+%F %T')] done ${pid} repro_rc=${rc} audit_rc=${arc} upload=${uploaded} (bundle: ${run_dir})"
  return 0
}

: > "${STATUS_FILE}"
set +e   # one bad paper must not abort the sweep
# Track only per-paper PIDs; the server is also a background child.
run_pids=()
inflight=0
for pid in "${PAPER_IDS[@]}"; do
  run_one "${pid}" &
  run_pids+=("$!")
  inflight=$((inflight + 1))
  if (( inflight >= MAX_PARALLEL )); then
    wait -n   # next paper to finish (the server never exits, so it's never this)
    inflight=$((inflight - 1))
  fi
done
wait "${run_pids[@]}" 2>/dev/null || true   # only the papers, not the server
set -e
echo "===== ${TIER} paired reproduce+audit complete (job ${OUTER_JOB}) ====="
repro_ok=$(awk -F'\t' '$2=="0"' "${STATUS_FILE}" | wc -l)
audit_ok=$(awk -F'\t' '$3=="0"' "${STATUS_FILE}" | wc -l)
upload_ok=$(awk -F'\t' '$4=="ok"' "${STATUS_FILE}" | wc -l)
total=$(wc -l < "${STATUS_FILE}")
harness_fail=$(awk -F'\t' '$3=="no-bundle" || $3=="server-down"' "${STATUS_FILE}" | wc -l)
if (( harness_fail > 0 )); then
  echo "WARNING: ${harness_fail} paper(s) failed WITHOUT producing a bundle."
  echo "         These are harness failures rather than model failures: they wrote no"
  echo "         verdict, so RESUME=1 re-picks them:"
  awk -F'\t' '$3=="no-bundle" || $3=="server-down" {print "           " $1 "  (" $3 ")"}' "${STATUS_FILE}"
fi
echo "reproduced ok: ${repro_ok}/${total}   audited ok: ${audit_ok}/${total}   uploaded ok: ${upload_ok}/${total}   (per-paper rc in ${STATUS_FILE})"
sort "${STATUS_FILE}"

# Summary
echo "===== ${TIER} sweep complete (job ${OUTER_JOB}) ====="
echo "run bundles : ${RUNS_DIR}/<arxiv_id>/<budget_hours>h/<run_id>/"
echo "reproduce   : ${SWEEP_DIR}/reproduce_<arxiv_id>.jsonl  (status: ${STATUS_FILE})"
echo "grades      : ${SWEEP_DIR}/audit_<arxiv_id>_extracted.jsonl  (score/verdict/reproduced per paper)"
echo "audit trace : ${SWEEP_DIR}/audit_<arxiv_id>_trace.jsonl"
