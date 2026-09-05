#!/bin/bash
# Serve one model with vLLM on the GPUs of a single Slurm node and publish its URL.
# Assumes Slurm with GPU nodes and a virtualenv holding vLLM and this repo's requirements.
# Consumers read the address with: export RECLAIM_ENDPOINT_FILE=$ENDPOINT_FILE
# Fill in the bracketed placeholders below, then: sbatch scripts/serve/serve_gh200.sbatch
#SBATCH -J reclaim_serve
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
#SBATCH --time=24:00:00
#SBATCH -o slurm-serve-%j.out
#SBATCH -e slurm-serve-%j.err
#SBATCH --export=ALL

set -euo pipefail

# Paths (override from the environment)
# <home>: the home root holding the checkout.
RECLAIM="${RECLAIM:-<home>/reclaim}"
# <venv>: the virtualenv root; the script sources <venv>/bin/activate.
VENV="${VENV:-<venv>}"
MODEL="${MODEL:-cyankiwi/MiniMax-M2.7-AWQ-4bit}"   # 4-bit AWQ W4A16, fits two 96 GB GPUs
SERVED_NAME="${SERVED_NAME:-MiniMaxAI/MiniMax-M2.7}"
PORT="${PORT:-8000}"
# <scratch>: a scratch/work root the compute nodes share, so other nodes can read the URL.
ENDPOINT_FILE="${ENDPOINT_FILE:-<scratch>/endpoints/minimax_m2.json}"

# Serve environment (shared with the reproduce sweeps)
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
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,ENV,NET}"
export NCCL_NET_PLUGIN="${NCCL_NET_PLUGIN_OVERRIDE:-none}"
export OMP_NUM_THREADS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_ENABLE_MONITORING="${TORCH_NCCL_ENABLE_MONITORING:-1}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1200}"

ulimit -l unlimited || true
ulimit -s unlimited || true

# <module>: a module that puts Python 3.11 on PATH; drop this line if the venv is enough.
module load "<module>"
source "${VENV}/bin/activate"
export PYTHONPATH="${RECLAIM}/src:${PYTHONPATH:-}"
cd "${RECLAIM}"

if [[ -z "${CXX:-}" || "${CXX}" == "CC" ]]; then
  if command -v g++ >/dev/null 2>&1; then export CXX="$(command -v g++)"; fi
fi
[[ -n "${CXX:-}" ]] && export TORCHINDUCTOR_CXX="${TORCHINDUCTOR_CXX:-${CXX}}"

echo "host=$(hostname)  job=${SLURM_JOB_ID:-?}  endpoint_file=${ENDPOINT_FILE}"
nvidia-smi topo -m || true

# The launcher binds 0.0.0.0, finds a routable fabric IP, and writes $ENDPOINT_FILE
# once /health is green. It blocks until the job ends. Without --max-model-len the
# model profile's native context window is served.
python -m reclaim_serve \
  --model "${MODEL}" \
  --served-model-name "${SERVED_NAME}" \
  --port "${PORT}" \
  --tensor-parallel-size 2 \
  --distributed-executor-backend mp \
  --compilation-config '{"mode":3,"pass_config":{"fuse_minimax_qk_norm":true}}' \
  --endpoint-file "${ENDPOINT_FILE}" \
  -- --kv-offloading-backend native --kv-offloading-size 64   # 64GB CPU RAM of KV-cache overflow
