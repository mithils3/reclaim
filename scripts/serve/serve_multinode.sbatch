#!/bin/bash
# Serve one model with vLLM across several Slurm nodes (TP within a node, PP across nodes).
# Assumes Slurm with GPU nodes, a shared filesystem, and a virtualenv holding vLLM.
# Rank 0 serves the API and publishes its URL; the other ranks run headless.
# Fill in the bracketed placeholders, then: sbatch --nodes=2 scripts/serve/serve_multinode.sbatch
#SBATCH -J reclaim_serve_mn
# <account>: the Slurm account to charge.
#SBATCH -A <account>
# <partition>: the GPU partition to run on.
#SBATCH -p <partition>
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
# Node share of the deployment these ran on. Resize for your nodes.
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --mem=440G
#SBATCH --time=12:00:00
#SBATCH -o slurm-serve-mn-%j.out
#SBATCH -e slurm-serve-mn-%j.err
#SBATCH --export=ALL

set -euo pipefail

# <home>: the home root holding the checkout.
RECLAIM="${RECLAIM:-<home>/reclaim}"
# <venv>: the virtualenv root; the script sources <venv>/bin/activate.
VENV="${VENV:-<venv>}"
# <scratch>: a scratch/work root every node can read and write.
MODEL="${MODEL:?checkpoint path or HF id}"
SERVED_NAME="${SERVED_NAME:?served model id}"
PORT="${PORT:-8000}"
ENDPOINT_FILE="${ENDPOINT_FILE:-<scratch>/endpoints/server.json}"
IFACE_NAME="${IFACE_NAME:?set the high-speed fabric interface, e.g. ib0}"
TP="${TP:-4}"

# Serve environment (no loopback pin: multi-node needs the fabric)
export TORCHINDUCTOR_CACHE_DIR="<scratch>/.cache/torchinductor"
export TRITON_CACHE_DIR="<scratch>/.cache/triton"
export VLLM_CACHE_ROOT="<scratch>/.cache/vllm"
export SAFETENSORS_FAST_GPU=1
export CUDA_MODULE_LOADING=LAZY
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONFAULTHANDLER=1
export NCCL_CUMEM_ENABLE=0
export NCCL_CUMEM_HOST_ENABLE=0
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_NET_PLUGIN="${NCCL_NET_PLUGIN_OVERRIDE:-none}"
export OMP_NUM_THREADS=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1200}"
# Gloo's TCPStore rendezvous needs one interface; NCCL data should use every fabric
# NIC, so match the whole interface family by prefix.
export GLOO_SOCKET_IFNAME="${IFACE_NAME}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${IFACE_NAME%%[0-9]*}}"
ulimit -l unlimited || true
ulimit -s unlimited || true

# <module>: a module that puts Python 3.11 on PATH; drop this line if the venv is enough.
module load "<module>"
source "${VENV}/bin/activate"
export PYTHONPATH="${RECLAIM}/src:${PYTHONPATH:-}"
cd "${RECLAIM}"

# Discover the head node's fabric IP (the rendezvous master)
mapfile -t NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
NNODES="${#NODES[@]}"
HEAD_NODE="${NODES[0]}"
HEAD_IP="$(
  srun --jobid="${SLURM_JOB_ID}" --nodes=1 --ntasks=1 --nodelist="${HEAD_NODE}" \
    bash -lc "ip -o -4 addr show ${IFACE_NAME} | awk '{split(\$4,a,\"/\"); print a[1]; exit}'"
)"
if [[ -z "${HEAD_IP}" || "${HEAD_IP}" == 127.* ]]; then
  echo "Bad HEAD_IP='${HEAD_IP}' from '${IFACE_NAME}' on ${HEAD_NODE}; check the fabric interface." >&2
  exit 1
fi
echo "NNODES=${NNODES} HEAD_NODE=${HEAD_NODE} HEAD_IP=${HEAD_IP} TP=${TP} PP=${NNODES}"

# One launcher process per node; only rank 0 serves the API & publishes
srun --jobid="${SLURM_JOB_ID}" --nodes="${NNODES}" --ntasks="${NNODES}" \
  --ntasks-per-node=1 --gpus-per-task="${TP}" --cpus-per-task=32 \
  bash -lc '
    set -euo pipefail
    module load "<module>"
    source "'"${VENV}"'/bin/activate"
    export PYTHONPATH="'"${RECLAIM}"'/src:${PYTHONPATH:-}"
    cd "'"${RECLAIM}"'"
    HEADLESS=()
    [[ "${SLURM_PROCID}" != "0" ]] && HEADLESS=(--headless)
    python -m reclaim_serve \
      --model "'"${MODEL}"'" \
      --served-model-name "'"${SERVED_NAME}"'" \
      --port "'"${PORT}"'" \
      --tensor-parallel-size "'"${TP}"'" \
      --pipeline-parallel-size "'"${NNODES}"'" \
      --nnodes "'"${NNODES}"'" \
      --node-rank "${SLURM_PROCID}" \
      --master-addr "'"${HEAD_IP}"'" \
      --advertise-ip "'"${HEAD_IP}"'" \
      --endpoint-file "'"${ENDPOINT_FILE}"'" \
      "${HEADLESS[@]}"
  '
