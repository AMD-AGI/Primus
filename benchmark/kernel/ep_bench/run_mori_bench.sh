#!/bin/bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Inner script (runs inside the MAD large_ep_benchmark Docker container) that
# executes MoRI dispatch/combine microbenchmarks WITHOUT torchrun.
#
# Both MoRI bench scripts use torch.multiprocessing.spawn internally to spawn
# one process per GPU, so this script is invoked exactly once per node.
#
# For the internode bench, MoRI expects:
#   RANK        = node rank (0..NNODES-1)
#   WORLD_SIZE  = number of nodes
#   MASTER_ADDR = rendezvous host (first node)
#   MASTER_PORT = rendezvous port
# These are derived from SLURM_NODEID / SLURM_NNODES / etc. by the outer
# run_slurm.sh and forwarded as NODE_RANK / NNODES / MASTER_ADDR / MASTER_PORT.
#
# Expected environment (set by run_slurm.sh):
#   NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT, GPUS_PER_NODE, IBDEVICES
#
# Output logs are written under /run_logs (host: LOG_DIR).
###############################################################################

set -eo pipefail

MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-2373}"
NODE_RANK="${NODE_RANK:-0}"
NNODES="${NNODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
IBDEVICES="${IBDEVICES:-mlx5_0}"

LOG_DIR="${LOG_DIR_IN_CONTAINER:-/run_logs}"
mkdir -p "${LOG_DIR}"

HOST_NAME="$(hostname)"
HOST_IP="$(hostname -I | awk '{print $1}')"

# Resolve default-route interface for Gloo (TCP) rendezvous.
DEFAULT_IFACE="$(ip route | awk '/^default/ {print $5; exit}')"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${DEFAULT_IFACE}}"

# rocSHMEM / MoRI RDMA env (shared with MAD run_ep_bench.sh defaults).
export ROCSHMEM_USE_IB_HCA="${ROCSHMEM_USE_IB_HCA:-${IBDEVICES}}"
export ROCSHMEM_MAX_NUM_CONTEXTS="${ROCSHMEM_MAX_NUM_CONTEXTS:-144}"

MORI_PATH="${MORI_PATH:-/app/mori}"
if [[ ! -d "${MORI_PATH}" ]]; then
    echo "[ERROR] MoRI not found at ${MORI_PATH}. Set MORI_PATH or use an image that ships MoRI." >&2
    exit 2
fi
export PYTHONPATH="${MORI_PATH}:${PYTHONPATH:-}"

cat <<EOF
================================================
 MoRI EP-Bench (inside container, no torchrun)
================================================
  Host             : ${HOST_NAME} (${HOST_IP})
  NNODES           : ${NNODES}
  NODE_RANK        : ${NODE_RANK}
  GPUS_PER_NODE    : ${GPUS_PER_NODE}
  MASTER_ADDR      : ${MASTER_ADDR}
  MASTER_PORT      : ${MASTER_PORT}
  IBDEVICES        : ${IBDEVICES}
  GLOO_SOCKET_IF   : ${GLOO_SOCKET_IFNAME}
  MORI_PATH        : ${MORI_PATH}
  LOG_DIR          : ${LOG_DIR}
================================================
EOF

cd "${MORI_PATH}"

if [[ "${NNODES}" -eq 1 ]]; then
    echo "[Node ${NODE_RANK}] Running MoRI INTRANODE dispatch/combine benchmark (bf16)..."
    python tests/python/ops/bench_dispatch_combine.py \
        2>&1 | tee "${LOG_DIR}/mori_intranode_bf16.log"

    echo "[Node ${NODE_RANK}] Running MoRI INTRANODE low-latency benchmark (fp8_e4m3_fnuz)..."
    python tests/python/ops/bench_dispatch_combine.py --dtype fp8_e4m3_fnuz \
        2>&1 | tee "${LOG_DIR}/mori_intranode_fp8_ll.log"
else
    # Internode: MoRI internode test reads RANK as node-rank and WORLD_SIZE as
    # number of nodes (it spawns one process per GPU itself), so we just
    # promote NODE_RANK / NNODES and call python directly.
    export RANK="${NODE_RANK}"
    export WORLD_SIZE="${NNODES}"
    export MASTER_ADDR
    export MASTER_PORT
    export GPU_PER_NODE="${GPUS_PER_NODE}"

    INTERNODE_SCRIPT="examples/ops/dispatch_combine/test_dispatch_combine_internode.py"

    export GLOO_SOCKET_IFNAME=$(ip route | grep '^default' | awk '{print $NF}' | head -n 1)

    echo "[Node ${NODE_RANK}] Running MoRI INTERNODE dispatch/combine benchmark (v1, bf16)..."
    torchrun --nnodes=$NNODES \
        --node_rank=$NODE_RANK \
        --nproc_per_node=1 \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        "${INTERNODE_SCRIPT}" --cmd bench \
        2>&1 | tee "${LOG_DIR}/mori_internode_v1_rank${NODE_RANK}.log"

    sleep 10

    echo "[Node ${NODE_RANK}] Running MoRI INTERNODE low-latency benchmark (v1_ll)..."
    torchrun --nnodes=$NNODES \
        --node_rank=$NODE_RANK \
        --nproc_per_node=1 \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        "${INTERNODE_SCRIPT}" --cmd bench --kernel-type v1_ll \
        2>&1 | tee "${LOG_DIR}/mori_internode_v1_ll_rank${NODE_RANK}.log"
fi

echo "[Node ${NODE_RANK}] MoRI EP benchmarks complete. Logs under ${LOG_DIR}/"
exit 0
