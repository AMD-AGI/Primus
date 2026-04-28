#!/usr/bin/env bash
###############################################################################
# Cluster Sphere — ib_write_bw across two Slurm nodes (no torchrun).
#
# Single srun step: task 0 runs ib_write_bw server, task 1 waits then client.
# Set SERVER_RDMA_IP to the **first allocated node’s** address on the RDMA/RoCE
# network (same node that runs the server — from Pipeline A or `ip` on the HCA).
# Requires a 2-node allocation: salloc -N 2 -n 2 -t ...  (adjust -p/-A/-t per site.)
#
#   export PRIMUS_ROOT=/path/to/Primus
#   export SERVER_RDMA_IP=10.x.x.x
#   ./cluster_sphere_ib_write_bw_two_node.sh
#
# Optional: IB_DEVICE (e.g. mlx5_0), PORT / PRIMUS_IB_WRITE_BW_PORT, CLIENT_DELAY
# (seconds before client connects; default 15 in verbs-pair).
###############################################################################
set -euo pipefail

if [[ -z "${SLURM_JOB_NODELIST:-}" ]]; then
  echo "error: SLURM_JOB_NODELIST not set — run inside a Slurm allocation with >= 2 nodes." >&2
  exit 1
fi
if [[ -z "${SERVER_RDMA_IP:-}" ]]; then
  echo "error: export SERVER_RDMA_IP=<first_node_rdma_ip>" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRIMUS_ROOT="${PRIMUS_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
export PYTHONPATH="${PRIMUS_ROOT}${PYTHONPATH:+:${PYTHONPATH:-}}"

mapfile -t HOSTS < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
if [[ -z "${HOSTS[1]:-}" ]]; then
  echo "error: need at least 2 nodes in allocation." >&2
  exit 1
fi

PORT="${PORT:-${PRIMUS_IB_WRITE_BW_PORT:-2000}}"
PAIR_OPTS=(--port "${PORT}")
if [[ -n "${IB_DEVICE:-}" ]]; then
  PAIR_OPTS+=(--device "${IB_DEVICE}")
fi
if [[ -n "${CLIENT_DELAY:-}" ]]; then
  PAIR_OPTS+=(--client-delay "${CLIENT_DELAY}")
fi

echo "verbs-pair: server on ${HOSTS[0]}, client on ${HOSTS[1]}; SERVER_RDMA_IP=${SERVER_RDMA_IP}:${PORT}" >&2

exec srun -N 2 -n 2 env PYTHONPATH="${PYTHONPATH}" SERVER_RDMA_IP="${SERVER_RDMA_IP}" \
  python3 -m primus.tools.preflight.cluster_sphere verbs-pair "${PAIR_OPTS[@]}"
