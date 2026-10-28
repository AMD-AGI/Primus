#!/bin/bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Inner container script for Primus cluster RDMA perf tests.
#
# Expects to be launched by run_slurm.sh, which exports:
#   NODE_RANK         0-based per-node index (rank 0 = server, rank 1 = client)
#   NNODES            number of allocated nodes
#   MASTER_ADDR       IB-routable IP of the rank-0 node (reserved; unused here)
#   MASTER_PORT       Reserved rendezvous port (kept for parity)
#   IPADDRS           comma-separated list of per-node IB-routable IPs
#   IBDEVICES         HCA passed to `ib_write_bw -d`
#   BARRIER_PORT      TCP port used by socket_barrier.py [default: 5000]
#   IB_WRITE_BW_PORT  Data port for ib_write_bw [default: 2000]
#   RDMA_TESTS_PATH   Path inside the container where this directory is mounted
#
# Ported from:
#   ROCm/dist-inf-cookbook : cluster-sphere/cluster-rdma-tests/slurm_scripts/
#                            run_rdma_tests.sh
#
# Logs are tee'd into /run_logs (= host LOG_PATH) so they survive the
# container teardown.
###############################################################################

set -euo pipefail

NODE_RANK="${NODE_RANK:-0}"
NNODES="${NNODES:-2}"
IPADDRS="${IPADDRS:-localhost}"
IBDEVICES="${IBDEVICES:-rdma0}"
BARRIER_PORT="${BARRIER_PORT:-5000}"
IB_WRITE_BW_PORT="${IB_WRITE_BW_PORT:-2000}"
RDMA_TESTS_PATH="${RDMA_TESTS_PATH:-/root/rdma-tests-repo}"

LOG_DIR="/run_logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/ib_write_bw_node${NODE_RANK}.log"

HOST_NAME="$(hostname)"
HOST_IP="$(hostname -I | awk '{print $1}')"
SERVER_IP="$(echo "${IPADDRS}" | awk -F',' '{print $1}')"

cat <<EOF | tee -a "${LOG_FILE}"
================================================
 Primus Cluster RDMA Perf Tests (inside container)
================================================
 Host           : ${HOST_NAME} (${HOST_IP})
 NODE_RANK      : ${NODE_RANK}
 NNODES         : ${NNODES}
 IPADDRS        : ${IPADDRS}
 SERVER_IP      : ${SERVER_IP}
 IBDEVICES      : ${IBDEVICES}
 BARRIER_PORT   : ${BARRIER_PORT}
 IB_WRITE_BW_PORT: ${IB_WRITE_BW_PORT}
 RDMA_TESTS_PATH: ${RDMA_TESTS_PATH}
 LOG_FILE       : ${LOG_FILE}
================================================
EOF

# Preflight: make sure perftest is available in the chosen image.
if ! command -v ib_write_bw >/dev/null 2>&1; then
    echo "[ERROR] ib_write_bw not found in PATH inside container." | tee -a "${LOG_FILE}" >&2
    echo "        Use a Docker image with linux-rdma/perftest preinstalled," | tee -a "${LOG_FILE}" >&2
    echo "        or apt-install/build it via EXTRA_DOCKER_ARGS overrides." | tee -a "${LOG_FILE}" >&2
    exit 2
fi

# Cross-node container startup barrier (blocks until all peers are listening).
echo "[${HOST_NAME}] Waiting at the container creation barrier ..." | tee -a "${LOG_FILE}"
python "${RDMA_TESTS_PATH}/socket_barrier.py" \
    --local-ip "${HOST_IP}" \
    --local-port "${BARRIER_PORT}" \
    --enable-port \
    --node-ips "${IPADDRS}" \
    --node-ports "${BARRIER_PORT}" 2>&1 | tee -a "${LOG_FILE}"

if [[ "${NODE_RANK}" -eq 0 ]]; then
    echo "-------------------------------------------------" | tee -a "${LOG_FILE}"
    echo "[${HOST_NAME}:${HOST_IP}] Running ib_write_bw as SERVER" | tee -a "${LOG_FILE}"
    echo "-------------------------------------------------" | tee -a "${LOG_FILE}"

    ib_write_bw -d "${IBDEVICES}" -q 4 -a --report_gbits -F -p "${IB_WRITE_BW_PORT}" \
        2>&1 | tee -a "${LOG_FILE}"
else
    echo "-------------------------------------------------" | tee -a "${LOG_FILE}"
    echo "[${HOST_NAME}:${HOST_IP}] Running ib_write_bw as CLIENT against ${SERVER_IP}" | tee -a "${LOG_FILE}"
    echo "-------------------------------------------------" | tee -a "${LOG_FILE}"

    echo "[${HOST_NAME}] Waiting for server port to open..." | tee -a "${LOG_FILE}"
    sleep 30

    ib_write_bw -d "${IBDEVICES}" -q 4 -a --report_gbits -F "${SERVER_IP}" -p "${IB_WRITE_BW_PORT}" \
        2>&1 | tee -a "${LOG_FILE}"
fi

echo "[${HOST_NAME}] Finished running RDMA tests. Log: ${LOG_FILE}" | tee -a "${LOG_FILE}"
exit 0
