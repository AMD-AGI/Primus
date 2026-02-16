#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# System hook: enable AINIC environment settings.
#
# Trigger:
#   export USING_AINIC=1
#
# This replaces the old env file:
#   runner/helpers/envs/enable_ainic.sh
#
# Note: hooks must print "env.VAR=VALUE" to persist changes back to the caller.
###############################################################################

set -euo pipefail

if [[ "${USING_AINIC:-0}" != "1" ]]; then
    exit 0
fi

ANP_HOME_DIR="${ANP_HOME_DIR:-/opt/amd-anp}"
RCCL_HOME_DIR="${RCCL_HOME_DIR:-/opt/rccl}"
MPI_HOME_DIR="${MPI_HOME_DIR:-/opt/ompi-4.1.6}"

NCCL_IB_TC="${NCCL_IB_TC:-104}"
NCCL_IB_FIFO_TC="${NCCL_IB_FIFO_TC:-184}"
NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-1}"
NCCL_IB_ROCE_VERSION_NUM="${NCCL_IB_ROCE_VERSION_NUM:-2}"
NCCL_MAX_P2P_CHANNELS="${NCCL_MAX_P2P_CHANNELS:-56}"
NET_OPTIONAL_RECV_COMPLETION="${NET_OPTIONAL_RECV_COMPLETION:-1}"
NCCL_IB_USE_INLINE="${NCCL_IB_USE_INLINE:-1}"
RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING="${RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING:-0}"
NCCL_GDR_FLUSH_DISABLE="${NCCL_GDR_FLUSH_DISABLE:-1}"
NCCL_DMABUF_ENABLE="${NCCL_DMABUF_ENABLE:-0}"
NCCL_IGNORE_CPU_AFFINITY="${NCCL_IGNORE_CPU_AFFINITY:-1}"
NCCL_IB_QPS_PER_CONNECTION="${NCCL_IB_QPS_PER_CONNECTION:-1}"

# LD_LIBRARY_PATH: prepend AINIC/RCCL/MPI paths while preserving existing.
_ld_base="/usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu/libibverbs:${RCCL_HOME_DIR}/build/release:${ANP_HOME_DIR}/build:${MPI_HOME_DIR}/install/lib"
# Need to append AINIC/RCCL/MPI paths to the existing LD_LIBRARY_PATH. Otherwise, 
# JAX MaxText will not find the appropriate ROCm libraries.
LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}${_ld_base}"
LOG_INFO_RANK0 "Using AINIC"
LOG_INFO_RANK0 "RCCL_HOME_DIR: ${RCCL_HOME_DIR}"
LOG_INFO_RANK0 "ANP_HOME_DIR: ${ANP_HOME_DIR}"
LOG_INFO_RANK0 "MPI_HOME_DIR: ${MPI_HOME_DIR}"

echo "env.ANP_HOME_DIR=${ANP_HOME_DIR}"
echo "env.RCCL_HOME_DIR=${RCCL_HOME_DIR}"
echo "env.MPI_HOME_DIR=${MPI_HOME_DIR}"

echo "env.NCCL_IB_TC=${NCCL_IB_TC}"
echo "env.NCCL_IB_FIFO_TC=${NCCL_IB_FIFO_TC}"
echo "env.NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX}"
echo "env.NCCL_IB_ROCE_VERSION_NUM=${NCCL_IB_ROCE_VERSION_NUM}"
echo "env.NCCL_MAX_P2P_CHANNELS=${NCCL_MAX_P2P_CHANNELS}"
echo "env.NET_OPTIONAL_RECV_COMPLETION=${NET_OPTIONAL_RECV_COMPLETION}"
echo "env.NCCL_IB_USE_INLINE=${NCCL_IB_USE_INLINE}"
echo "env.RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING=${RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING}"
echo "env.NCCL_GDR_FLUSH_DISABLE=${NCCL_GDR_FLUSH_DISABLE}"
echo "env.NCCL_DMABUF_ENABLE=${NCCL_DMABUF_ENABLE}"
echo "env.NCCL_IGNORE_CPU_AFFINITY=${NCCL_IGNORE_CPU_AFFINITY}"
echo "env.NCCL_IB_QPS_PER_CONNECTION=${NCCL_IB_QPS_PER_CONNECTION}"
echo "env.LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
