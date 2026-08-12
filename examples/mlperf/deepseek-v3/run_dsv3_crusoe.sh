#!/bin/bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# DeepSeek-V3 pretraining on a Crusoe (Spur) cluster, submitted from a login
# node. Everything model-related comes from the shared recipe
# examples/moe_package/run_deepseek_v3_pretrain_mi355x.sh; this wrapper only
# supplies what is site-specific -- the queue, the container image and the
# interconnect names -- plus two ready-made shapes.
#
# -----------------------------------------------------------------------------
# Small-scale bring-up: 1 / 2 / 4 nodes
# -----------------------------------------------------------------------------
# Start here to prove the environment (queue, image, ionic RDMA, rendezvous)
# before spending 32 nodes on it:
#
#   SMOKE=1 NNODES=1 bash examples/mlperf/deepseek-v3/run_dsv3_crusoe.sh
#   SMOKE=1 NNODES=2 bash examples/mlperf/deepseek-v3/run_dsv3_crusoe.sh
#   SMOKE=1 NNODES=4 bash examples/mlperf/deepseek-v3/run_dsv3_crusoe.sh
#
# All three run the *same* model and parallelism -- 4 layers, TP1/PP1/VPP1/EP8 --
# so going from 1 to 4 nodes only widens data parallelism (DP 8 -> 16 -> 32) and
# nothing else changes. Two things make that work:
#
#   * PP=1 is required, not a simplification. EP is bounded by
#     DP = world/(TP*PP), so EP=8 on a single 8-GPU node is only reachable at
#     PP=1. Keeping PP=1 for 2N/4N is what makes the three runs comparable.
#   * The recipe's GBS default is 128*NNODES, which already scales exactly with
#     DP, so the per-GPU batch and the microbatch count stay fixed.
#
# -----------------------------------------------------------------------------
# Full 32-node run
# -----------------------------------------------------------------------------
# The recipe defaults are already the tuned 32-node shape (61 layers,
# PP16 x VPP2, EP8, MBS 2, GBS 4096):
#
#   bash examples/mlperf/deepseek-v3/run_dsv3_crusoe.sh
#
# -----------------------------------------------------------------------------
# Every export below is `${VAR:-<default>}`, so anything can be overridden:
#
#   NNODES=16 TRAIN_ITERS=50 bash examples/mlperf/deepseek-v3/run_dsv3_crusoe.sh
#   SLURM_EXCLUDE=node-a,node-b bash examples/mlperf/deepseek-v3/run_dsv3_crusoe.sh
#
# Notes on this cluster, in case a job never starts:
#   * The default QOS (amd-primus-qos) sits at its group node limit, so jobs
#     stay PENDING forever with Reason=QOSGrpNodeLimit. amd-burst-qos schedules.
#   * "dispatch confirmation failed: N of M nodes confirmed" is a Spur-side
#     launch failure, not a config problem. Re-submitting picks a different set
#     of nodes; SLURM_EXCLUDE takes a list of ones to avoid.
###############################################################################
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRIMUS_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PRIMUS_ROOT"

RECIPE=examples/moe_package/run_deepseek_v3_pretrain_mi355x.sh
[ -f "$RECIPE" ] || { echo "[dsv3-crusoe] recipe not found: $PRIMUS_ROOT/$RECIPE" >&2; exit 1; }

# --- scheduler -------------------------------------------------------------
# sbatch rather than srun: the job is submitted and returns, and only sbatch
# accepts --exclusive on Spur.
export PRIMUS_LAUNCHER=slurm
export SLURM_LAUNCH_CMD="${SLURM_LAUNCH_CMD:-sbatch}"
export SLURM_PARTITION="${SLURM_PARTITION:-amd-spur}"
export SLURM_QOS="${SLURM_QOS:-amd-burst-qos}"
export SLURM_ACCOUNT="${SLURM_ACCOUNT:-amd-primus}"
export SLURM_EXCLUDE="${SLURM_EXCLUDE:-}"

# --- container -------------------------------------------------------------
export DOCKER_IMAGE="${DOCKER_IMAGE:-docker.io/tasimage/primus:pr-927}"

# Some AINIC images ship a libionic provider that only advertises uverbs ABI 1
# while the host driver exposes ABI 4, which leaves every ionic_* device
# rejected and RDMA falling back to TCP. runner/helpers/patches/
# 10_fix_libionic_abi4.sh swaps in an ABI-4 build when this points at one, and
# self-skips when it is empty. The container only mounts the repository root,
# so the .so has to live under it.
if [ -z "${PRIMUS_LIBIONIC_SRC_ABI4_SO:-}" ]; then
    _libionic=$(ls "$PRIMUS_ROOT"/bak/ainic/libionic*.so* 2>/dev/null | head -1 || true)
    export PRIMUS_LIBIONIC_SRC_ABI4_SO="${_libionic:-}"
fi

# --- interconnect ----------------------------------------------------------
# Front-end NIC is ens3 here; RDMA devices are ionic_0..7. The recipe defaults
# target a different site (ens9np0, ionic_8/9), so both have to be set.
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-ens3}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ens3}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-ionic_0:1,ionic_1:1,ionic_2:1,ionic_3:1,ionic_4:1,ionic_5:1,ionic_6:1,ionic_7:1}"

# --- scale -----------------------------------------------------------------
export NNODES="${NNODES:-32}"

# SMOKE=1: the small 1/2/4-node shape described at the top. Only the model size
# and the parallelism are pinned here; GBS stays on the recipe's 128*NNODES so
# scaling out adds data parallelism and changes nothing else. This runs before
# TRAIN_ITERS gets its default so the smoke default can differ.
if [ "${SMOKE:-0}" = "1" ]; then
    export PRIMUS_TOTAL_LAYERS="${PRIMUS_TOTAL_LAYERS:-4}"
    export PRIMUS_PP="${PRIMUS_PP:-1}"
    export PRIMUS_VPP="${PRIMUS_VPP:-1}"
    export PRIMUS_EP="${PRIMUS_EP:-8}"
    export PRIMUS_RECOMPUTE_LAYERS="${PRIMUS_RECOMPUTE_LAYERS:-0}"
    export TRAIN_ITERS="${TRAIN_ITERS:-10}"
    if [ $((NNODES * 8 / PRIMUS_PP)) -lt "$PRIMUS_EP" ]; then
        echo "[dsv3-crusoe] EP=$PRIMUS_EP needs DP >= EP, but NNODES=$NNODES with PP=$PRIMUS_PP gives DP=$((NNODES * 8 / PRIMUS_PP))" >&2
        exit 1
    fi
fi

export TRAIN_ITERS="${TRAIN_ITERS:-20}"

echo "[dsv3-crusoe] root=$PRIMUS_ROOT"
echo "[dsv3-crusoe] nodes=$NNODES iters=$TRAIN_ITERS image=$DOCKER_IMAGE"
if [ "${SMOKE:-0}" = "1" ]; then
    echo "[dsv3-crusoe] smoke shape: layers=$PRIMUS_TOTAL_LAYERS TP1/PP$PRIMUS_PP/VPP$PRIMUS_VPP/EP$PRIMUS_EP dp=$((NNODES * 8 / PRIMUS_PP))"
fi
echo "[dsv3-crusoe] partition=$SLURM_PARTITION qos=$SLURM_QOS account=$SLURM_ACCOUNT"
echo "[dsv3-crusoe] socket=$NCCL_SOCKET_IFNAME hca=$NCCL_IB_HCA"
if [ -n "${PRIMUS_LIBIONIC_SRC_ABI4_SO:-}" ]; then
    echo "[dsv3-crusoe] libionic abi4=$PRIMUS_LIBIONIC_SRC_ABI4_SO"
else
    echo "[dsv3-crusoe] libionic abi4 not provided; ionic RDMA may fall back to TCP."
    echo "[dsv3-crusoe]   put the .so under $PRIMUS_ROOT/bak/ainic/ or set PRIMUS_LIBIONIC_SRC_ABI4_SO"
fi

exec bash "$RECIPE"
