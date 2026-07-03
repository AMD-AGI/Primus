#!/bin/bash
# =============================================================================
# run_odc.sh — portable ODC LB-mini launcher (single-node)
#
# Runs an ODC LB-mini training job. Default P2P backend is `mori`; passing
# `rocshmem` switches to the rocSHMEM host-API backend that ships inside this
# project (odc_rocm_dev/rocshmem_runtime/), so it works on any base image that
# mounts ONLY the project directory — no dependence on the container `/root`
# layer.
#
# usage: run_odc.sh <mori|rocshmem> <pad|nopad> <exp_yaml_relpath> <exp_name> [KEY=VAL ...]
#   pad   -> LB_MINI_SAME_MICRO=1 ; nopad -> 0
#   extra KEY=VAL args are exported verbatim.
#
# Overridable env (all have portable defaults):
#   PRIMUS_ROOT            project root (auto-derived from this script's path)
#   ODC_ROCSHMEM_LIB       full path to librs_host5.so (default: project copy)
#   HF_HOME                HF cache dir (default: /data/hf_cache_ablation)
#   PRIMUS_PACK_CACHE_DIR  packed-sequence cache (default: $HOME/primus_packed)
#   TRITON_CACHE_DIR       triton cache (rocshmem: fresh per-run unless pinned)
#   TRAIN_LOG_DIR          where to write runlog_*.log (default: $HOME/odc_logs)
#   MASTER_PORT            default 29600
#   ROCSHMEM_HEAP_SIZE     symmetric heap RAW BYTES (default 8 GiB)
# =============================================================================
set -u
BACKEND=$1; PAD=$2; EXP_REL=$3; EXPNAME=$4; shift 4

# --- derive project root from this script's location (portable) -------------
# scripts/ -> rocshmem_runtime/ -> odc_rocm_dev/ -> <PRIMUS_ROOT>
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"            # rocshmem_runtime/
ODC_ROCM_DEV="$(cd "$RUNTIME_DIR/.." && pwd)"          # odc_rocm_dev/
PRIMUS_ROOT="${PRIMUS_ROOT:-$(cd "$ODC_ROCM_DEV/.." && pwd)}"
cd "$PRIMUS_ROOT"

export EXP=$EXP_REL
# ODC arm env
export PYTHONPATH="$ODC_ROCM_DEV/odc_early:$ODC_ROCM_DEV"
export ODC_ENABLE=1 ODC_LB_MINI=1 ODC_PHASE=2 MORI_SHMEM_HEAP_SIZE=8G
if [ "$PAD" = "pad" ]; then export LB_MINI_SAME_MICRO=1; else export LB_MINI_SAME_MICRO=0; fi
# public env
export HF_HOME=${HF_HOME:-/data/hf_cache_ablation} DATA_PATH=${DATA_PATH:-/workspace}
export GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo NCCL_IB_DISABLE=1
export LB_MINI_PACKING=kk FUSED_LINEAR_CE=1
export PRIMUS_PACK_CACHE_DIR=${PRIMUS_PACK_CACHE_DIR:-$HOME/primus_packed}
export PRIMUS_EXP_NAME=$EXPNAME
export MASTER_PORT=${MASTER_PORT:-29600}
mkdir -p "$PRIMUS_PACK_CACHE_DIR"
# backend selection
if [ "$BACKEND" = "rocshmem" ]; then
  export ODC_P2P_BACKEND=rocshmem
  # Default to the project-shipped host binding. _rocshmem_backend.py applies
  # the same project-relative default when ODC_ROCSHMEM_LIB is unset, so this
  # export is belt-and-braces for child shells / subprocess inheritance.
  export ODC_ROCSHMEM_LIB=${ODC_ROCSHMEM_LIB:-$RUNTIME_DIR/host_bindings/librs_host5.so}
  export ROCSHMEM_BOOTSTRAP_SOCKET_IFNAME=lo
  # rocSHMEM symmetric heap size, RAW BYTES (the env parser is decimal-only and
  # does NOT accept K/M/G suffixes). 8 GiB matches MORI_SHMEM_HEAP_SIZE.
  export ROCSHMEM_HEAP_SIZE=${ROCSHMEM_HEAP_SIZE:-8589934592}
  # IMPORTANT: the rocshmem device kernels bake per-PE peer deltas as Triton
  # constexpr. Reusing a Triton cache built by a *different* toolchain (or a
  # different launch) was observed to silently load mismatched kernels ->
  # garbage int_p/wait signalling -> NaN grads from iter 1. Always start from a
  # fresh per-run cache for rocshmem unless the caller pins TRITON_CACHE_DIR.
  export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/tcache_rocshmem_$(date +%Y%m%d_%H%M%S)_$$}
else
  export ODC_P2P_BACKEND=mori
  export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/tcache_mori}
fi
# extra KEY=VAL env
for kv in "$@"; do export "$kv"; done

# Unique per-run timestamp so reruns never clobber earlier logs. Override
# TRAIN_LOG_TS to pin a specific stamp (e.g. shared across multinode ranks).
TRAIN_LOG_TS=${TRAIN_LOG_TS:-$(date +%Y%m%d_%H%M%S)}
TRAIN_LOG_DIR=${TRAIN_LOG_DIR:-$HOME/odc_logs}; mkdir -p "$TRAIN_LOG_DIR"
export TRAIN_LOG="$TRAIN_LOG_DIR/runlog_${EXPNAME}_${TRAIN_LOG_TS}.log"
echo "[run_odc] ROOT=$PRIMUS_ROOT BACKEND=$BACKEND PAD=$PAD P2P=$ODC_P2P_BACKEND SAME_MICRO=$LB_MINI_SAME_MICRO EXP=$EXP NAME=$EXPNAME TS=$TRAIN_LOG_TS"
echo "[run_odc] LIB=${ODC_ROCSHMEM_LIB:-<mori>} TRITON_CACHE_DIR=$TRITON_CACHE_DIR LOG=$TRAIN_LOG"
bash examples/run_pretrain.sh
echo "[run_odc] DONE exit=$? log=$TRAIN_LOG"
