#!/bin/bash
# MLPerf DLRMv4 (dlrmv4) configuration for MI355X, 4 nodes x 8 GPUs.
# config_MI355X_1x8x1.sh plus what changes with the node count.

# Node rank and rendezvous come from the launcher: values it set before this file
# are kept; otherwise run_and_time.sh / primus-cli derive them from SLURM_* (pass
# MASTER_ADDR where scontrol is not available, e.g. inside a container). The
# 1-node values (NODE_RANK=0, MASTER_ADDR=localhost) must not leak into the run.
_dlrm_node_rank=${NODE_RANK-} _dlrm_master_addr=${MASTER_ADDR-}
# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/config_MI355X_1x8x1.sh"
unset NODE_RANK MASTER_ADDR
[[ -n "$_dlrm_node_rank" ]] && export NODE_RANK="$_dlrm_node_rank"
[[ -n "$_dlrm_master_addr" ]] && export MASTER_ADDR="$_dlrm_master_addr"
unset _dlrm_node_rank _dlrm_master_addr

export DGXSYSTEM=MI355X_4x8x1
export NNODES=4

# MLPerf RCP shape for 4 nodes: global batch 32768 (1024 per GPU), LR 4e-6.
# The first eval follows the MLPerf rule for the global batch (SKIP_EVAL_EPOCH_PCT
# unset), as on one node.
export DENSE_LR=4e-6
export SPARSE_LR=4e-6

# Normal-priority NCCL streams at 2+ nodes, as in the MLPerf submission version (it
# leaves TORCH_NCCL_HIGH_PRIORITY unset). Primus' runner/helpers/envs/base_env.sh sets
# 1 unless it is set here. At 2 and 4 nodes, 1 stalls every training step on
# rocm/primus:v26.9.
# The comment above this setting in config_MI355X_2x8x1.sh covers 2 and 4 nodes: the
# mechanism, how it was checked, how to re-check it and the related settings.
export TORCH_NCCL_HIGH_PRIORITY=0

MLLOG_CONFIG_FILENAME=$(basename "${BASH_SOURCE[0]}")
export MLLOG_CONFIG_FILENAME
