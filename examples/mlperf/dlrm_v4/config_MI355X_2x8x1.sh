#!/bin/bash
# MLPerf DLRMv4 (dlrmv4) configuration for MI355X, 2 nodes x 8 GPUs.
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

export DGXSYSTEM=MI355X_2x8x1
export NNODES=2

# MLPerf RCP shape for 2 nodes: global batch 16384 (1024 per GPU), LR 2e-6.
# The first eval follows the MLPerf rule for the global batch (SKIP_EVAL_EPOCH_PCT
# unset), as on one node.
export DENSE_LR=2e-6
export SPARSE_LR=2e-6

# Normal-priority NCCL streams at 2+ nodes, as in the MLPerf submission version (it
# leaves TORCH_NCCL_HIGH_PRIORITY unset). Primus' runner/helpers/envs/base_env.sh sets
# 1 unless it is set here. At 2 and 4 nodes, 1 stalls every training step on
# rocm/primus:v26.9.
#
# What it does: it sets the priority of the stream on which each ProcessGroupNCCL
# (legacy, as in this image, or nccl2) runs its async collectives. Here those are
# DDP's all_reduce and torchrec's all-to-alls. 1 = high (HIP priority -1); 0 = normal
# (torch's default).
#
# Why 1 hurts: the HIP in rocm/primus:v26.9 (ROCm 10.2, HIP 7.17) defaults
# DEBUG_HIP_IGNORE_STREAM_PRIORITY=1. All streams then share one pool of
# GPU_MAX_HW_QUEUES hardware (HSA) queues per GPU, at normal hardware priority. A
# priority -1 stream keeps the queue HIP gave it at creation (torch's first use) for
# the whole run. A priority-0 stream (main, the compute stream, is one) releases its
# queue when idle, if there are more streams than queues (as here). At its next launch
# it takes an unused queue, else the one with the fewest waiting packets and streams.
# The train pipeline's memcpy and data-dist streams are -1 (item (67), see Related).
# With 1 the NCCL stream is pinned too. Traces at 2 and 4 nodes (node 0's 8 ranks)
# show it on the data-dist stream's queue, and main on another. Each step the host
# waits in torchrec's SplitsAllToAllAwaitable.wait() for a few-KB D2H copy (the next
# batch's all-to-all split sizes) on the data-dist stream. On the shared queue that
# copy runs only after the backward's collectives queued ahead of it: 56-87 ms host
# block and 3.8-5.8 ms GPU idle per step.
#
# Why 0: it matches the submission version. Its ROCm 7.2.1 keeps a queue pool per
# priority, so there the NCCL stream never shares a queue with the -1 side streams;
# the copy took at most ~4 ms and the GPU idled under 1 ms per step. On this HIP, 0
# changes placement, not hardware priority: the NCCL stream is no longer pinned and is
# placed by load. That can still mean the data-dist (or main) queue.
# Same-tree traces (these configs, 2 and 4 nodes, node 0's 8 ranks; only this variable
# changed): with 0, main, NCCL and data-dist ran on three different HSA queues in every
# trace, the copy overtook the pending collectives (under 3 ms) and the GPU idled
# 0.15-0.4 ms per step. With 1 the NCCL and data-dist streams shared a queue again
# (56-85 ms host block, 4.4-5.8 ms GPU idle per step).
#
# Evidence (OLCF Lux, Nov 2026; each pair on the same nodes, compared at the last eval
# both runs reached): before these configs, time to target was 1.6% (2 nodes) and 1.5%
# (4 nodes) above the submission version's (3 seeds each); setting 0 on that older tree
# took 3.1-3.3% off (3 jobs). These configs vs the submission version (3 seeds x 2
# pairs per node count, both run orders, all runs MLPerf-compliant): -1.6% at 1 node
# (Primus' 1 stays there), -1.7% at 2 nodes, -2.1% at 4 nodes. Every pair was faster
# (-1.4% to -2.4%; one 1-node pair with a disturbed submission run left out). Against
# the submission campaign's HSA_USERPTR_FOR_PAGED_MEM=0 (on Lux its evals ~8% slower at
# 1 node, ~21% at 2 and 4 nodes): -2.6/-3.9/-4.1%. At 4 nodes ~12% of data windows still
# start slow with 0 (first-batch wait over 90 ms; submission version ~5%); the cause is
# not traced.
# One node keeps Primus' 1 (config_MI355X_1x8x1.sh): rank 0's first-batch wait (see
# Re-check) is 31-39 ms there with 1 (submission version: 39-44 ms); 0 vs 1: yaml note.
#
# Re-check after a change of image (HIP, RCCL, torch, torchrec), GPU_MAX_HW_QUEUES,
# DEBUG_HIP_*, TORCH_DIST_USE_NCCL2 or the trainer's streams, and at other node counts.
# Log: TORCH_NCCL_HIGH_PRIORITY=0 in the runner's env dump. Median of rank 0's
# "[boundary] train_ts=N train first-batch data-wait=" lines (one per data window):
# 27-37 ms with 0, 59-66 ms in the submission version, 106-119 ms with the stall.
# Trace (OUTPUT_TRACE=1; on v26.9 kernel events carry hsa_queue): the D2H
# hipMemcpyAsync in "## SplitsAllToAllAwaitable wait() ##" blocks the host 56-87 ms
# with the stall (its copy behind ncclDevKernel_* on one hsa_queue), under 3 ms with 0.
# Time in the "## splits poll ##" before it is the host waiting ahead of the GPU, not
# the stall.
#
# Related: the yaml's GPU_MAX_HW_QUEUES=4 (Primus' runner default: 2) sizes the shared
# pool; 0 was measured at 2 and 4 nodes with 4 only. DEBUG_HIP_IGNORE_STREAM_PRIORITY=0
# would restore a pool per priority (up to 8 queues per GPU). It is untested in
# training; AMD made 1 the default because high-priority queues contend for the same
# hardware pipe. Item (67) (DLRM_HSTU_FP8_PIPE_STREAM_PRIO_MN) asks for priority-0 side
# streams at 2+ nodes. Its one-time check against main (alloc_pregrow.guard_train_pair)
# fell back to -1 in every 2- and 4-node run on Lux in Nov 2026 ("side-stream priority
# fallback" in the log), as in the submission version.
export TORCH_NCCL_HIGH_PRIORITY=0

MLLOG_CONFIG_FILENAME=$(basename "${BASH_SOURCE[0]}")
export MLLOG_CONFIG_FILENAME
