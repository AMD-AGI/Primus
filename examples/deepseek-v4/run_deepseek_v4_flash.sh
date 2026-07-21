#!/bin/bash

set -euo pipefail

# On the spur cluster (amd-spur), set the launcher / QOS / networking defaults for
# the DeepSeek-V4 Flash multi-node runs. Detected via the `spur` command.
if command -v spur >/dev/null 2>&1; then
    export PRIMUS_LAUNCHER=slurm
    export SLURM_LAUNCH_CMD=sbatch
    export SLURM_PARTITION=amd-spur
    export SLURM_QOS=amd-burst-qos
    export SLURM_ACCOUNT=amd-primus
    # Empty = let the scheduler allocate nodes (skip the hardcoded smci355 default
    # in run_deepseek_v4.sh, whose SLURM_NODELIST uses ${VAR-default} so empty wins).
    # Honor an incoming SLURM_NODELIST so callers can pin to specific good nodes
    # (e.g. to work around a bad node that keeps causing JobLaunchFailure).
    export SLURM_NODELIST="${SLURM_NODELIST:-}"
    # ABI-4 libionic provider .so to swap into the container at launch (fixes ionic
    # RDMA on AINIC images whose bundled libionic only advertises uverbs ABI 1).
    # The tools/patches/fix_libionic_abi4.sh patch reads this; set empty to disable.
    export PRIMUS_LIBIONIC_SRC_ABI4_SO="bak/ainic/libionic-rdmav34.so.host-abi4/libionic.so.1.0.54.0-149.g3304be71"
    export NCCL_DEBUG="${NCCL_DEBUG:-}"
    export GLOO_SOCKET_IFNAME=ens3
    export NCCL_SOCKET_IFNAME=ens3
fi

export PRIMUS_TOTAL_LAYERS=${PRIMUS_TOTAL_LAYERS:-43}
export PRIMUS_NUM_EXPERTS=${PRIMUS_NUM_EXPERTS:-256}
export PRIMUS_MOE_TOPK=${PRIMUS_MOE_TOPK:-6}
export PRIMUS_MOE_FFN_HIDDEN_SIZE=${PRIMUS_MOE_FFN_HIDDEN_SIZE:-2048}
export PRIMUS_INDEX_TOPK=${PRIMUS_INDEX_TOPK:-512}
export PRIMUS_COMPRESS_RATIOS=${PRIMUS_COMPRESS_RATIOS:-'[0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0]'}
export MTP_NUM_LAYERS=${MTP_NUM_LAYERS:-1}

export NNODES=${NNODES:-8}

if [ "$NNODES" -ge 8 ]; then
    export PRIMUS_TP=${PRIMUS_TP:-1}
    export PRIMUS_PP=${PRIMUS_PP:-8}
    export PRIMUS_EP=${PRIMUS_EP:-8}
    export PRIMUS_RECOMPUTE_LAYERS=0
    if [ "$MTP_NUM_LAYERS" -eq 1 ]; then
      export PRIMUS_PP_LAYOUT='Et*4|t*5|(t*6|)*5,t*4mL'
    else
      export PRIMUS_PP_LAYOUT='Et*4|t*5|(t*6|)*5,t*4L'
    fi
elif [ "$NNODES" -eq 4 ]; then
    export PRIMUS_TP=${PRIMUS_TP:-1}
    export PRIMUS_PP=${PRIMUS_PP:-4}
    export PRIMUS_EP=${PRIMUS_EP:-8}
    export PRIMUS_RECOMPUTE_LAYERS=3
    if [ "$MTP_NUM_LAYERS" -eq 1 ]; then
      export PRIMUS_PP_LAYOUT='Et*10|t*11|t*11|t*11mL'
    else
      export PRIMUS_PP_LAYOUT='Et*10|t*11|t*11|t*11L'
    fi
fi

export MBS=${MBS:-1}
export GBS=${GBS:-$((64 * NNODES * MBS))}
export TRAIN_ITERS=${TRAIN_ITERS:-10}

export PRIMUS_SEQ_LENGTH=${PRIMUS_SEQ_LENGTH:-4096}
export PRIMUS_MAX_POSITION_EMBEDDINGS=${PRIMUS_MAX_POSITION_EMBEDDINGS:-${PRIMUS_SEQ_LENGTH}}

export USE_V4_FP8_INDEXER=${USE_V4_FP8_INDEXER:-True}
export USE_V4_ATTENTION_BACKEND=${USE_V4_ATTENTION_BACKEND:-triton_v2}
export USE_V4_CSA_ATTENTION_BACKEND=${USE_V4_CSA_ATTENTION_BACKEND:-triton_v2}
export USE_TURBO_DEEPEP=${USE_TURBO_DEEPEP:-True}
export TURBO_USE_GROUPED_MLP=${TURBO_USE_GROUPED_MLP:-True}
export USE_V4_COMPILED_SINKHORN=${USE_V4_COMPILED_SINKHORN:-True}
export USE_TURBO_ATTENTION=${USE_TURBO_ATTENTION:-False}
export PRIMUS_V4_ATTN_BWD_USE_SPLIT=${PRIMUS_V4_ATTN_BWD_USE_SPLIT:-1}
export PRIMUS_V4_CSA_BWD_SEGREDUCE=${PRIMUS_V4_CSA_BWD_SEGREDUCE:-1}
export PRIMUS_STACK_GROUPED_WEIGHT_TRITON=${PRIMUS_STACK_GROUPED_WEIGHT_TRITON:-1}
export PRIMUS_ROPE_TRITON=${PRIMUS_ROPE_TRITON:-1}
export PRIMUS_SINKHORN_TRITON=${PRIMUS_SINKHORN_TRITON:-1}
export PRIMUS_HC_TRITON=${PRIMUS_HC_TRITON:-1}
export PRIMUS_INDEXER_TRITON=${PRIMUS_INDEXER_TRITON:-1}
export PRIMUS_INDEXER_TRITON_FULL=${PRIMUS_INDEXER_TRITON_FULL:-0}
export PRIMUS_V4_ROUTER_TRITON=${PRIMUS_V4_ROUTER_TRITON:-1}
export PROFILE=${PROFILE:-False}
export PRIMUS_EXP_NAME=${PRIMUS_EXP_NAME:-deepseek_v4_flash_proxy_nodes${NNODES}_pp${PRIMUS_PP}_ep${PRIMUS_EP}_seq${PRIMUS_SEQ_LENGTH}}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_deepseek_v4.sh"
