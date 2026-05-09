#!/bin/bash
# Plan-5 P30 — smoke gate for dense v4_attention SWA K-loop pruning.
#
# Runs the V4-Flash production-shape proxy for 10 iterations with the
# post-P29 baseline knobs still ON.  P30's first optimisation is code-side:
# dense (cr == 0) V4 Triton attention now routes through the kernel's
# in-kernel SWA path and skips K tiles that are outside the sliding window.
#
# Output:
#   output/<team>/<user>/p30_smoke_v4_attention_swa_prune_ep8/log_node_0.txt
#
# Raw logs are gitignored (.gitignore in this dir).
set -euo pipefail
set -x

cd /shared/amdgpu/home/wen_xie_qle/workspace/Primus-deepseek-v4

export TRAIN_ITERS=${TRAIN_ITERS:-10}
export PROFILE=False

export USE_V4_TRITON_ATTENTION=True
export USE_V4_TRITON_CSA_ATTENTION=True
export USE_V4_COMPILED_SINKHORN=True
export USE_TURBO_DEEPEP=True
export TURBO_USE_GROUPED_MLP=True
export USE_TURBO_ATTENTION=False

export PRIMUS_EXP_NAME=p30_smoke_v4_attention_swa_prune_ep8

./run_deepseek_v4_flash_proxy.sh
