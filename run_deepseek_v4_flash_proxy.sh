#!/bin/bash
###############################################################################
# Plan-5 P28 — DeepSeek-V4 Flash perf-baseline PROXY runner.
#
# Wraps `run_deepseek_v4.sh` with a V4-Flash production-shape proxy:
#
#   - num_layers       8                        (vs production 43; PROXY)
#   - hidden_size      4096                     (full V4-Flash; from yaml)
#   - num_heads        64                       (full V4-Flash; from yaml)
#   - head_dim         512                      (full V4-Flash; from yaml)
#   - num_experts      256                      (full V4-Flash; PROXY-friendly
#                                                32 experts/rank at EP=8)
#   - moe_router_topk  6                        (full V4-Flash)
#   - moe_ffn_hidden   2048                     (full V4-Flash)
#   - index_topk       512                      (full V4-Flash CSA top-K)
#   - compress_ratios  [0,0,4,128,4,128,4,0]    (8-layer slice exercising
#                                                every layer kind: 3 cr=0,
#                                                3 cr=4, 2 cr=128)
#   - parallel         TP=1 PP=1 EP=8           (single-node 8 GPU)
#   - seq_length       4096 (default)           (V4 pretrain target;
#                                                CALIBRATE DOWN if OOM —
#                                                set PRIMUS_SEQ_LENGTH=2048
#                                                / 1024 / 512 on the
#                                                command line; the chosen
#                                                value lands in the P28
#                                                bottleneck report)
#
# All four plan-5 perf knobs default ON:
#   - USE_V4_TRITON_ATTENTION       (cr ∈ {0, 128} -> Primus Triton kernel)
#   - USE_V4_TRITON_CSA_ATTENTION   (cr == 4       -> Primus Triton CSA)
#   - USE_TURBO_DEEPEP              (PrimusTurboDeepEPTokenDispatcher)
#   - TURBO_USE_GROUPED_MLP         (Turbo grouped-GEMM MoE expert path)
#
# Plan-5 P29 (RESCOPED) adds a fifth perf knob, also ON by default in
# the proxy after G32 + G33b green:
#   - USE_V4_COMPILED_SINKHORN      (torch.compile-fused HyperMixer
#                                    Sinkhorn-Knopp projection — kills
#                                    the 7.6 s aten::sum fp32 reduce
#                                    that dominated the P28 baseline).
#
# USE_TURBO_ATTENTION stays OFF — Turbo would take precedence over the V4
# Triton dense path in `DeepseekV4Attention.forward` (plan-4 P27 dispatch
# precedence: turbo > v4_triton > eager for cr ∈ {0, 128}).
#
# Every override is `${VAR:-DEFAULT}`-guarded, so the caller can flip any
# knob via `PRIMUS_SEQ_LENGTH=2048 ./run_deepseek_v4_flash_proxy.sh` etc.
# without editing the script.
#
# Usage:
#   ./run_deepseek_v4_flash_proxy.sh                 # 10-iter smoke (G31)
#   TRAIN_ITERS=20 ./run_deepseek_v4_flash_proxy.sh  # longer warmup pass
#   PROFILE=True  ./run_deepseek_v4_flash_proxy.sh   # ineffective —
#       run_deepseek_v4.sh hard-codes --disable_tensorboard True; for
#       chrome-trace capture, use the self-contained
#       `deepseek-v4/develop/progress/p28/run_baseline_trace_ep8.sh`
#       instead (mirrors the plan-4 P25 / plan-3 P23 profile-script
#       pattern).
###############################################################################
set -euo pipefail

# ---------- V4-Flash production widths (8-layer proxy slice) ----------------
export PRIMUS_TOTAL_LAYERS=${PRIMUS_TOTAL_LAYERS:-8}
export PRIMUS_NUM_EXPERTS=${PRIMUS_NUM_EXPERTS:-256}
export PRIMUS_MOE_TOPK=${PRIMUS_MOE_TOPK:-6}
export PRIMUS_MOE_FFN_HIDDEN_SIZE=${PRIMUS_MOE_FFN_HIDDEN_SIZE:-2048}
export PRIMUS_INDEX_TOPK=${PRIMUS_INDEX_TOPK:-512}
# 8-layer slice — every V4 attention layer kind exercised:
#   layer 0 / 1 : cr=0  (dense + SWA + sink)
#   layer 2     : cr=4  (CSA)
#   layer 3     : cr=128 (HCA)
#   layer 4     : cr=4  (CSA)
#   layer 5     : cr=128 (HCA)
#   layer 6     : cr=4  (CSA)
#   layer 7     : cr=0  (dense + SWA + sink)  -- V4-Flash production has
#                                                cr=0 first/last layer
export PRIMUS_COMPRESS_RATIOS=${PRIMUS_COMPRESS_RATIOS:-"[0,0,4,128,4,128,4,0]"}

# ---------- Single-node EP=8 ------------------------------------------------
export PRIMUS_TP=${PRIMUS_TP:-1}
export PRIMUS_PP=${PRIMUS_PP:-1}
export PRIMUS_EP=${PRIMUS_EP:-8}

# DP=8 with TP=1 PP=1 EP=8 on 8 GPUs (EP shards experts within DP group).
# GBS=8, MBS=1 -> 1 microbatch / DP rank / iter.  Profiling-friendly:
# minimises iter-to-iter variance + keeps activation memory bounded.
export MBS=${MBS:-1}
export GBS=${GBS:-8}

# ---------- Production seq length target ------------------------------------
# The CSA wrapper-side gather (plan-4 P26) materialises
#   [B, H, Sq, K_topk, D] = [1, 64, Sq, 512, 512] * 2 bytes per microbatch
# in HBM.  At Sq=4096 that is 64 GiB / microbatch on top of the 256-expert
# MoE state (~12 GiB / rank for 8 layers) + KV cache + activations +
# optimizer state — likely OOMs at MI355X (192 GiB HBM).  The plan-5 P28
# task is to CALIBRATE this value (try 4096 -> 2048 -> 1024 -> 512) and
# document the chosen value in `develop/profile/profile-baseline-ep8-*`.
# Plan-5 P31 (in-kernel `topk_idxs` gather) is the structural fix that
# eventually lets this default reach 4096.
export PRIMUS_SEQ_LENGTH=${PRIMUS_SEQ_LENGTH:-4096}
export PRIMUS_MAX_POSITION_EMBEDDINGS=${PRIMUS_MAX_POSITION_EMBEDDINGS:-${PRIMUS_SEQ_LENGTH}}

# ---------- Plan-5 perf knobs (all five ON) ---------------------------------
export USE_V4_TRITON_ATTENTION=${USE_V4_TRITON_ATTENTION:-True}
export USE_V4_TRITON_CSA_ATTENTION=${USE_V4_TRITON_CSA_ATTENTION:-True}
export USE_TURBO_DEEPEP=${USE_TURBO_DEEPEP:-True}
export TURBO_USE_GROUPED_MLP=${TURBO_USE_GROUPED_MLP:-True}
# Plan-5 P29 (RESCOPED): torch.compile-fused HyperMixer Sinkhorn.  Kills
# the 7.6 s aten::sum fp32 reduce (87.3 % of step time in the P28
# baseline trace).  Default ON in the proxy after G32 + G33b are green.
export USE_V4_COMPILED_SINKHORN=${USE_V4_COMPILED_SINKHORN:-True}

# Turbo attention OFF — would take precedence over V4 Triton dense path
# in DeepseekV4Attention.forward (plan-4 P27 dispatch precedence:
#   turbo > v4_triton > eager       for cr ∈ {0, 128}
#   v4_triton_csa > eager           for cr == 4 ).
export USE_TURBO_ATTENTION=${USE_TURBO_ATTENTION:-False}

# ---------- Bookkeeping -----------------------------------------------------
# Distinguish the proxy run output dir from the smoke run output dir so the
# trace-capture script + the smoke run land side-by-side without clobbering
# each other's logs.
export PRIMUS_EXP_NAME=${PRIMUS_EXP_NAME:-deepseek_v4_flash_proxy_pp${PRIMUS_PP}_ep${PRIMUS_EP}_seq${PRIMUS_SEQ_LENGTH}}

# Defer to run_deepseek_v4.sh for the actual training launch — every
# CLI flag and the primus-cli invocation lives there.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_deepseek_v4.sh"
