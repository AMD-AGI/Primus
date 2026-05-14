#!/bin/bash
###############################################################################
# DeepSeek-V4 Flash perf PROXY runner (latest config: plan-5 P32 final).
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
#                                                set PRIMUS_SEQ_LENGTH=2048
#                                                / 1024 / 512 on the
#                                                command line if OOM)
#
# Plan-5 perf knobs default ON:
#   - USE_V4_TRITON_ATTENTION       (cr ∈ {0, 128} -> Primus Triton kernel)
#   - USE_V4_TRITON_CSA_ATTENTION   (cr == 4       -> Primus Triton CSA)
#   - USE_TURBO_DEEPEP              (PrimusTurboDeepEPTokenDispatcher)
#   - TURBO_USE_GROUPED_MLP         (Turbo grouped-GEMM MoE expert path)
#   - USE_V4_COMPILED_SINKHORN      (P29: torch.compile-fused Sinkhorn,
#                                    kills the 7.6 s aten::sum fp32 reduce
#                                    that dominated the P28 baseline)
#
# Plan-5 P32 final attention-kernel knobs (also default ON in code; surfaced
# here for visibility / easy A/B):
#   - PRIMUS_V4_ATTN_BWD_USE_SPLIT  (atomic-free split V4 attention BWD:
#                                    dQ kernel + dK/dV kernel, each writes
#                                    its own disjoint tiles via tl.store
#                                    instead of atomic_add on a shared buf)
#   - PRIMUS_V4_CSA_BWD_SEGREDUCE   (atomic-free CSA pool BWD via per-visit
#                                    partial buffer + sorted inverse-index
#                                    segmented reduction into dpool)
#
# These two relied on the **plan-5 P32 dual-RoPE bf16 cast fix** in
# `apply_interleaved_partial_rope` (`primus/backends/megatron/core/transformer/
# dual_rope.py`) to actually win in the proxy: pre-fix, cos/sin from
# `position_ids.float() * inv_freq` was fp32, so `bf16 * fp32 = fp32`
# silently upcast Q / K leaving RoPE — every V4 attention kernel paid 2x
# HBM traffic and ran the slow fp32-specialised Triton binary, inflating
# kernel times 1.8-7x in the proxy and masking the split / segreduce wins.
# The one-line cast of cos/sin to `x.dtype` after the unsqueeze lets the
# microbench-optimal kernels also win end-to-end. See
# `deepseek-v4/develop/progress/p32/p32-summary.md` for the full
# diagnostic walk-through.
#
# USE_TURBO_ATTENTION stays OFF — Turbo would take precedence over the V4
# Triton dense path in `DeepseekV4Attention.forward` (plan-4 P27 dispatch
# precedence: turbo > v4_triton > eager for cr ∈ {0, 128}).
#
# Steady-state perf (P32 final, mi355-gpu-8 / dev_primus_wenx_693,
# iter 10 of 10, ${VAR:-DEFAULT} only):
#
#   iter time       :  603 ms / iter   (vs P28 baseline 8837 ms; 14.64x)
#   TFLOP/s/GPU     :  1134            (vs P28 baseline 77.5)
#   HBM peak / rank :  ~170 GiB
#
# Every override is `${VAR:-DEFAULT}`-guarded, so the caller can flip any
# knob via `PRIMUS_SEQ_LENGTH=2048 ./run_deepseek_v4_flash_proxy.sh` etc.
# without editing the script.
#
# Usage:
#   ./run_deepseek_v4_flash_proxy.sh                                # 10-iter smoke
#   TRAIN_ITERS=20 ./run_deepseek_v4_flash_proxy.sh                 # longer warmup pass
#   PRIMUS_V4_ATTN_BWD_USE_SPLIT=0 ./run_deepseek_v4_flash_proxy.sh # fall back to
#                                                                   #   monolithic V4 BWD
#   PRIMUS_V4_CSA_BWD_SEGREDUCE=0 ./run_deepseek_v4_flash_proxy.sh  # fall back to
#                                                                   #   gather+atomic CSA BWD
#   PRIMUS_V4_DIAG_TIME=1 ./run_deepseek_v4_flash_proxy.sh          # dump per-call
#                                                                   #   cuda.Event timings for
#                                                                   #   v4_attention (rank 0)
#
# Profile is intentionally OFF in this script (this is the SMOKE / perf
# runner, not the trace capture). For chrome-trace capture use
# `deepseek-v4/develop/progress/p32/run_baseline_trace_ep8_p32_final.sh`
# (mirrors the plan-4 P25 / plan-3 P23 profile-script pattern).
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

# ---------- Plan-5 P32 final attention-kernel knobs (split + segreduce) -----
# Both default ON in the kernel code post-RoPE-fix; surface them here so
# the proxy script self-documents the P32 final perf recipe and so a quick
# A/B fallback is a single env-var flip. See header for the full root-cause
# write-up.
export PRIMUS_V4_ATTN_BWD_USE_SPLIT=${PRIMUS_V4_ATTN_BWD_USE_SPLIT:-1}
export PRIMUS_V4_CSA_BWD_SEGREDUCE=${PRIMUS_V4_CSA_BWD_SEGREDUCE:-1}

# ---------- Plan-6 elemwise-fusion knobs (default ON; A/B with =0) ----------
# Each plan-6 phase that wins the EP=8 proxy A/B adds its env knob here as
# default ON, mirroring the plan-5 P32 final precedent above. The kernel
# code already defaults each to "1"; this block makes the runner script
# self-document the recipe and lets users flip individual fusions to "0"
# for A/B without editing source.
#
# P34 — stack_grouped_weight Triton FWD/BWD fusion in
#   PrimusTurboGroupedMLP._stack_grouped_linear_weight.
#   EP=8 proxy A/B win: 580.65 -> 530.85 ms / iter, -49.8 ms (-8.6%);
#   TFLOP/s/GPU 463.2 -> 507.2, +9.5%; lm_loss bit-identical (pure
#   layout transform). Default ON since 29baf151 (2026-05-14).
export PRIMUS_STACK_GROUPED_WEIGHT_TRITON=${PRIMUS_STACK_GROUPED_WEIGHT_TRITON:-1}

# P35 — apply_interleaved_partial_rope Triton FWD/BWD fusion in
#   dual_rope.py. Collapses the 9-op eager chain (slice + reshape +
#   four broadcast muls + stack + reshape + cat) into one Triton
#   kernel that does a single contiguous write with the rotation
#   baked in.
#   EP=8 proxy A/B win: 531.7 -> 526.7 ms / iter, -5.0 ms (-0.94%);
#   TFLOP/s/GPU 507.1 -> 513.3, +1.2%; lm_loss bit-identical (pure
#   analytic rotation). Default ON since landing (2026-05-14).
export PRIMUS_ROPE_TRITON=${PRIMUS_ROPE_TRITON:-1}

# P36 — sinkhorn_normalize Triton FWD/BWD fusion in
#   hyper_connection.py.  Replaces the plan-5 P29 ``torch.compile``
#   cached Sinkhorn body with a hand-rolled Triton kernel that runs
#   the 1 + 2*(n_iters - 1) alternating row/col normalize trajectory
#   in registers per row of the leading axis (V4-Flash uses K=4).
#   Microbench at V4-Flash K=4 (B=1, S=4096):
#     FWD 0.045 ms (vs eager 0.600 ms = 13.4x; vs P29 compiled 0.270
#                   ms = 6.0x)
#     BWD 0.105 ms (vs eager 1.520 ms = 14.5x; vs P29 compiled 0.628
#                   ms = 6.0x)
#   The compiled-region overhead (`Torch-Compiled Region` ~21 ms / 16
#   calls + `CompiledFunctionBackward` ~41 ms / 16 calls) is removed
#   entirely.  Default ON since landing (2026-05-14).
export PRIMUS_SINKHORN_TRITON=${PRIMUS_SINKHORN_TRITON:-1}

# P37 — HyperConnection compute_weights elemwise tail Triton fusion in
#   hyper_connection.HyperMixer.compute_weights.  Fuses the 3 slices +
#   3 fused-multiply-adds + 2 sigmoid + 1 softmax + 2 eps adds (the
#   post-_packed_logits, pre-Sinkhorn chain) into one FWD + one BWD
#   Triton kernel.  Microbench at V4-Flash K=4 (B=1, S=4096):
#     FWD 0.044 ms (vs eager 0.102 ms = 2.34x)
#     BWD 0.276 ms (vs eager 0.405 ms = 1.47x)
#   The matmul inside _packed_logits stays as F.linear; collapse / expand
#   (matmul-adjacent) stay eager too -- they are not net wins as
#   separate Triton kernels.  Default ON since landing (2026-05-14).
export PRIMUS_HC_TRITON=${PRIMUS_HC_TRITON:-1}

# P38 — Indexer.forward scoring Triton fusion.  Descoped (default OFF)
#   because the eager `einsum + relu + mul + sum + causal_mask` chain
#   already maps to a cuBLAS / hipBLASLt batched-matmul that runs at
#   ~28 TFLOP/s on MI355 at V4-Flash widths (B=1, S=4096, P=1024, H=8,
#   Hd=128).  The generic Triton kernel here is FWD-competitive only at
#   small shapes (3.35x FWD at B=2, S=128, P=32) but regresses ~30% at
#   the production V4-Flash shape and BWD regresses ~12x due to cross-
#   tile atomic_add traffic on dq / dk / dw.  Keep the env knob so the
#   kernel stays available for future tuning + small-shape paths.
export PRIMUS_INDEXER_TRITON=${PRIMUS_INDEXER_TRITON:-0}

# P39 — V4 Router post-logits Triton FWD/BWD fusion (shared by topk +
#   hash router).  Descoped to **default OFF** (same precedent as P38).
#   Microbench at V4-Flash widths (N=4096, E=256, K=8) does win on
#   `sqrtsoftplus` (1.56x FWD / 1.22x BWD), but the EP=8 proxy A/B
#   (10 iters each) shows ~534 ms / iter both ways and lm_loss
#   bit-identical -- the per-call savings (~1 ms / iter aggregate)
#   are submerged in dispatch + grouped-MLP variance.  Kernel ships
#   behind the env knob, ready for future tuning.  Set
#   PRIMUS_V4_ROUTER_TRITON=1 to enable.
export PRIMUS_V4_ROUTER_TRITON=${PRIMUS_V4_ROUTER_TRITON:-0}

# ---------- Profile OFF in the proxy smoke runner ---------------------------
# This script is the steady-state perf / smoke runner — kineto profiling
# stays OFF to avoid contaminating the iter timer with profiler-collection
# overhead. For chrome-trace capture use
# `deepseek-v4/develop/progress/p32/run_baseline_trace_ep8_p32_final.sh`.
export PROFILE=${PROFILE:-False}

# ---------- Bookkeeping -----------------------------------------------------
# Distinguish the proxy run output dir from the smoke run output dir so the
# trace-capture script + the smoke run land side-by-side without clobbering
# each other's logs.
export PRIMUS_EXP_NAME=${PRIMUS_EXP_NAME:-deepseek_v4_flash_proxy_pp${PRIMUS_PP}_ep${PRIMUS_EP}_seq${PRIMUS_SEQ_LENGTH}}

# Defer to run_deepseek_v4.sh for the actual training launch — every
# CLI flag and the primus-cli invocation lives there.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_deepseek_v4.sh"
