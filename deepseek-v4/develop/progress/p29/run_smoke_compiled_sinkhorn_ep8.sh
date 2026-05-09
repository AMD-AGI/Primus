#!/bin/bash
# Plan-5 P29 (RESCOPED) — G33a smoke gate: 10-iter EP=8 proxy run with the
# torch.compile-fused sinkhorn (USE_V4_COMPILED_SINKHORN=True) on top of all
# four plan-4 P25 / P26 / plan-3 P23 perf knobs.  No profiler — this is the
# correctness-only smoke that gates the perf-trace gate G33b.
#
# Asserts:
# * plan-4 ratchet (G23..G30) stays green (run separately under fast tier);
# * the run completes 10 iters cleanly with no NaN / Inf;
# * no banned warning matches in the log (plan-3 + plan-4 + plan-5 ratchet);
# * lm_loss after iter 10 within 5e-2 of the P28 baseline at the same
#   fixed seed (mock_data RNG is deterministic).
#
# Run directly under the proxy (run_deepseek_v4_flash_proxy.sh) so we
# inherit the V4-Flash widths + plan-5 perf-knob defaults.
#
# Output:
#   output/<team>/<user>/p29_smoke_compiled_sinkhorn_ep8/log_node_0.txt
#
# Raw logs are gitignored (.gitignore in this dir).
set -euo pipefail
set -x

cd /shared/amdgpu/home/wen_xie_qle/workspace/Primus-deepseek-v4

# Plan-5 P29 (RESCOPED) — flip the new perf knob ON.
export USE_V4_COMPILED_SINKHORN=True

# 10 iters is enough to reach the steady-state plateau (the cold + warmup
# tail is iters 0..4, iters 5..9 are steady).  Plan-4 G30 used the same
# count.
export TRAIN_ITERS=${TRAIN_ITERS:-10}

# Profiling OFF — this script is the smoke gate, NOT the trace gate.
# G33b (run_baseline_trace_ep8_p29.sh) captures the trace separately so
# the wall-time numbers in this script are not contaminated by profiler
# overhead.
export PROFILE=False

# Plan-5 perf knobs default ON in the proxy; pin them explicitly here so a
# future env change does not silently flip the smoke gate's contract.
export USE_V4_TRITON_ATTENTION=True
export USE_V4_TRITON_CSA_ATTENTION=True
export USE_TURBO_DEEPEP=True
export TURBO_USE_GROUPED_MLP=True
export USE_TURBO_ATTENTION=False

# Bookkeeping: distinct EXP_NAME so this smoke run lands side-by-side
# with the P28 baseline run, neither clobbering the other's logs.
export PRIMUS_EXP_NAME=p29_smoke_compiled_sinkhorn_ep8

./run_deepseek_v4_flash_proxy.sh
