#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# Regenerate the git-ignored 4-GPU MaxText configs from the tracked 1-GPU proxy configs.
# Usage: make_local_configs.sh <Primus root> [dest dir]
# Default dest: <Primus root>/examples/maxtext/configs/MI455X/local (ignored by git via `local/`).
set -euo pipefail
PRIMUS=${1:?usage: make_local_configs.sh <Primus root> [dest dir]}
SRC=$PRIMUS/examples/maxtext/configs/MI455X
L=${2:-$SRC/local}
mkdir -p "$L/sweep"

fp8() { awk '{print; if ($1 == "attention:") print "      quantization: \"fp8\""}' "$1" > "$2"; }

# 26B and E2B hardcode remat "full"; make it PRIMUS_REMAT_POLICY-driven like 31B (same default).
for m in 26B 31B e2b; do
  sed -e 's/ici_fsdp_parallelism: 1/ici_fsdp_parallelism: 4/' \
      -e 's/remat_policy: "full"/remat_policy: ${PRIMUS_REMAT_POLICY:full}/' \
    "$SRC/gemma4_${m}-bf16-pretrain_1gpu_proxy.yaml" > "$L/gemma4_${m}-4gpu.yaml"
  fp8 "$L/gemma4_${m}-4gpu.yaml" "$L/gemma4_${m}-fp8-4gpu.yaml"
done

for p in "" -fp8; do
  b=$L/gemma4_31B${p}-4gpu.yaml
  sed -e 's/ici_fsdp_parallelism: 4/ici_fsdp_parallelism: 1/' \
      -e 's/ici_data_parallelism: 1/ici_data_parallelism: 4/' "$b" > "$L/sweep/gemma4_31B-dp4${p}.yaml"
  sed -e 's/ici_fsdp_parallelism: 4/ici_fsdp_parallelism: 2/' \
      -e 's/ici_data_parallelism: 1/ici_data_parallelism: 2/' "$b" > "$L/sweep/gemma4_31B-fsdp2-dp2${p}.yaml"
  sed -e 's/ici_fsdp_parallelism: 4/ici_fsdp_parallelism: 1/' \
      -e 's/ici_tensor_sequence_parallelism: 1/ici_tensor_sequence_parallelism: 4/' "$b" > "$L/sweep/gemma4_31B-tp4${p}.yaml"
done

for p in "" -fp8; do
  sed -e 's/ici_fsdp_parallelism: 4/ici_fsdp_parallelism: 1/' \
      -e 's/ici_expert_parallelism: 1/ici_expert_parallelism: 4/' \
      "$L/gemma4_26B${p}-4gpu.yaml" > "$L/sweep/gemma4_26B-ep4${p}.yaml"
done

# MoE via jax.lax.ragged_dot instead of dense dispatch (megablox is a TPU Pallas kernel).
sed 's/sparse_matmul: false/sparse_matmul: true/' "$L/gemma4_26B-4gpu.yaml" > "$L/sweep/gemma4_26B-ragged.yaml"

# xplane profile of steps 1-5 (MaxText defaults skip_first_n_steps_for_profiler=1, profiler_steps=5).
for m in 26B 31B; do
  sed 's/profiler: ""/profiler: "xplane"/' "$L/gemma4_${m}-4gpu.yaml" > "$L/sweep/gemma4_${m}-prof.yaml"
done

echo "wrote configs under $L"
