#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
set -euxo pipefail

start_time=$(date +%s)

# Get directory of this script
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRIMUS_PATH="$SCRIPT_PATH/../../"
TORCHTITAN_PATH="$PRIMUS_PATH/third_party/torchtitan/"

pip install --upgrade pip
pip install --no-cache-dir -r "$PRIMUS_PATH/requirements.txt"
pip install tomli_w

export CUDA_DEVICE_MAX_CONNECTIONS=1
export HIP_FORCE_DEV_KERNARG=1
export HSA_ENABLE_SDMA=1
export HSA_NO_SCRATCH_RECLAIM=1
export NCCL_DEBUG=WARN
export RCCL_MSCCL_ENABLE=0

TEST_CASES=(
  default
  1d_compile
  1d_compile_sac_op
  full_checkpoint
  model_weights_only_bf16
  model_weights_only_fp32
  # fix follows needs update pytorch, https://github.com/pytorch/torchtitan/issues/1188
  # pp_looped_zero_bubble
  # pp_zbv
  # pp_custom_csv
  pp_1f1b
  pp_gpipe
  pp_dp_1f1b
  pp_dp_gpipe
  pp_tp
  pp_looped_1f1b
  optimizer_foreach
  ddp
  hsdp
  fsdp+flex_attn
  cp_allgather
  cp_alltoall
  fsdp+cp
  hsdp+cp_without_dp_shard
  fsdp2_memory_estimation
  test_generate
  fsdp_reshard_always
  float8_emulation
  # Set CUDA_DEVICE_MAX_CONNECTIONS=1 to test the follows to avoid nccl timeout
  2d_eager
  2d_compile
  pp_dp_tp
  3d_compile
  hsdp+tp
  hsdp+cp_with_dp_shard
  fsdp+tp+cp
  cpu_offload+opt_in_bwd+TP+DP+CP
  optional_checkpoint
)

cd "$TORCHTITAN_PATH"
for case in "${TEST_CASES[@]}"; do
  if [ -d "$TORCHTITAN_PATH/artifacts-to-be-uploaded" ]; then
    rm -rf "$TORCHTITAN_PATH/artifacts-to-be-uploaded"
  fi
  mkdir -p "$TORCHTITAN_PATH/artifacts-to-be-uploaded"
  python ./tests/integration_tests.py artifacts-to-be-uploaded --test "$case" --ngpu 8
done

# unsupport async-tp
TEST_CASES=(
  # 2d_asynctp_compile
  float8
  # fsdp+tp+cp+compile+float8
  hsdp+cp+compile+float8
)

for case in "${TEST_CASES[@]}"; do
  if [ -d "$TORCHTITAN_PATH/artifacts-to-be-uploaded" ]; then
    rm -rf "$TORCHTITAN_PATH/artifacts-to-be-uploaded"
  fi
  mkdir -p "$TORCHTITAN_PATH/artifacts-to-be-uploaded"
  python ./tests/integration_tests_h100.py artifacts-to-be-uploaded --test "$case" --ngpu 8
done

TEST_CASES=(
  default
  full_checkpoint
  model_weights_only_fp32
  fsdp
  hsdp
)

for case in "${TEST_CASES[@]}"; do
  if [ -d "$TORCHTITAN_PATH/artifacts-to-be-uploaded" ]; then
    rm -rf "$TORCHTITAN_PATH/artifacts-to-be-uploaded"
  fi
  mkdir -p "$TORCHTITAN_PATH/artifacts-to-be-uploaded"
  python -m torchtitan.experiments.flux.tests.integration_tests artifacts-to-be-uploaded --test "$case" --ngpu 8
done

TEST_CASES=(
  1d
  1d_sac_op
  1d_full_ac
  # AssertionError: Unexpected type <class 'torch.distributed.tensor.placement_types._StridedShard'>
  # 2d
  # pp_dp_tp
  # fsdp+tp+cp
  ddp
  hsdp
  # 2d_asynctp
  # full_checkpoint
  ddp
  hsdp
  # hsdp+tp
  ddp+tp
  hsdp+cp_with_dp_shard
  # optional_checkpoint
)

for case in "${TEST_CASES[@]}"; do
  if [ -d "$TORCHTITAN_PATH/artifacts-to-be-uploaded" ]; then
    rm -rf "$TORCHTITAN_PATH/artifacts-to-be-uploaded"
  fi
  mkdir -p "$TORCHTITAN_PATH/artifacts-to-be-uploaded"
  python -m torchtitan.experiments.simple_fsdp.tests.integration_tests artifacts-to-be-uploaded --test "$case" --ngpu 8
done

if [ -d "$TORCHTITAN_PATH/artifacts-to-be-uploaded" ]; then
  rm -rf "$TORCHTITAN_PATH/artifacts-to-be-uploaded"
fi

end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Total execution time: ${elapsed} seconds"
