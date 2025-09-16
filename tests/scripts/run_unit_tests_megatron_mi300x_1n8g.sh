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
MEGATRON_PATH="$PRIMUS_PATH/third_party/Megatron-LM/"
disable_moe_dispatcher_check=""

pip install mock
# pip install multi-storage-client
pip install pytest-asyncio==0.23.6

export CUDA_DEVICE_MAX_CONNECTIONS=1
export HIP_FORCE_DEV_KERNARG=1
export HSA_ENABLE_SDMA=1
export HSA_NO_SCRATCH_RECLAIM=1
export NCCL_DEBUG=WARN
export RCCL_MSCCL_ENABLE=0
export TENSILE_DISABLE_STAGGERU=1

TORCHRUN_ARGS=(
  --nproc_per_node 8
  --nnodes 1
  --node_rank 0
  --master_addr localhost
  --master_port 50326
)

PYTEST_COV_ARGS=(
  --cov-branch
  --cov megatron
  --cov-append
  --no-cov-on-fail
)

clear_previous_runs() {
    pgrep -f python | xargs -r -n 1 kill -9 2>/dev/null || true
    sleep 10
}

cd "$MEGATRON_PATH"
if git apply --reverse --check "$SCRIPT_PATH/megatron_ut.patch"; then
    echo "Patch already applied, skipping."
else
    git apply "$SCRIPT_PATH/megatron_ut.patch"
fi

clear_previous_runs
torchrun \
  "${TORCHRUN_ARGS[@]}" \
  -m pytest -vxrs \
  "${PYTEST_COV_ARGS[@]}" \
  --ignore tests/unit_tests/data \
  --ignore tests/unit_tests/dist_checkpointing \
  --ignore tests/unit_tests/distributed \
  --ignore tests/unit_tests/export \
  --ignore tests/unit_tests/fusions \
  --ignore tests/unit_tests/inference \
  --ignore tests/unit_tests/models \
  --ignore tests/unit_tests/pipeline_parallel \
  --ignore tests/unit_tests/post_training \
  --ignore tests/unit_tests/ssm \
  --ignore tests/unit_tests/tensor_parallel \
  --ignore tests/unit_tests/transformer \
  --deselect "tests/unit_tests/test_parallel_state.py::test_different_initialize_order_unconsistency[src_tp_pp3-2]" \
  --deselect "tests/unit_tests/test_parallel_state.py::test_different_initialize_order_unconsistency[src_tp_pp4-2]" \
  --deselect "tests/unit_tests/test_parallel_state.py::test_different_initialize_order_unconsistency[src_tp_pp5-2]" \
  tests/unit_tests

# test_bin_reader requires multi-storage-client which leads to instability in dist_checkpointing
clear_previous_runs
torchrun \
  "${TORCHRUN_ARGS[@]}" \
  -m pytest -vxrs \
  "${PYTEST_COV_ARGS[@]}" \
  --ignore tests/unit_tests/data/test_bin_reader.py \
  tests/unit_tests/data

clear_previous_runs
torchrun \
  "${TORCHRUN_ARGS[@]}" \
  -m pytest -vxrs \
  "${PYTEST_COV_ARGS[@]}" \
  --ignore tests/unit_tests/dist_checkpointing/test_optimizer.py \
  tests/unit_tests/dist_checkpointing

clear_previous_runs
torchrun \
  "${TORCHRUN_ARGS[@]}" \
  -m pytest -vxrs \
  "${PYTEST_COV_ARGS[@]}" \
  --deselect "tests/unit_tests/distributed/test_distributed_data_parallel.py::TestDistributedDataParallel::test_ddp_with_dp_process_groups[2]" \
  --deselect "tests/unit_tests/distributed/test_mcore_fully_sharded_data_parallel.py::TestFullyShardedDataParallel::test_fsdp_with_process_groups[2]" \
  tests/unit_tests/distributed

clear_previous_runs
torchrun \
  "${TORCHRUN_ARGS[@]}" \
  -m pytest -vxrs \
  "${PYTEST_COV_ARGS[@]}" \
  tests/unit_tests/export

clear_previous_runs
torchrun \
  "${TORCHRUN_ARGS[@]}" \
  -m pytest -vxrs \
  "${PYTEST_COV_ARGS[@]}" \
  tests/unit_tests/fusions

clear_previous_runs
torchrun \
  "${TORCHRUN_ARGS[@]}" \
  -m pytest -vxrs \
  "${PYTEST_COV_ARGS[@]}" \
  --deselect tests/unit_tests/inference/engines/test_dynamic_engine.py::TestDynamicInferenceEngine \
  tests/unit_tests/inference

clear_previous_runs
torchrun \
  "${TORCHRUN_ARGS[@]}" \
  -m pytest -vxrs \
  "${PYTEST_COV_ARGS[@]}" \
  --deselect tests/unit_tests/models/test_bert_model.py::TestBertModelAttentionDimensions::test_transformer_engine_version_1_7_to_1_10_rng_error \
  --deselect tests/unit_tests/models/test_t5_model.py::TestT5Model::test_post_process_forward \
  --deselect tests/unit_tests/models/test_t5_model.py::TestT5Model::test_forward_with_encoder_hidden_states \
  --deselect tests/unit_tests/models/test_t5_model.py::TestT5Model::test_forward_output_encoder_hidden_only \
  tests/unit_tests/models

clear_previous_runs
torchrun \
  "${TORCHRUN_ARGS[@]}" \
  -m pytest -vxrs \
  "${PYTEST_COV_ARGS[@]}" \
  tests/unit_tests/pipeline_parallel

clear_previous_runs
torchrun \
  "${TORCHRUN_ARGS[@]}" \
  -m pytest -vxrs \
  "${PYTEST_COV_ARGS[@]}" \
  tests/unit_tests/post_training

clear_previous_runs
torchrun \
  "${TORCHRUN_ARGS[@]}" \
  -m pytest -vxrs \
  "${PYTEST_COV_ARGS[@]}" \
  tests/unit_tests/ssm

clear_previous_runs
torchrun \
  "${TORCHRUN_ARGS[@]}" \
  -m pytest -vxrs \
  "${PYTEST_COV_ARGS[@]}" \
  tests/unit_tests/tensor_parallel

clear_previous_runs
torchrun \
  "${TORCHRUN_ARGS[@]}" \
  -m pytest -vxrs \
  "${PYTEST_COV_ARGS[@]}" \
  --deselect "tests/unit_tests/transformer/test_transformer_block_custom_pgs.py::TestTransformerBlockWithProcessGroups::test_params_and_grads_match_transformer_block[1-1-1-1]" \
  --deselect "tests/unit_tests/transformer/test_transformer_block_custom_pgs.py::TestTransformerBlockWithProcessGroups::test_fwd_bwd_pass_non_uniform_transformer_block[1-1-1-1]" \
  --deselect "tests/unit_tests/transformer/test_transformer_block_custom_pgs.py::TestTransformerBlockWithProcessGroups::test_mlp_with_custom_pgs[1-1-1-False]" \
  --deselect "tests/unit_tests/transformer/test_transformer_block_custom_pgs.py::TestTransformerBlockWithProcessGroups::test_params_and_grads_match_transformer_block[2-2-1-1]" \
  --deselect "tests/unit_tests/transformer/test_transformer_block_custom_pgs.py::TestTransformerBlockWithProcessGroups::test_params_and_grads_match_transformer_block[2-1-2-1]" \
  --deselect "tests/unit_tests/transformer/test_transformer_block_custom_pgs.py::TestTransformerBlockWithProcessGroups::test_fwd_bwd_pass_non_uniform_transformer_block[2-1-1-2]" \
  --deselect "tests/unit_tests/transformer/test_transformer_block_custom_pgs.py::TestTransformerBlockWithProcessGroups::test_mlp_with_custom_pgs[2-1-2-False]" \
  --deselect "tests/unit_tests/transformer/test_transformer_block_custom_pgs.py::TestTransformerBlockWithProcessGroups::test_mlp_with_custom_pgs[2-2-1-False]" \
  --deselect "tests/unit_tests/transformer/test_transformer_block_custom_pgs.py::TestTransformerBlockWithProcessGroups::test_fwd_bwd_pass_non_uniform_transformer_block[2-2-1-1]" \
  --deselect "tests/unit_tests/transformer/test_multi_latent_attention.py::TestParallelMLAAttention::test_gpu_forward_thd" \
  --deselect "tests/unit_tests/transformer/test_multi_latent_attention.py::TestParallelMLAAttentionPrecision::test_gpu_forward_thd_precision" \
  --deselect "tests/unit_tests/transformer/test_retro_attention.py::TestRetroAttention::test_gpu_forward" \
  "$disable_moe_dispatcher_check" \
  tests/unit_tests/transformer

export CUDA_VISIBLE_DEVICES=0
TORCHRUN_ARGS=(
  --nproc_per_node 1
  --nnodes 1
  --node_rank 0
  --master_addr localhost
  --master_port 50326
)

clear_previous_runs
torchrun \
  "${TORCHRUN_ARGS[@]}" \
  -m pytest -vxrs \
  "${PYTEST_COV_ARGS[@]}" \
  "tests/unit_tests/transformer/test_transformer_block_custom_pgs.py::TestTransformerBlockWithProcessGroups::test_params_and_grads_match_transformer_block[1-1-1-1]" \
  "tests/unit_tests/transformer/test_transformer_block_custom_pgs.py::TestTransformerBlockWithProcessGroups::test_fwd_bwd_pass_non_uniform_transformer_block[1-1-1-1]" \
  "tests/unit_tests/transformer/test_transformer_block_custom_pgs.py::TestTransformerBlockWithProcessGroups::test_mlp_with_custom_pgs[1-1-1-False]"

export CUDA_VISIBLE_DEVICES=0,1
TORCHRUN_ARGS=(
  --nproc_per_node 2
  --nnodes 1
  --node_rank 0
  --master_addr localhost
  --master_port 50326
)

clear_previous_runs
torchrun \
  "${TORCHRUN_ARGS[@]}" \
  -m pytest -vxrs \
  "${PYTEST_COV_ARGS[@]}" \
  "tests/unit_tests/distributed/test_distributed_data_parallel.py::TestDistributedDataParallel::test_ddp_with_dp_process_groups[2]" \
  "tests/unit_tests/distributed/test_mcore_fully_sharded_data_parallel.py::TestFullyShardedDataParallel::test_fsdp_with_process_groups[2]" \
  "tests/unit_tests/transformer/test_transformer_block_custom_pgs.py::TestTransformerBlockWithProcessGroups::test_params_and_grads_match_transformer_block[2-2-1-1]" \
  "tests/unit_tests/transformer/test_transformer_block_custom_pgs.py::TestTransformerBlockWithProcessGroups::test_params_and_grads_match_transformer_block[2-1-2-1]" \
  "tests/unit_tests/transformer/test_transformer_block_custom_pgs.py::TestTransformerBlockWithProcessGroups::test_fwd_bwd_pass_non_uniform_transformer_block[2-1-1-2]" \
  "tests/unit_tests/transformer/test_transformer_block_custom_pgs.py::TestTransformerBlockWithProcessGroups::test_fwd_bwd_pass_non_uniform_transformer_block[2-2-1-1]" \
  "tests/unit_tests/transformer/test_transformer_block_custom_pgs.py::TestTransformerBlockWithProcessGroups::test_mlp_with_custom_pgs[2-1-2-False]" \
  "tests/unit_tests/transformer/test_transformer_block_custom_pgs.py::TestTransformerBlockWithProcessGroups::test_mlp_with_custom_pgs[2-2-1-False]"

end_time=$(date +%s)
elapsed=$((end_time - start_time))

echo "Total execution time: ${elapsed} seconds"
