#!/bin/bash

export HF_TOKEN="your_hf_token"
# export DOCKER_IMAGE="docker.io/tasimage/primus:pr-556-ainic"
export DOCKER_IMAGE="docker.io/tasimage/primus:pr-563-ainic"

export NNODES=2
# export NCCL_DEBUG=INFO
export USING_AINIC=1
export NCCL_IB_HCA="ionic_0:1,ionic_2:1,ionic_3:1,ionic_4:1,ionic_5:1,ionic_7:1,ionic_8:1,ionic_9:1"
export CLEAN_DOCKER_CONTAINER=1

export MBS=2
export GBS=64
export HEAD_DIM=64
export TURBO_GROUPED_MLP=True
export TURBO_DEEPEEP=True
export LEGACY_GG=True
export TURBO_SYNC_FREE_MOE_STAGE=1
export TURBO_PERMUTE_FUSION=True

# export EXP=examples/megatron/configs/MI355X/llama3.1_8B-BF16-pretrain.yaml
export EXP=examples/megatron/configs/MI355X/moe_proxy-BF16-pretrain.yaml
export PRIMUS_TEAM=amd
export PRIMUS_USER=tas
export PRIMUS_EXP_NAME=moe_proxy-pretrain-mbs_$MBS-gbs_$GBS-headdim_$HEAD_DIM-turbogg_$TURBO_GROUPED_MLP-turbodeepep_$TURBO_DEEPEEP-legacygg_$LEGACY_GG-syncfree_$TURBO_SYNC_FREE_MOE_STAGE-permute_$TURBO_PERMUTE_FUSION


mkdir -p output/$PRIMUS_TEAM/$PRIMUS_USER/$PRIMUS_EXP_NAME
bash ./examples/run_slurm_pretrain.sh \
    --train_iters 10 \
    --micro_batch_size $MBS \
    --global_batch_size $GBS \
    --seq_length 16384 \
    --max_position_embeddings 16384 \
    --use_turbo_grouped_mlp $TURBO_GROUPED_MLP \
    --use_turbo_deepep $TURBO_DEEPEEP \
    --moe_use_legacy_grouped_gemm $LEGACY_GG \
    --turbo_sync_free_moe_stage $TURBO_SYNC_FREE_MOE_STAGE \
    --moe_permute_fusion $TURBO_PERMUTE_FUSION \
    --qk_head_dim $HEAD_DIM \
    --v_head_dim $HEAD_DIM \
    --kv_channels $HEAD_DIM \
    --profile True \
    --use_pytorch_profiler True \
    --profile_step_end 7 \
    --profile_step_start 6 \
    2>&1 | tee output/$PRIMUS_TEAM/$PRIMUS_USER/$PRIMUS_EXP_NAME/log.txt
