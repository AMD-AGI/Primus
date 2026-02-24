#!/bin/bash

export HF_TOKEN="your_hf_token"
export DOCKER_IMAGE="docker.io/tasimage/primus:pr-556-ainic"

export NNODES=2 
# export NCCL_DEBUG=INFO
export USING_AINIC=1
export NCCL_IB_HCA="ionic_0:1,ionic_2:1,ionic_3:1,ionic_4:1,ionic_5:1,ionic_7:1,ionic_8:1,ionic_9:1"
export CLEAN_DOCKER_CONTAINER=1

# export EXP=examples/megatron/configs/MI355X/llama3.1_8B-BF16-pretrain.yaml
export EXP=examples/megatron/configs/MI355X/moe_proxy-BF16-pretrain.yaml

bash ./examples/run_slurm_pretrain.sh \
    --train_iters 10 \
    --micro_batch_size 2 \
    --global_batch_size 64 \
    --seq_length 16384 \
    --max_position_embeddings 16384 \
    --profile True \
    --use_pytorch_profiler True \
    --profile_step_end 7 \
    --profile_step_start 6 \
    2>&1 | tee log.txt