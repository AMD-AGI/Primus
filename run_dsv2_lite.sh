#!/bin/bash
export HF_TOKEN="your_hf_token" # change it to your own hf token
export WANDB_API_KEY="your_wandb_api_key" # change it to your own wandb api key
export DOCKER_IMAGE=primus_kernel_benchmark:backup
export NNODES=4
export NCCL_IB_HCA=ionic_0,ionic_2,ionic_3,ionic_4,ionic_5,ionic_7,ionic_8,ionic_9
export GLOBAL_BATCH_SIZE=$((96 * NNODES))
export ANP_HOME_DIR=${ANP_HOME_DIR:-"/workspace/ainic/amd-anp"}
export RCCL_HOME_DIR=${RCCL_HOME_DIR:-"/workspace/ainic/rccl"}
export MPI_HOME_DIR=${MPI_HOME_DIR:-"/workspace/ainic/ompi-4.1.6"}
export EXP=examples/megatron/configs/MI355X/deepseek_v2_lite-BF16-pretrain.yaml
export USING_AINIC=1
export GLOO_SOCKET_IFNAME=ens9np0
export NCCL_SOCKET_IFNAME=ens9np0
export PRIMUS_DETERMINISTIC=0
bash ./examples/run_slurm_pretrain.sh --global_batch_size $GLOBAL_BATCH_SIZE --train_iters 50 --debug