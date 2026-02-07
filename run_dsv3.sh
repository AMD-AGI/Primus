#!/bin/bash
export HF_TOKEN="your_hf_token" # change it to your own hf token
export WANDB_API_KEY="your_wandb_api_key" # change it to your own wandb api key
export DOCKER_IMAGE=primus_kernel_benchmark:backup
export NNODES=8
export NCCL_IB_HCA=ionic_0,ionic_2,ionic_3,ionic_4,ionic_5,ionic_7,ionic_8,ionic_9
export MICRO_BATCH_SIZE=1
export GLOBAL_BATCH_SIZE=$((64 * NNODES))
export ANP_HOME_DIR=${ANP_HOME_DIR:-"/workspace/ainic/amd-anp"}
export RCCL_HOME_DIR=${RCCL_HOME_DIR:-"/workspace/ainic/rccl"}
export MPI_HOME_DIR=${MPI_HOME_DIR:-"/workspace/ainic/ompi-4.1.6"}
export EXP=examples/megatron/configs/MI355X/deepseek_v3-BF16-pretrain.yaml
export USING_AINIC=1
export PRIMUS_TP=1
export PRIMUS_PP=8
export PRIMUS_EP=8
export PRIMUS_VPP=1
export TOTAL_ITERS=50
export PRIMUS_TOTAL_LAYERS=61
export PRIMUS_RECOMPUTE_LAYERS=8
export PRIMUS_MOE_LAYER_FREQ=1
export GLOO_SOCKET_IFNAME=ens9np0
export NCCL_SOCKET_IFNAME=ens9np0
export PRIMUS_DETERMINISTIC=0
bash ./examples/run_slurm_pretrain.sh \
--mtp_num_layers 0 \
--manual_gc True \
--manual_gc_interval 1 \
--pp_warmup True  \
--mock_data True \
--decoder_last_pipeline_num_layers 5 \
--micro_batch_size $MICRO_BATCH_SIZE --global_batch_size $GLOBAL_BATCH_SIZE --train_iters $TOTAL_ITERS \
--tensor_model_parallel_size $PRIMUS_TP \
--pipeline_model_parallel_size $PRIMUS_PP \
--expert_model_parallel_size $PRIMUS_EP \
--num_layers $PRIMUS_TOTAL_LAYERS --recompute_num_layers $PRIMUS_RECOMPUTE_LAYERS --moe_layer_freq $PRIMUS_MOE_LAYER_FREQ

# --manual_gc True \
# --manual_gc_interval 1 \
# --pp_warmup True  \