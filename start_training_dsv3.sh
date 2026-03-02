#!/bin/bash

export HF_TOKEN="your_hf_token"  # make it your own hf token
export WANDB_API_KEY="your_wandb_api_key"  # make it your own wandb api key
export DOCKER_IMAGE="docker.io/tasimage/primus:pr-563-ainic"
# export SLURM_TREE_WIDTH=128 
export NNODES=128
export TRAIN_ITERS=5000
export SLURM_TIME=07:00:00
export SLURM_PARTITION=amd-aig
export SLURM_NODELIST="uswslocpm2m-106-[005,015,021,030-031,038-039,042,050,056-057,063,069,077,079-080,082,084-086,091-092,122,125,138,142,145,151,176,179-180,185,190,194,196-197,199,212-215,218,220-221,224-226,273,275,281,285,297,310,319,340,346,359-360,362,373,387,392,395,399,419,423,433,442,444-445,448-450,452,454,456-457,472-475,479-481,629,631,635,640,646,656,658,663-664,667,678,681,687,695-696,700,723,732,735,740-741,757,760-761,766,772,781,784,833,841-842,851,857,865,868,883,889,895,899-900,905,1066,1070,1177]"

# export NCCL_DEBUG=INFO
export USING_AINIC=1
export NCCL_IB_HCA="ionic_0:1,ionic_2:1,ionic_3:1,ionic_4:1,ionic_5:1,ionic_7:1,ionic_8:1,ionic_9:1"
export GLOO_SOCKET_IFNAME=ens9np0
export NCCL_SOCKET_IFNAME=ens9np0
export CLEAN_DOCKER_CONTAINER=1

export MBS=1
export GBS=$((128 * NNODES))
export PRIMUS_TOTAL_LAYERS=61
export PRIMUS_RECOMPUTE_LAYERS=4
export PRIMUS_MOE_LAYER_FREQ=1
export PRIMUS_PP=4
export DECODER_LAST_PIPELINE_NUM_LAYERS=13
export PRIMUS_EP=8
export PRIMUS_VPP=1
export PROFILE=False
export TURBO_DEEPEEP=True
export LEGACY_GG=True
export PRIMUS_DETERMINISTIC=0
# Enable NUMA binding for better memory locality (increase stability for large models)
export ENABLE_NUMA_BINDING=1
export HSA_KERNARG_POOL_SIZE=12582912
# export SLURM_NODELIST="uswslocpm2m-106-[273,297,310,319,687,732,836,892]"
# export EXP=examples/megatron/configs/MI355X/llama3.1_8B-BF16-pretrain.yaml
export EXP=examples/megatron/configs/MI355X/deepseek_v3-BF16-pretrain.yaml
export PRIMUS_TEAM=amd
export PRIMUS_USER=tas
export PRIMUS_TOKENIZED_DATA_PATH=/shared_aig/c4/tokenized/c4_en_train_text_document # this is the tokenized data path for the training
export PRIMUS_EXP_NAME=dsv3-pretrain-mbs_$MBS-gbs_$GBS-PP_$PRIMUS_PP-EP_$PRIMUS_EP-VPP_$PRIMUS_VPP-turbodeepep_$TURBO_DEEPEEP-legacygg_$LEGACY_GG-profile_$PROFILE
export PRIMUS_EXP_NAME=dsv3-pp4


mkdir -p output/$PRIMUS_TEAM/$PRIMUS_USER/$PRIMUS_EXP_NAME
bash ./examples/run_slurm_pretrain.sh \
  --num_layers $PRIMUS_TOTAL_LAYERS \
  --train_iters $TRAIN_ITERS \
  --micro_batch_size $MBS \
  --global_batch_size $GBS \
  --use_turbo_deepep $TURBO_DEEPEEP \
  --lr 2.2e-4 \
  --min_lr 2.2e-5 \
  --lr_warmup_iters 200 \
  --lr_decay_iters 5000 \
  --lr_decay_style cosine \
  --moe_use_legacy_grouped_gemm $LEGACY_GG \
  --pipeline_model_parallel_size $PRIMUS_PP \
  --expert_model_parallel_size $PRIMUS_EP \
  --decoder_last_pipeline_num_layers $DECODER_LAST_PIPELINE_NUM_LAYERS \
  --cross_entropy_fusion_impl "te" \
  --cross_entropy_loss_fusion True \
  --recompute_num_layers $PRIMUS_RECOMPUTE_LAYERS \
  --recompute_granularity full \
  --recompute_method block \
  --moe_layer_freq $PRIMUS_MOE_LAYER_FREQ \
  --mock_data False \
  --manual_gc True \
  --manual_gc_interval 1 \
  --pp_warmup True  \
  --mtp_num_layers 0 \
  --profile $PROFILE \
  --use_pytorch_profiler $PROFILE \
  --profile_step_end 7 \
  --profile_step_start 6 \
  --disable_wandb False \
  --disable_tensorboard False \
  2>&1 | tee output/$PRIMUS_TEAM/$PRIMUS_USER/$PRIMUS_EXP_NAME/log.txt
