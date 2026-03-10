#!/bin/bash
set -x

export WANDB_API_KEY="your_wandb_api_key"  # make it your own wandb api key
export DOCKER_IMAGE="docker.io/tasimage/primus:pr-563-ainic"
# export SLURM_TREE_WIDTH=128 
export NNODES=8
export TRAIN_ITERS=10
export SLURM_TIME=48:00:00
export SLURM_PARTITION="mi355x"
# export SLURM_NODELIST="uswslocpm2m-106-[005,015,021,030-031,038-039,042,050,056-057,063,069,077,079-080,082,084-086,091-092,122,125,138,142,145,151,176,179-180,185,190,194,196-197,199,212-215,218,220-221,224-226,273,275,281,285,297,310,319,340,346,359-360,362,373,387,392,395,399,419,423,433,442,444-445,448-450,452,454,456-457,472-475,479-481,629,631,635,640,646,656,658,663-664,667,678,681,687,695,103,700,723,732,735,740-741,757,760-761,766,772,781,784,833,841-842,851,857,865,868,883,889,895,899-900,905,1066,1070,1177]"
# export SLURM_NODELIST="uswslocpm2m-106-[005,015,021,030-031,038-039,042,050,056-057,063,069,077,079-080,082,084-086,091-092,122,125,138,142,145,151,176,179-180,185,190,194,196-197,199,212-215,218,220-221,224-226,273,275,281,285,297,310,319,340,346,359-360,362,373,387,392,395,399,419,423,433,442,444-445,448-450,452,454,456-457,472-475,479-481,629,631,635,640,646,656,658,663-664,667,678,681,687,695-696,700,723,732,735,740-741,757,760-761,766,772,781,784,833,841-842,851,857,865,868,883,889,895,899-900,905,1066,1070,1177]"
# export SLURM_NODELIST="uswslocpm2m-106-[005,015,021,030-031,038-039,042,050,056-057,063,069,077,079-080,082,084-086,091-092,122,125,138,142,145,151,176,179-180,185,190,194,196-197,199,212-215,218,220-221,224-226,273,275,281,285,297,310,319,340,346,359-360,362,373,387,392,395,399,419,423,433,442,444-445,448-450,452,454,456-457,472-475,479-481,629,631,635,640,646,628,658,663-664,667,678,681,687,695,103,700,723,732,735,740-741,757,760-761,766,772,781,784,833,841-842,851,857,865,868,883,889,895,899-900,905,1066,1070,1177]"
#export SLURM_NODELIST="chi[2817,2820]"
#export SLURM_NODELIST="chi[2817,2820-2822,2832,2834,2836-2837,2841,2849,2851,2863,2874-2875,2877,2882]"
#export NCCL_DEBUG=INFO
export USING_AINIC=1
export NCCL_IB_HCA="ionic_0:1,ionic_1:1,ionic_2:1,ionic_3:1,ionic_4:1,ionic_5:1,ionic_6:1,ionic_7:1"
export GLOO_SOCKET_IFNAME="enp193s0f0np0"
export NCCL_SOCKET_IFNAME="enp193s0f0np0"
export CLEAN_DOCKER_CONTAINER=1

export MBS=2
export GBS=$((16 * NNODES))
export PRIMUS_TOTAL_LAYERS=61
export PRIMUS_RECOMPUTE_LAYERS=4
export PRIMUS_MOE_LAYER_FREQ=1
export PRIMUS_PP=8
export PRIMUS_EP=128
export PRIMUS_VPP=2
export PROFILE=False
export TURBO_DEEPEEP=False
export LEGACY_GG=True
export PRIMUS_DETERMINISTIC=0
# Enable NUMA binding for better memory locality (increase stability for large models)
export ENABLE_NUMA_BINDING=1
export HSA_KERNARG_POOL_SIZE=12582912
# export SLURM_NODELIST="uswslocpm2m-106-[273,297,310,319,687,732,836,892]"
# export EXP=examples/megatron/configs/MI355X/llama3.1_8B-BF16-pretrain.yaml


FEATURE_ARGS=()
PIPELINE_ARGS=()
if [ "$PRIMUS_VPP" -gt 1 ]; then
  case "$PRIMUS_VPP" in
    2)
      if [ "$PRIMUS_PP" -eq 4 ]; then
        # DeepSeek-V3 has 61 decoder layers. For PP4+VPP2 (8 pipeline chunks),
        # use a balanced split: 8,8,8,8,8,7,7,7.
        FEATURE_ARGS+=("--pipeline_model_parallel_layout" "'Et*8|t*8|t*8|t*8|t*8|t*7|t*7|t*7,L'")
      elif [ "$PRIMUS_PP" -eq 8 ]; then
        # DeepSeek-V3 has 61 decoder layers. For PP8+VPP2 (16 pipeline chunks),
        # use a balanced split: 4x13 + 3x3 = 61 (13 stages with 4 layers, 3 stages with 3 layers).
        FEATURE_ARGS+=("--pipeline_model_parallel_layout" "'Et*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*3|t*3|t*3,L'")
      else
        echo "Unsupported PRIMUS_PP=${PRIMUS_PP} for PRIMUS_VPP=2. Supported PP values: 4, 8." >&2
        exit 1
      fi
      ;;
    4)
      # DeepSeek-V3 has 61 decoder layers. For PP4+VPP4 (16 pipeline chunks),
      # use a balanced split: 4x13 + 3x3 = 61 (13 stages with 4 layers, 3 stages with 3 layers).
      FEATURE_ARGS+=("--pipeline_model_parallel_layout" "'Et*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*3|t*3|t*3,L'")
      ;;
    *)
      echo "Unsupported PRIMUS_VPP=${PRIMUS_VPP}. Supported values in this script: 1, 2, 4." >&2
      exit 1
      ;;
  esac
else
  if [ -z "${DECODER_LAST_PIPELINE_NUM_LAYERS:-}" ]; then
    if [ "$PRIMUS_PP" -eq 4 ]; then
      DECODER_LAST_PIPELINE_NUM_LAYERS=13
    elif [ "$PRIMUS_PP" -eq 8 ]; then
      DECODER_LAST_PIPELINE_NUM_LAYERS=12
    else
      DECODER_LAST_PIPELINE_NUM_LAYERS=13
    fi
  fi
  export DECODER_LAST_PIPELINE_NUM_LAYERS
  MIDDLE_PP_SIZE=$((PRIMUS_PP - 1))
  if [ "$MIDDLE_PP_SIZE" -le 0 ]; then
    echo "Invalid PRIMUS_PP=${PRIMUS_PP}. PRIMUS_PP must be >= 2 when PRIMUS_VPP <= 1." >&2
    exit 1
  fi
  MIDDLE_STAGE_LAYERS=$((PRIMUS_TOTAL_LAYERS - DECODER_LAST_PIPELINE_NUM_LAYERS))
  if [ $((MIDDLE_STAGE_LAYERS % MIDDLE_PP_SIZE)) -ne 0 ]; then
    echo "Invalid split: PRIMUS_TOTAL_LAYERS=${PRIMUS_TOTAL_LAYERS}, DECODER_LAST_PIPELINE_NUM_LAYERS=${DECODER_LAST_PIPELINE_NUM_LAYERS}, PRIMUS_PP=${PRIMUS_PP}. (PRIMUS_TOTAL_LAYERS - DECODER_LAST_PIPELINE_NUM_LAYERS) must be divisible by (PRIMUS_PP - 1)." >&2
    exit 1
  fi
  PIPELINE_ARGS+=("--decoder_last_pipeline_num_layers" "$DECODER_LAST_PIPELINE_NUM_LAYERS")
fi

export EXP=examples/megatron/configs/MI355X/deepseek_v3-FP8-pretrain.yaml
export PRIMUS_TEAM=amd
export PRIMUS_USER=tas
export PRIMUS_TOKENIZED_DATA_PATH=/shared_aig/c4/tokenized/c4_en_train_text_document # this is the tokenized data path for the training
export PRIMUS_EXP_NAME=dsv3-pretrain-mbs_$MBS-gbs_$GBS-PP_$PRIMUS_PP-EP_$PRIMUS_EP-VPP_$PRIMUS_VPP-turbodeepep_$TURBO_DEEPEEP-legacygg_$LEGACY_GG-profile_$PROFILE

#CKPT_DIR=output/$PRIMUS_TEAM/$PRIMUS_USER/$PRIMUS_EXP_NAME/checkpoints

mkdir -p output/$PRIMUS_TEAM/$PRIMUS_USER/$PRIMUS_EXP_NAME
# mkdir -p "$CKPT_DIR"
#bash ./examples/run_slurm_pretrain.sh \
./runner/primus-cli --debug slurm -- train pretrain --config ${EXP} \
--num_layers $PRIMUS_TOTAL_LAYERS \
--train_iters $TRAIN_ITERS \
--micro_batch_size $MBS \
--global_batch_size $GBS \
--use_turbo_deepep $TURBO_DEEPEEP \
--turbo_sync_free_moe_stage 1 \
--lr 2.2e-4 \
--min_lr 2.2e-5 \
--lr_warmup_iters 200 \
--lr_decay_iters 5000 \
--use_turbo_grouped_mlp True \
--lr_decay_style cosine \
--moe_use_legacy_grouped_gemm $LEGACY_GG \
--pipeline_model_parallel_size $PRIMUS_PP \
--expert_model_parallel_size $PRIMUS_EP \
"${PIPELINE_ARGS[@]}" \
"${FEATURE_ARGS[@]}" \
--cross_entropy_fusion_impl "te" \
--cross_entropy_loss_fusion True \
--recompute_num_layers $PRIMUS_RECOMPUTE_LAYERS \
--recompute_granularity full \
--recompute_method block \
--disable_last_saving True \
--moe_layer_freq $PRIMUS_MOE_LAYER_FREQ \
--mock_data True \
--manual_gc True \
--manual_gc_interval 1 \
--pp_warmup True  \
--mtp_num_layers 0 \
--profile $PROFILE \
--use_pytorch_profiler $PROFILE \
--profile_step_end 7 \
--profile_step_start 6 \
--disable_wandb True \
--disable_tensorboard False \
2>&1 | tee output/$PRIMUS_TEAM/$PRIMUS_USER/$PRIMUS_EXP_NAME/log.txt
