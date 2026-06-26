#!/bin/bash

# Make this script callable from ANY cwd. The line below `bash ./primus-cli`
# uses a RELATIVE path, so we MUST be in PRIMUS_ROOT for it to resolve.
# Without this block, running `bash examples/moe_package/start_training_dsv2_lite.sh`
# from inside `examples/` (or anywhere else) hits:
#   bash: ./primus-cli: No such file or directory
PRIMUS_ROOT="$(cd "$(dirname "$(realpath "$0")")/../.." && pwd)"
cd "$PRIMUS_ROOT" || { echo "[ERROR] cannot cd to PRIMUS_ROOT=$PRIMUS_ROOT" >&2; exit 1; }
echo "[start_training_dsv2_lite] PRIMUS_ROOT=$PRIMUS_ROOT, cwd=$(pwd)"

export HF_TOKEN="your_hf_token"  # make it your own hf token
export WANDB_API_KEY="your_wandb_api_key"  # make it your own wandb api key
export DOCKER_IMAGE="docker.io/tasimage/primus:pr-563-ainic"
#export SLURM_TREE_WIDTH=128

export NNODES=1
export MASTER_PORT=29500
export SLURM_TIME=07:00:00
export SLURM_PARTITION=amd-aig

# export NCCL_DEBUG=INFO
export USING_AINIC=1
# export NCCL_IB_HCA="ionic_0:1,ionic_2:1,ionic_3:1,ionic_4:1,ionic_5:1,ionic_7:1,ionic_8:1,ionic_9:1"
# export GLOO_SOCKET_IFNAME=ens9np0
# export NCCL_SOCKET_IFNAME=ens9np0
export HSA_NO_SCRATCH_RECLAIM=1
export NVTE_CK_USES_BWD_V3=1
export GPU_MAX_HW_QUEUES=4
export CLEAN_DOCKER_CONTAINER=1

export MBS=8   #12
export GBS=$((128 * NNODES)) # 96
export PROFILE=True
export TURBO_GROUPED_GEMM=False
export TURBO_DEEPEEP=True
export LEGACY_GG=True
export PRIMUS_DETERMINISTIC=0

# export EXP=examples/megatron/configs/MI355X/llama3.1_8B-BF16-pretrain.yaml
export EXP=examples/megatron/configs/MI355X/deepseek_v2_lite-BF16-pretrain.yaml
export PRIMUS_TEAM=amd
export PRIMUS_USER=tas
export PRIMUS_EXP_NAME=dsv2_lite-pretrain-mbs_$MBS-gbs_$GBS-turbogg_$TURBO_GROUPED_GEMM-turbodeepep_$TURBO_DEEPEEP-legacygg_$LEGACY_GG-profile_$PROFILE

mkdir -p output/$PRIMUS_TEAM/$PRIMUS_USER/$PRIMUS_EXP_NAME

# primus-cli usage:
#   primus-cli [global-options] <mode> [mode-args] -- <python-cli-cmd> [python-cli-args]
# Mode is "direct" (launch torchrun on the current host/container).
# After "--", we tell the primus python CLI to run "train pretrain --config <yaml>",
# then everything else (--train_iters, --micro_batch_size, ...) is forwarded as
# yaml-override args to the trainer (see primus/cli/main.py L161-162: train accepts
# unknown_args).
bash ./primus-cli direct -- train pretrain --config "$EXP" \
  --train_iters 20 \
  --disable_wandb True \
  --disable_tensorboard True \
  --micro_batch_size $MBS \
  --global_batch_size $GBS \
  --seq_length 4096 \
  --max_position_embeddings 4096 \
  --use_turbo_grouped_gemm $TURBO_GROUPED_GEMM \
  --use_turbo_deepep $TURBO_DEEPEEP \
  --moe_use_legacy_grouped_gemm $LEGACY_GG \
  --cross_entropy_fusion_impl "te" \
  --cross_entropy_loss_fusion True \
  --profile $PROFILE \
  --use_pytorch_profiler $PROFILE \
  --profile_step_end 7 \
  --profile_step_start 6 \
  2>&1 | tee output/$PRIMUS_TEAM/$PRIMUS_USER/$PRIMUS_EXP_NAME/log.txt
