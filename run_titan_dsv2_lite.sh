#!/bin/bash

export HF_TOKEN=${HF_TOKEN:-"your_hf_token"}
export USE_ROCM_AITER_ROPE_BACKEND=0
export CLEAN_DOCKER_CONTAINER=0

export USING_AINIC=0
export NCCL_IB_HCA="bnxt_re0,bnxt_re1,bnxt_re2,bnxt_re3,bnxt_re4,bnxt_re5,bnxt_re7,bnxt_re8"
# export AINIC_LIB="/apps/gpuperf/ainic-driver-20251007/lib/"
export ANP_HOME_DIR="/shared/apps/ubuntu/rocm-7.0.1/amd-anp-1.1.0-5"
export RCCL_HOME_DIR="/shared/apps/ubuntu/rocm-7.0.1/rccl-drop-2025-08"
# export NCCL_SOCKET_IFNAME="lo"
# export GLOO_SOCKET_IFNAME="lo"

export DOCKER_IMAGE="docker.io/rocm/pyt-megatron-lm-jax-nightly-private:primus_rocm7.1_20251117"
export CPUS_PER_TASK=128
export HSA_NO_SCRATCH_RECLAIM=1
export NVTE_CK_USES_BWD_V3=1

# export EXP="examples/torchtitan/configs/MI300X/deepseek_v3_16b-pretrain.yaml"
mkdir -p data
# the real number of nodes to run
export NNODES=1
MBS=8
TP=1
ETP=1
GBS=$(($NNODES * 512))
SEQ_LENGTH=4096
PP=1
EP=8
CP=1
VPP=1
OPTIMIZER=adam
RECOMPUTE_LAYERS=0
RECOMPUTE_ID_START=0
BALANCE=True
LEGACY_GG=False
FP8=False

CONFIG="titain-DSv2-Lite-FP8-$FP8.GBS$GBS.PP$PP.EP$EP.CP$CP.VPP$VPP.TOPK$TOPK.rc-$RECOMPUTE_LAYERS.rcids-$RECOMPUTE_ID_START.nodes$NNODES.$OPTIMIZER.BALANCE-$BALANCE-legacygg-$LEGACY_GG-noturboattn-noturbogg"
echo "config: $CONFIG"

if [ $VPP -gt 1 ]; then
    export VPP_CONFIG="--num_virtual_stages_per_pipeline_rank $VPP"
fi

if [ "$FP8" = "True" ]; then
    export FP8_CONFIG="--fp8 hybrid"
fi

export PRIMUS_TEAM="date-new-$(date +%Y%m%d)"
export PRIMUS_USER=liying
export PRIMUS_EXP_NAME=$CONFIG


LOG_DIR=./output/$PRIMUS_TEAM/$PRIMUS_USER/$PRIMUS_EXP_NAME/
export DUMP_PP_DIR=$LOG_DIR/pp_dump/
mkdir -p $LOG_DIR
LOG_FILE=$LOG_DIR/training.log
echo $LOG_FILE

EXPORT_CONFIG=$LOG_DIR/config.yaml

# bash ./examples/run_pretrain.sh   2>&1 | tee $LOG_FILE

export EXP="examples/torchtitan/configs/MI300X/deepseek_v3_671b-pretrain.yaml"
bash ./examples/run_pretrain.sh --model.n_layers 4 --model.n_dense_layers 0  2>&1 | tee $LOG_FILE
