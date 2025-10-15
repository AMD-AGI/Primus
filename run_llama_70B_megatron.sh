#!/bin/bash

######################### Training Docker and Envs #########################
export HF_TOKEN=${HF_TOKEN:-"your_hf_token"}
export DOCKER_IMAGE=${DOCKER_IMAGE:="docker.io/rocm/pytorch-training-private:20250929_gfx950_25dot9_rc4"}
export CLEAN_DOCKER_CONTAINER=1

export USING_AINIC=1
export REBUILD_PRIMUS_TURBO=1
export NCCL_IB_HCA="rocep105s0,rocep121s0,rocep137s0,rocep153s0,rocep233s0,rocep249s0,rocep25s0,rocep9s0"
# export AINIC_LIB="/apps/gpuperf/ainic-driver-20251007/lib/"
export ANP_HOME_DIR="/shared/apps/ubuntu/rocm-7.0.1/amd-anp-1.1.0-5"
export RCCL_HOME_DIR="/shared/apps/ubuntu/rocm-7.0.1/rccl-drop-2025-08"
export NCCL_SOCKET_IFNAME="enp193s0f1np1"
export GLOO_SOCKET_IFNAME="enp193s0f1np1"
export CPUS_PER_TASK=128
# export HSA_NO_SCRATCH_RECLAIM=0 
export NVTE_CK_USES_BWD_V3=1
export USE_ROCM_AITER_ROPE_BACKEND=0

######################### Training Config #########################
export EXP="examples/torchtitan/configs/llama3.1_70B-FP8-pretrain.yaml"
export NNODES=1
MBS=4
TP=1
ETP=1
GBS=$(($NNODES * 512))
SEQ_LENGTH=8192
PP=1
EP=1
CP=1
VPP=1
OPTIMIZER=adam
RECOMPUTE_LAYERS=0
RECOMPUTE_ID_START=0
BALANCE=True
LEGACY_GG=False
FP8=True # True for fp8, False for bf16

if [ $VPP -gt 1 ]; then
    export VPP_CONFIG="--num_virtual_stages_per_pipeline_rank $VPP"
fi

if [ "$FP8" = "True" ]; then
    export FP8_CONFIG="--fp8 hybrid"
fi

CONFIG="llama3.1_70B.FP8-$FP8.nodes$NNODES.GBS$GBS.MBS$MBS.PP$PP.EP$EP.CP$CP.VPP$VPP.rc-$RECOMPUTE_LAYERS"
echo "config: $CONFIG"

######################### Training Experiments #########################
export PRIMUS_TEAM="date-$(date +%Y%m%d)"
export PRIMUS_USER=user-abc
export PRIMUS_EXP_NAME=$CONFIG

LOG_DIR=./output/$PRIMUS_TEAM/$PRIMUS_USER/$PRIMUS_EXP_NAME/
export DUMP_PP_DIR=$LOG_DIR/pp_dump/
export LOG_FILE=$LOG_DIR/training.log
export EXPORT_CONFIG=$LOG_DIR/config.yaml
mkdir -p $LOG_DIR
echo $LOG_FILE


######################### Training Job #########################
bash ./examples/run_local_pretrain.sh --micro_batch_size $MBS \
                                      --global_batch_size $GBS \
                                      --tensor_model_parallel_size $TP \
                                      --expert_tensor_parallel_size $ETP \
                                      --pipeline_model_parallel_size $PP \
                                      --seq_length $SEQ_LENGTH \
                                      --expert_model_parallel_size $EP \
                                      --context_parallel_size $CP \
                                      --moe_router_force_load_balancing $BALANCE \
                                      --manual_gc False \
                                      --pp_warmup True \
                                      --manual_gc_interval 1 \
                                      --optimizer $OPTIMIZER \
                                      --cp_comm_type a2a \
                                      --recompute_num_layers $RECOMPUTE_LAYERS \
                                      --moe_use_legacy_grouped_gemm $LEGACY_GG \
                                      ${VPP_CONFIG} \
                                      ${FP8_CONFIG} \
                                      --profile True \
                                      --disable_profiler_activity_cpu True \
                                      --use_pytorch_profiler True \
                                      --profile_step_start 5 \
                                      --profile_step_end 6 \
                                      --train_iters 10 2>&1 | tee $LOG_FILE