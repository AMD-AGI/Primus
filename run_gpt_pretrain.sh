#!/bin/bash
export PRIMUS_WORKSPACE=output/gpt/gpt_oss_20b_run
export PRIMUS_USER=gpt
export PRIMUS_GROUP="date-$(date +%Y%m%d)"
export PRIMUS_EXP_NAME=$CONFIG
mkdir -p $PRIMUS_WORKSPACE

# alpha docker image for MI250X
export DOCKER_IMAGE=docker.io/rocm/pyt-megatron-lm-jax-nightly-private:pytorch_gfx950_20250908 
# export NCCL_IB_HCA=${NCCL_IB_HCA:="benic1p1,benic2p1,benic3p1,benic4p1,benic5p1,benic7p1,benic8p1"} # modify based on the GPU NiC settings

export HF_TOKEN="your_huggingface_token"
export NNODES=1
export HSA_NO_SCRATCH_RECLAIM=1
export NVTE_CK_USES_BWD_V3=1
TP=1
ETP=1
BALANCE=True
PROFILE=true
MBS=6
GBS=480
PR=bf16 # bf16 # fp8 # need to change the config examples/megatron/config/gpt_oss_xxx-pretrain.yaml file accordingly
MOE_USE_LEGACY_GG=false
PP=1
EP=8
CP=1
VPP=1
OPTIMIZER=adam
RECOMPUTE_LAYERS=0
RECOMPUTE_ID_START=0


CONFIG="mockdata.gc.${PR}.GBS$GBS.MBS.${MBS}.PP$PP.EP$EP.CP$CP.VPP$VPP.rc-$RECOMPUTE_LAYERS.rcids-$RECOMPUTE_ID_START.nodes$NNODES.$OPTIMIZER.BALANCE-$BALANCE-USE_LEGACY_GG-$MOE_USE_LEGACY_GG"
echo "config: $CONFIG"
export PRIMUS_EXP_NAME=$CONFIG
LOG_DIR=./$PRIMUS_WORKSPACE/$PRIMUS_GROUP/$PRIMUS_USER/$CONFIG/
mkdir -p $LOG_DIR
LOG_FILE=$LOG_DIR/training.log
echo $LOG_FILE

EXPORT_CONFIG=$LOG_DIR/config.yaml

export EXP=examples/megatron/configs/gpt_oss_20B-pretrain.yaml 
# export EXP=examples/megatron/configs/gpt_oss_120B-pretrain.yaml 

docker stop | grep $DOCKER_IMAGE | awk '{print $1}' | xargs -r docker rm -f

bash ./examples/run_slurm_pretrain.sh --micro_batch_size $MBS \
                                      --global_batch_size $GBS \
                                      --tensor_model_parallel_size $TP \
                                      --expert_tensor_parallel_size $ETP \
                                      --pipeline_model_parallel_size $PP \
                                      --expert_model_parallel_size $EP \
                                      --context_parallel_size $CP \
                                      --manual_gc True \
                                      --manual_gc_interval 1 \
                                      --moe_router_force_load_balancing $BALANCE \
                                      --optimizer $OPTIMIZER \
                                      --recompute_num_layers $RECOMPUTE_LAYERS \
                                      --moe_use_legacy_grouped_gemm ${MOE_USE_LEGACY_GG} \
                                      --moe_router_num_groups 1 \
                                      --moe_router_group_topk 1 \
                                      --profile ${PROFILE} \
                                      --train_iters 20 \
                                      --disable_profiler_activity_cpu False \
                                      --use_pytorch_profiler True \
                                      --profile_step_start 5 \
                                      --profile_step_end 6 \
                                      --train_iters 10 2>&1 | tee $LOG_FILE

                                    