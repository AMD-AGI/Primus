# export NCCL_IB_HCA=mlx5_ib0:1,mlx5_ib1:1,mlx5_ib2:1,mlx5_ib3:1,mlx5_ib4:1,mlx5_ib5:1,mlx5_ib6:1,mlx5_ib7:1
# export NCCL_IB_HCA=^mlx5_1,mlx5_6
export CPUS_PER_TASK=96
export HSA_NO_SCRATCH_RECLAIM=1 # change to 0
export NVTE_CK_USES_BWD_V3=1 # change to 0

export EXP="examples/megatron/configs/deepseek_1T-pretrain.yaml"
# export EXP="examples/megatron/configs/MI300X/deepseek_v2_lite-pretrain.yaml"
# export EXP="examples/megatron/configs/MI300X/deepseek_v3-pretrain.yaml"
# the real number of nodes to run
export NNODES=4

MBS=1
TP=1
ETP=1

GBS=256
PP=2
EP=16
VPP=4
OPTIMIZER=adam
RECOMPUTE_LAYERS=0
RECOMPUTE_ID_START=0
BALANCE=True


CONFIG="base.deepep.Turbo-attn-gg-deepep.mockdata.gc.BF16.GBS$GBS.PP$PP.EP$EP.CP$CP.VPP$VPP.TOPK$TOPK.rc-$RECOMPUTE_LAYERS.rcids-$RECOMPUTE_ID_START.nodes$NNODES.$OPTIMIZER.BALANCE-$BALANCE"
echo "config: $CONFIG"

export VPP_CONFIG=" "

if [ $VPP -gt 1 ]; then
    export VPP_CONFIG="--num_virtual_stages_per_pipeline_rank $VPP"
fi

export PRIMUS_WORKSPACE=output/poolside/1T
export PRIMUS_USER=yuankai
export PRIMUS_GROUP="date-$(date +%Y%m%d)"x
export PRIMUS_EXP_NAME=$CONFIG
mkdir -p $PRIMUS_WORKSPACE

export GPU_MAX_HW_QUEUES=8
export DOCKER_IMAGE=docker.io/yuankaichenamd/megatron_rocm_private:25.9
# export DOCKER_IMAGE=docker.io/rocm/mad-private:pytorch_rocm7.0_ci_7bf4233_20251106

LOG_DIR=./$PRIMUS_WORKSPACE/$PRIMUS_GROUP/$PRIMUS_USER/$CONFIG/
export PRIMUS_PPDUMP_FILE=$LOG_DIR/pp_dump/
mkdir -p $LOG_DIR
LOG_FILE=$LOG_DIR/training.log
echo $LOG_FILE

EXPORT_CONFIG=$LOG_DIR/config.yaml
bash ./examples/run_slurm_pretrain.sh --micro_batch_size $MBS \
                                      --global_batch_size $GBS \
                                      --tensor_model_parallel_size $TP \
                                      --expert_tensor_parallel_size $ETP \
                                      --pipeline_model_parallel_size $PP \
                                      --expert_model_parallel_size $EP \
                                      --moe_router_force_load_balancing $BALANCE \
                                      --manual_gc True \
                                      --manual_gc_interval 1 \
                                      --optimizer $OPTIMIZER \
                                      --cp_comm_type a2a \
                                      --dump_pp_data False \
                                      ${VPP_CONFIG} \
                                      --pp_warmup False \
                                      --profile True \
                                      --disable_profiler_activity_cpu True \
                                      --use_pytorch_profiler True \
                                      --profile_step_start 5 \
                                      --profile_step_end 6 \
                                      --num_layers 16 \
                                      --num_experts 256 \
                                      --train_iters 10 2>&1 | tee $LOG_FILE

                                    #   --recompute_layer_ids_start $RECOMPUTE_ID_START \
                                    #   --moe_permute_fusion True \
                                    #   --use_turbo_token_dispatcher_fp8_alltoall False \
