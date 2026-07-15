#!/bin/bash
# MLPerf 6.0 environment for Llama2-70B LoRA on MI355X (8 GPUs, 1 node).
# Source before run_and_time.sh or primus-cli direct train posttrain.

export DGXSYSTEM=MI355X_1x8x1
export GPUS_PER_NODE=8
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=29502

# MLPerf timed runs set SEED=$RANDOM per trial; default here for single-shot primus-cli.
export SEED="${SEED:-$RANDOM}"

export PRIMUS_PATH="${PRIMUS_PATH:-/workspace/Primus}"
export PYTHONPATH="${PRIMUS_PATH}:${PRIMUS_PATH}/third_party/Megatron-Bridge:${PYTHONPATH:-}"
export EXP="${EXP:-${PRIMUS_PATH}/examples/mlperf/llama2_70b/configs/MI355X/llama2_70b_lora_mlperf_posttrain.yaml}"
export DATA_PATH="${DATA_PATH:-/data}"

export PACKED_TRAIN_DATA_PATH="${DATA_PATH}/train.npy"
export PACKED_VAL_DATA_PATH="${DATA_PATH}/validation.npy"
export PACKED_METADATA_PATH="${DATA_PATH}/packed_metadata.jsonl"
export PACKED_DATA_DIR="${DATA_PATH}"

export PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-/data/megatron_checkpoints/Llama-2-70b-hf}"

export LR=0.0006

export HSA_NO_SCRATCH_RECLAIM=1
export HSA_ENABLE_SDMA=1
export HSA_ENABLE_INTERRUPT=0
export GPU_MAX_HW_QUEUES=2
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_HIGH_PRIORITY=1
export NCCL_CHECKS_DISABLE=1
# Single-node: use GPU P2P, not ionic IB. Broken libibverbs ABI (ionic kernel
# mismatch) commonly hangs NCCL here with 0% GPU util. MLPerf systems with
# working RDMA can override: NCCL_IB_DISABLE=0
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
export OMP_NUM_THREADS=1

export NVTE_USE_AITER_ROPE=1
# Fused attention for MXFP4 training (must match recipe attention_backend=fused).
export NVTE_FLASH_ATTN=0
export NVTE_FUSED_ATTN=1
export NVTE_UNFUSED_ATTN=0
export NVTE_FUSED_ATTN_CK=1
export NVTE_FUSED_ATTN_AOTRITON=1
export NVTE_CK_USES_FWD_V3=1
export NVTE_CK_USES_BWD_V3=1
export NVTE_CK_IS_V3_ATOMIC_FP32=0
export NVTE_RS_STRIDED_ATOMIC=2
export NVTE_FP8_DPA_BWD=1
export NVTE_USE_HIPBLASLT=1
export NVTE_USE_CAST_TRANSPOSE_TRITON=0
export NVTE_USE_OPTIMIZED_HIPIFIED_CAST_TRANSPOSE=0
export NVTE_USE_RMSNORM_TRITON=0
export USE_TE_SWIGLU=1
export ENABLE_TRANSPOSE_CACHE=0
export NVTE_MXFP4_USE_HADAMARD=${NVTE_MXFP4_USE_HADAMARD:-1}
export NVTE_DEBUG=0
export NVTE_DEBUG_LEVEL=0

export HEALING_ITER=${HEALING_ITER:-340}
export HEALING_PRECISION=${HEALING_PRECISION:-FP8_DS}
export PRE_QUANTIZED_MODEL=${PRE_QUANTIZED_MODEL:-True}
export NCCL_MIN_P2P_NCHANNELS=${NCCL_MIN_P2P_NCHANNELS:-32}
export NCCL_MIN_CTAS=${NCCL_MIN_CTAS:-32}
export NCCL_NCHANNELS_PER_NET_PEER=${NCCL_NCHANNELS_PER_NET_PEER:-32}
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}

export MEGATRON_BRIDGE_LOGGING_LEVEL=50
export PYTHONWARNINGS=ignore
export PRIMUS_LOG_LEVEL=ERROR

# Print rank-0 GPU memory (allocated/reserved/peak + torch memory_stats) every log_interval.
export PRIMUS_LOG_GPU_MEM=${PRIMUS_LOG_GPU_MEM:-1}
# Megatron iteration / TFLOP / loss lines use print_rank_0 (always on). Primus log_rank_0
# helpers need this for recipe-internal banners during bring-up.
export VERBOSE_TRAINING_LOG=${VERBOSE_TRAINING_LOG:-1}

export SYNTH_WARMUP_STEPS=5
export SYNTH_WARMUP_VALID_STEPS=5

export ENABLE_MLLOG=1
export MLLOG_OUTPUT_FILE=/results/mlperf_logging.out
export MLLOG_TRAIN_LOSS_LOG_FREQ=0
export MLLOG_TARGET_EVAL_LOSS=0.925
export MLLOG_SUBMISSION_BENCHMARK=llama2_70b_lora
export MLLOG_SUBMISSION_DIVISION=closed
export MLLOG_SUBMISSION_ORG=AMD
export MLLOG_SUBMISSION_PLATFORM=MI355X

export MLLOG_TENSOR_PARALLELISM=1
export MLLOG_PIPELINE_PARALLELISM=1
export MLLOG_CONTEXT_PARALLELISM=1
export MLLOG_EXPERT_PARALLELISM=1
export MLLOG_MICRO_BATCH_SIZE=1
export MLLOG_CONFIG_FILENAME=$(basename "${BASH_SOURCE[0]}")
export MLLOG_LOWEST_NUMERICAL_PRECISION_LINEAR=mxfp4

export TP_COMM_OVERLAP=False
export MC_TP_OVERLAP_AG=False
export MC_TP_OVERLAP_RS=False
export MC_TP_OVERLAP_RS_DGRAD=False

export CUBLAS_FORCE_XMMA_KERNEL_INIT=DEVICE

export LORA_A2A=1
export POSSIBLE_USER_WARNINGS=0
export CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT=0

export TP=1
export PP=1
export CP=1
export SP=False
export VBOOST_VALUE=1
export MBS=1
export MINIBS=1
export SKIP_EVALS=3
export VAL_CHECK_INTERVAL=384
export HYDRA_FULL_ERROR=1

export FP8_DPA=0
# FP8 env flags below apply to healing (HEALING_ITER=340) and TE delayed scaling,
# not Megatron model_cfg.fp8 during the MXFP4 phase (recipe sets fp8=None, fp4=mxfp4).
export FP8=True
export FP8_AMAX_ALGO=most_recent
export FP8_REDUCE_AMAX=False
export FP8_AMAX_HISTORY=4
export FP8_ACTIVATION=True

export FUSED_SOFTMAX=0
export RMSNORM_CAST=0

export PT_TENSOR_VALIDATION=0
export PROFILE_RPD=0

export USE_HIPBLASLT=1
export TORCH_BLAS_PREFER_HIPBLASLT=1

export LOGGING_INTERVAL=5000
# FP4 weights (MXFP4 e2m1 linear layers); distinct from FP8_* healing flags above.
export FP4=True
export FP4_RECIPE=mxfp4
export MAX_STEPS=550
export NEXP=1

export LOAD_CKPT=True
export MCORE_CUDA_GRAPH=False
export RESET_CG_AFTER_HEALING=False

export RECOMPUTE_GRANULARITY=null
export RECOMPUTE_METHOD=null
export RECOMPUTE_NUM_LAYERS=null

export FP8_ACT=0
export AITER_CONFIG_GEMM_A4W4="${PRIMUS_PATH}/examples/mlperf/llama2_70b/a4w4_tuned_gemms.csv"
export AITER_LOG_TUNED_CONFIG=0
export NVTE_FP4_LOG_GEMM_SHAPES=0
export AITER_LOG_LEVEL=ERROR
export AITER_LOG_MORE=0
