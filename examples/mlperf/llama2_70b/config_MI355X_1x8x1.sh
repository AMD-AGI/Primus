#!/bin/bash
# MLPerf 6.0 environment for Llama2-70B LoRA on MI355X (8 GPUs, 1 node).
# Source before run_and_time.sh, then from ${PRIMUS_PATH}:
#   ./primus-cli direct --log_file /results/logs/log_*.txt -- train posttrain --config "${EXP}"

export DGXSYSTEM=MI355X_1x8x1
# Used by runner/helpers/envs/primus-env.sh when rocm-smi is not in PATH (e.g. minimal Docker exec).
export PRIMUS_GPU_MODEL="${PRIMUS_GPU_MODEL:-MI355X}"
export GPUS_PER_NODE=8
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=29502

# MLPerf timed runs set SEED=$RANDOM per trial; default here for single-shot primus-cli.
export SEED="${SEED:-$RANDOM}"

export PRIMUS_PATH="${PRIMUS_PATH:-/workspace/Primus}"
# The image ships its own primus_turbo, which does not have lora_fp4_frozen.
# The mounted checkout must come first so that module resolves from here.
export PRIMUS_TURBO_PATH="${PRIMUS_TURBO_PATH:-/workspace/Primus-Turbo}"
export PYTHONPATH="${PRIMUS_TURBO_PATH}:${PRIMUS_PATH}:${PRIMUS_PATH}/third_party/Megatron-Bridge:${PYTHONPATH:-}"
export EXP="${EXP:-${PRIMUS_PATH}/examples/mlperf/llama2_70b/configs/MI355X/llama2_70b_lora_mlperf_posttrain.yaml}"
export DATA_PATH="${DATA_PATH:-/data}"

export PACKED_TRAIN_DATA_PATH="${DATA_PATH}/train.npy"
export PACKED_VAL_DATA_PATH="${DATA_PATH}/validation.npy"
export PACKED_METADATA_PATH="${DATA_PATH}/packed_metadata.jsonl"
export PACKED_DATA_DIR="${DATA_PATH}"

export PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-/data/megatron_checkpoints/Llama-2-70b-hf}"

export LR=0.0006

# Primus-Turbo FlyDSL path: frozen MXFP4 GEMMs with the LoRA adapters folded into the
# base GEMM epilogue, FlyDSL THD attention and fused QKV RoPE, fused dense MLP, fused
# residual RMSNorm, and FlyDSL tensorwise FP8 GEMMs after healing. Set
# ENABLE_PRIMUS_TURBO=false PRIMUS_USE_TURBO_GEMM=false PRIMUS_USE_TURBO_ATTENTION=false
# to fall back to the TransformerEngine MXFP4 path.
export ENABLE_PRIMUS_TURBO="${ENABLE_PRIMUS_TURBO:-true}"
export PRIMUS_USE_TURBO_GEMM="${PRIMUS_USE_TURBO_GEMM:-true}"
export PRIMUS_TURBO_GEMM_BACKEND="${PRIMUS_TURBO_GEMM_BACKEND:-flydsl}"
export PRIMUS_TURBO_AUTO_TUNE="${PRIMUS_TURBO_AUTO_TUNE:-0}"

export PRIMUS_USE_TURBO_ATTENTION="${PRIMUS_USE_TURBO_ATTENTION:-true}"
export PRIMUS_TURBO_ATTN_BACKEND="${PRIMUS_TURBO_ATTN_BACKEND:-FLYDSL}"
export PRIMUS_TURBO_FLYDSL_ROPE="${PRIMUS_TURBO_FLYDSL_ROPE:-1}"
export PRIMUS_ATTENTION_BACKEND="${PRIMUS_ATTENTION_BACKEND:-fused}"
export PRIMUS_TURBO_ATTENTION_FP8="${PRIMUS_TURBO_ATTENTION_FP8:-false}"
export PRIMUS_USE_TURBO_RMS_NORM="${PRIMUS_USE_TURBO_RMS_NORM:-false}"

export PRIMUS_TURBO_FP4_AUTOCAST="${PRIMUS_TURBO_FP4_AUTOCAST:-1}"
export PRIMUS_TURBO_FP4_WEIGHT_CACHE="${PRIMUS_TURBO_FP4_WEIGHT_CACHE:-1}"
export PRIMUS_TURBO_FP4_WEIGHTS="${PRIMUS_TURBO_FP4_WEIGHTS:-1}"
export PRIMUS_TURBO_FP4_FROZEN_WEIGHT="${PRIMUS_TURBO_FP4_FROZEN_WEIGHT:-1}"
export PRIMUS_TURBO_FUSED_DENSE_MLP="${PRIMUS_TURBO_FUSED_DENSE_MLP:-1}"
export PRIMUS_TURBO_FP8_HEALING="${PRIMUS_TURBO_FP8_HEALING:-1}"
export PRIMUS_FUSED_RESIDUAL_NORM="${PRIMUS_FUSED_RESIDUAL_NORM:-1}"
export PRIMUS_FUSED_RESIDUAL_NORM_V2="${PRIMUS_FUSED_RESIDUAL_NORM_V2:-1}"
export PRIMUS_TURBO_LORA_FP4_FUSED="${PRIMUS_TURBO_LORA_FP4_FUSED:-1}"
export LORA_FP4_PARITY="${LORA_FP4_PARITY:-0}"
export LORA_FP4_PARITY_EVERY="${LORA_FP4_PARITY_EVERY:-1}"
export LORA_FP4_PARITY_BWD_EVERY="${LORA_FP4_PARITY_BWD_EVERY:-40}"
export LORA_FP4_FUSED_SCOPE="${LORA_FP4_FUSED_SCOPE:-all}"
export LORA_FP4_PIN_CFG="${LORA_FP4_PIN_CFG:-0}"
export PRIMUS_TURBO_ATTN_DETERMINISTIC="${PRIMUS_TURBO_ATTN_DETERMINISTIC:-0}"
export MXFP4_GRADIENT_SR="${MXFP4_GRADIENT_SR:-false}"
export TURBO_DEBUG="${TURBO_DEBUG:-0}"
export TORCHRUN_RANK_LOGS="${TORCHRUN_RANK_LOGS:-0}"
export HEALING_TIMING="${HEALING_TIMING:-0}"
export HEALING_KEEP_CACHE="${HEALING_KEEP_CACHE:-1}"
export PRIMUS_TURBO_FP4_DROP_TE_WEIGHT="${PRIMUS_TURBO_FP4_DROP_TE_WEIGHT:-0}"
export HEALING_PREFETCH_GB="${HEALING_PREFETCH_GB:-0}"
export HEALING_PREFETCH_STEPS="${HEALING_PREFETCH_STEPS:-4}"
export WARMUP_KEEP_CACHE="${WARMUP_KEEP_CACHE:-1}"

# RCCL disables cuMem when uname reports a kernel older than 6.8. Lux reports
# RHEL 5.14 even though amdgpu supports cuMem, so the container preloads
# libfakeuname.so (LUX_FAKE_KERNEL=6.8.0-lux) and RCCL then enables cuMem.
# With cuMem, the parameter all-gather and the gradient reduce-scatter use
# the GPU copy engines, matching the submission config.
export MEGATRON_PARAM_GATHER_BACKEND="${MEGATRON_PARAM_GATHER_BACKEND:-rccl_sdma}"
export MEGATRON_RCCL_SDMA_LOG="${MEGATRON_RCCL_SDMA_LOG:-1}"

export RCCL_CE_REDUCESCATTER=1
export RCCL_FORCE_CE_REDUCESCATTER=1
export RCCL_CE_REDUCE_PER_CHUNK=1
export RCCL_CE_REDUCE_MAX_BLOCKS=92
export RCCL_CE_REDUCE_NORMAL_PRIORITY=0
export RCCL_CE_AR_STAGING_BYTES=268435456
export RCCL_CE_COOP_LAUNCH=0
export RCCL_DDA_ENABLE=1
export NCCL_CUMEM_ENABLE=1
export NCCL_LOCAL_REGISTER=0
export HIP_FORCE_DEV_KERNARG=1
export HSA_ENABLE_SDMA_OPTIMIZATIONS=1
export PYTORCH_ROC_ALLOC_CONF=expandable_segments:True
# cuMem P2P exchanges one fd per mapped segment through the RCCL proxy. With
# the 70B weights resident, a 1024 soft nofile limit fails the zero-CTA group
# barrier with "Too many open files" / ncclProxyClientGetFd ... failed : 2.
ulimit -Sn "$(ulimit -Hn)" 2>/dev/null || true

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
export NVTE_MXFP4_USE_HADAMARD=${NVTE_MXFP4_USE_HADAMARD:-0}
export NVTE_DEBUG=0
export NVTE_DEBUG_LEVEL=0

export HEALING_ITER=${HEALING_ITER:-360}
export HEALING_PRECISION=${HEALING_PRECISION:-FP8_DS}
export PRE_QUANTIZED_MODEL=${PRE_QUANTIZED_MODEL:-True}
export NCCL_MIN_P2P_NCHANNELS=${NCCL_MIN_P2P_NCHANNELS:-32}
export NCCL_MIN_CTAS=${NCCL_MIN_CTAS:-32}
export NCCL_NCHANNELS_PER_NET_PEER=${NCCL_NCHANNELS_PER_NET_PEER:-32}
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
export NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-5}
export NCCL_SINGLE_RING_THRESHOLD=${NCCL_SINGLE_RING_THRESHOLD:-0}
export NCCL_BUFFSIZE=${NCCL_BUFFSIZE:-2097152}

export MEGATRON_BRIDGE_LOGGING_LEVEL=50
export PYTHONWARNINGS=ignore
export PRIMUS_LOG_LEVEL=ERROR

# Submission-style logging: suppress framework noise; emit :::MLLOG on stdout.
# Override for bring-up/debug: VERBOSE_TRAINING_LOG=1 PRIMUS_LOG_GPU_MEM=1 MLPERF_VERBOSE_LOGS=1
export SUBMISSION_QUIET=1
export PRIMUS_LOG_SUPPRESSION=1
export MLPERF_VERBOSE_LOGS=0
export VERBOSE_TRAINING_LOG=0
export PRIMUS_LOG_GPU_MEM=0

export SYNTH_WARMUP_STEPS=1
export SYNTH_WARMUP_VALID_STEPS=5

export ENABLE_MLLOG=1
export MLLOG_OUTPUT_FILE=/results/mlperf_logging.out
export MLLOG_SAVE_TO_FILE=0
export MLLOG_TRAIN_LOSS_LOG_FREQ=${MLLOG_TRAIN_LOSS_LOG_FREQ:-0}
export MLLOG_TARGET_EVAL_LOSS=0.925
export TARGET_EVAL_LOSS=0.925
export MLLOG_SUBMISSION_BENCHMARK=llama2_70b_lora
export MLLOG_SUBMISSION_DIVISION=closed
export MLLOG_SUBMISSION_ORG=AMD
export MLLOG_SUBMISSION_PLATFORM=MI355X

export MLLOG_TENSOR_PARALLELISM=1
export MLLOG_PIPELINE_PARALLELISM=1
export MLLOG_CONTEXT_PARALLELISM=1
export MLLOG_EXPERT_PARALLELISM=1
export MLLOG_MICRO_BATCH_SIZE=1
MLLOG_CONFIG_FILENAME=$(basename "${BASH_SOURCE[0]}")
export MLLOG_CONFIG_FILENAME
export MLLOG_LOWEST_NUMERICAL_PRECISION_IN_LINEAR=mxfp4
export MLLOG_LOWEST_NUMERICAL_PRECISION_IN_ATTN=bfloat16
export MLLOG_LOWEST_NUMERICAL_PRECISION_IN_COMM=bfloat16

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
export LIMIT_VAL_BATCHES=1.0
export HYDRA_FULL_ERROR=1

export FP8_DPA=0
# FP8 env flags below apply to healing (HEALING_ITER) and TE delayed scaling,
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

export LOGGING_INTERVAL=${LOGGING_INTERVAL:-5000}
# FP4 weights (MXFP4 e2m1 linear layers); distinct from FP8_* healing flags above.
export FP4=True
export FP4_RECIPE=mxfp4
export MAX_STEPS=${MAX_STEPS:-550}
export PRIMUS_TRAIN_ITERS=${PRIMUS_TRAIN_ITERS:-550}
export PRIMUS_EVAL_INTERVAL=${PRIMUS_EVAL_INTERVAL:-48}
export NEXP=${NEXP:-1}

export LOAD_CKPT=True
export MCORE_CUDA_GRAPH=${MCORE_CUDA_GRAPH:-False}
export CUDA_GRAPH_IMPL=${CUDA_GRAPH_IMPL:-local}
export CUDA_GRAPH_SCOPE=${CUDA_GRAPH_SCOPE:-full_iteration}
export CUDA_GRAPH_WARMUP_STEPS="${CUDA_GRAPH_WARMUP_STEPS:-1}"
export RESET_CG_AFTER_HEALING="${RESET_CG_AFTER_HEALING:-False}"

export RECOMPUTE_GRANULARITY=null
export RECOMPUTE_METHOD=null
export RECOMPUTE_NUM_LAYERS=null

export FP8_ACT=0
export AITER_CONFIG_GEMM_A4W4="${PRIMUS_PATH}/examples/mlperf/llama2_70b/a4w4_tuned_gemms.csv"
export AITER_LOG_TUNED_CONFIG=0
export NVTE_FP4_LOG_GEMM_SHAPES=${NVTE_FP4_LOG_GEMM_SHAPES:-0}
export AITER_LOG_LEVEL=ERROR
export AITER_LOG_MORE=0
