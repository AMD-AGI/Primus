#!/bin/bash
# MLPerf GPT-OSS-20B configuration for MI355X (1x8x1).

export DGXSYSTEM=MI355X_1x8x1
export GPUS_PER_NODE=8
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=29501

export PRIMUS_PATH=/workspace/Primus
export PRIMUS_MLPERF=1
export PYTHONPATH="${PRIMUS_PATH}:${PRIMUS_PATH}/third_party/Megatron-LM:${PYTHONPATH:-}"
# MXFP4 + de-oscillation is the submission recipe. For the FP8 recipe:
#   export EXP=${PRIMUS_PATH}/examples/mlperf/gpt_oss_20b/configs/MI355/gpt_oss_20B-FP8-turbo-attn-mlperf-pretrain.yaml
export EXP="${EXP:-${PRIMUS_PATH}/examples/mlperf/gpt_oss_20b/configs/MI355/gpt_oss_20B-MXFP4-deosc-mlperf-pretrain.yaml}"
export DATA_PATH=/data
if [[ "${EXP}" == *MXFP4* ]]; then
    _linear_precision=mxfp4
else
    _linear_precision=fp8
fi

export PRIMUS_MICRO_BATCH_SIZE=4
export PRIMUS_GLOBAL_BATCH_SIZE=32
# MXFP4 uses the MLPerf 6.1 submission schedule (min LR is 10% of max LR).
# The FP8 yaml still defaults to 8e-4 when PRIMUS_LR is not exported.
if [[ "${_linear_precision}" == mxfp4 ]]; then
    export PRIMUS_LR=5.7e-4
    export PRIMUS_MIN_LR=5.7e-5
    export PRIMUS_LR_WARMUP_ITERS=117
else
    export PRIMUS_LR=4.0e-4
    export PRIMUS_MIN_LR=4.0e-5
    export PRIMUS_LR_WARMUP_ITERS=128
fi
export PRIMUS_TRAIN_ITERS=1200000
export PRIMUS_LR_DECAY_ITERS=$((PRIMUS_TRAIN_ITERS - PRIMUS_LR_WARMUP_ITERS))
export EVAL_ITERS=$((1024 / PRIMUS_GLOBAL_BATCH_SIZE))
export EVAL_SAMPLES_INTERVAL=12288
export PRIMUS_EVAL_INTERVAL=$((EVAL_SAMPLES_INTERVAL / PRIMUS_GLOBAL_BATCH_SIZE))

export PRIMUS_TP=1
export PRIMUS_PP=1
export PRIMUS_EP=1

export PRIMUS_TURBO_FUSED_WGRAD_ACCUM="${PRIMUS_TURBO_FUSED_WGRAD_ACCUM:-1}"
export PRIMUS_NUM_WORKERS="${PRIMUS_NUM_WORKERS:-2}"
export PRIMUS_GRAD_REDUCE_IN_BF16=true
export USE_TURBO_RMS_NORM=true
export USE_TURBO_NORM_TE_LINEAR=true

# E8M0 block-scale rounding: 0=half ULP, 1=one ULP, 2=three-eighths ULP. This is
# not gradient stochastic rounding, which mxfp4_gradient_stochastic_rounding
# controls separately. The MXFP4 recipe is validated at 2. Read by the recipe as
# mxfp4_scale_rounding_mode; Turbo builds from before the UoS work took it as an
# environment variable of the same name instead.
export PRIMUS_TURBO_MXFP4_SCALE_ROUNDING="${PRIMUS_TURBO_MXFP4_SCALE_ROUNDING:-2}"

export HSA_ENABLE_INTERRUPT=0
export HSA_NO_SCRATCH_RECLAIM=1
export HSA_ENABLE_SDMA=1
# With 2 queues the FlyDSL attention-backward side stream shares a hardware
# queue with the RCCL copy-engine ReduceScatter chain and stalls behind it.
export GPU_MAX_HW_QUEUES=4
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HIP_FORCE_DEV_KERNARG=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export HSA_KERNARG_POOL_SIZE=12582912
export PYTORCH_ALLOC_CONF=expandable_segments:False
export ENABLE_NUMA_BINDING=1
export OMP_NUM_THREADS=1
export PYTHONWARNINGS=ignore
export TOKENIZERS_PARALLELISM=false

export NCCL_MIN_P2P_NCHANNELS=32
export NCCL_MIN_CTAS=32
export NCCL_NCHANNELS_PER_NET_PEER=32
export NCCL_NVLS_ENABLE=0
export NCCL_CHECKS_DISABLE=1
export TORCH_NCCL_HIGH_PRIORITY=1

# Route distributed-optimizer parameter AllGather through the dedicated
# zero-CTA RCCL communicator backed by symmetric memory. The eager allocation
# size is the validated GPT-OSS-20B reservation for this 8x MI355X recipe.
export MEGATRON_PARAM_GATHER_BACKEND="${MEGATRON_PARAM_GATHER_BACKEND:-rccl_sdma}"
export MEGATRON_RCCL_SDMA_EAGER_PARAM_BYTES="${MEGATRON_RCCL_SDMA_EAGER_PARAM_BYTES:-40978350080}"

# More than one synthetic warmup step can exceed memory headroom while the
# approximately 41 GB symmetric parameter buffer is resident.
export SYNTH_WARMUP_STEPS="${SYNTH_WARMUP_STEPS:-1}"

export USE_HIPBLASLT=1
export TORCH_BLAS_PREFER_HIPBLASLT=1
if [ -z "${HIPBLASLT_TUNING_OVERRIDE_FILE:-}" ]; then
    _tune="${PRIMUS_PATH}/examples/mlperf/gpt_oss_20b/tune_gemm_results-v26.5.txt"
    if [ -f "${_tune}" ]; then
        export HIPBLASLT_TUNING_OVERRIDE_FILE="${_tune}"
    fi
fi

export NVTE_FLASH_ATTN=0
export NVTE_ROCM_ENABLE_MXFP8=0
export NVTE_USE_CAST_TRANSPOSE_TRITON=0
export NVTE_USE_OPTIMIZED_HIPIFIED_CAST_TRANSPOSE=1
export NVTE_DEBUG=0
export NVTE_DEBUG_LEVEL=0
export NVTE_LOG_FUSED_ATTN_CONFIG=0
export NVTE_LOG_CK_CONFIG=0
export CK_FUSED_ATTN_LOG_CONFIG=0

export LOG_INTERVAL=999999
export ENABLE_MLLOG=1
export MLLOG_TRAIN_LOSS_LOG_FREQ=0
export MLLOG_TARGET_EVAL_LOSS=3.34
export TARGET_EVAL_LOSS=3.34
export MLLOG_OUTPUT_FILE=/results/mlperf_logging.out
export MLLOG_SAVE_TO_FILE=0
export MLLOG_SUBMISSION_BENCHMARK=gpt_oss_20b
export MLLOG_SUBMISSION_DIVISION=closed
export MLLOG_SUBMISSION_ORG=AMD
export MLLOG_SUBMISSION_PLATFORM=MI355X
export MLLOG_TENSOR_PARALLELISM=1
export MLLOG_PIPELINE_PARALLELISM=1
export MLLOG_CONTEXT_PARALLELISM=1
export MLLOG_EXPERT_PARALLELISM=1
export MLLOG_MICRO_BATCH_SIZE=4
MLLOG_CONFIG_FILENAME=$(basename "${BASH_SOURCE[0]}")
export MLLOG_CONFIG_FILENAME
export MLLOG_LOWEST_NUMERICAL_PRECISION_IN_LINEAR="${MLLOG_LOWEST_NUMERICAL_PRECISION_IN_LINEAR:-${_linear_precision}}"
export MLLOG_LOWEST_NUMERICAL_PRECISION_IN_ATTN="${MLLOG_LOWEST_NUMERICAL_PRECISION_IN_ATTN:-bfloat16}"
export MLLOG_LOWEST_NUMERICAL_PRECISION_IN_COMM="${MLLOG_LOWEST_NUMERICAL_PRECISION_IN_COMM:-bfloat16}"

export MOE_SKIP_IDENTITY_SORT=1
export PRIMUS_FUSED_RESIDUAL_NORM=1
export PRIMUS_MOE_SWIGLU_NOCAT=1
export MLLOG_BLOCK_TPUT_LOG=0
export MLPERF_VERBOSE_LOGS=${MLPERF_VERBOSE_LOGS:-0}

if [[ "${_linear_precision}" == mxfp4 ]]; then
    # Cross-layer residual RMSNorm; the layer-input norm also emits the MXFP4
    # operands consumed directly by QKV.
    export PRIMUS_FUSED_RESIDUAL_NORM_V2="${PRIMUS_FUSED_RESIDUAL_NORM_V2:-1}"
    export PRIMUS_FUSED_RMSNORM_MXFP4_QKV="${PRIMUS_FUSED_RMSNORM_MXFP4_QKV:-1}"

    # The expert WGRAD overwrites main_grad (beta=0) and the grad buffer is
    # reset by ownership transfer instead of a memset.
    export PRIMUS_TURBO_GRAD_OWNERSHIP="${PRIMUS_TURBO_GRAD_OWNERSHIP:-1}"
    export PRIMUS_TURBO_WGRAD_ACCUM_OVERWRITE_OUT="${PRIMUS_TURBO_WGRAD_ACCUM_OVERWRITE_OUT:-1}"
    export PRIMUS_TURBO_AUTO_TUNE=0

    # Fused AdamW: gradient clipping, BF16 gradient shards and BF16 weight
    # writeback in one kernel.
    export PRIMUS_FUSED_ADAM_CLIP="${PRIMUS_FUSED_ADAM_CLIP:-1}"
    export PRIMUS_FUSED_ADAM_BF16_GRAD="${PRIMUS_FUSED_ADAM_BF16_GRAD:-1}"
    export PRIMUS_FUSED_ADAM_BF16_WRITEBACK="${PRIMUS_FUSED_ADAM_BF16_WRITEBACK:-1}"
    export PRIMUS_STATIC_BF16_LOSS_SCALE_CPU="${PRIMUS_STATIC_BF16_LOSS_SCALE_CPU:-1}"
    # BF16 Adam moments. The FP32 MCore master is required by de-oscillation.
    export PRIMUS_USE_PRECISION_AWARE_OPTIMIZER="${PRIMUS_USE_PRECISION_AWARE_OPTIMIZER:-true}"
    if [[ "${PRIMUS_USE_PRECISION_AWARE_OPTIMIZER}" == true ]]; then
        export PRIMUS_EXP_AVG_DTYPE="${PRIMUS_EXP_AVG_DTYPE:-bf16}"
        export PRIMUS_EXP_AVG_SQ_DTYPE="${PRIMUS_EXP_AVG_SQ_DTYPE:-bf16}"
        export PRIMUS_FUSED_ADAM_MCORE_MASTER=1
    fi

    # Gradient ReduceScatter on RCCL copy engines. RCCL ignores these unless it
    # has CE ReduceScatter (ROCm/rocm-systems#12310); RCCL 2.30.7 does not.
    export RCCL_CE_REDUCESCATTER=1
    export RCCL_FORCE_CE_REDUCESCATTER=1
    export RCCL_CE_AR_STAGING_BYTES=268435456
    export RCCL_CE_COOP_LAUNCH=0
    export RCCL_CE_REDUCE_PER_CHUNK=1
    export RCCL_CE_REDUCE_MAX_BLOCKS=64
    export RCCL_DDA_ENABLE=1
    export NCCL_CUMEM_ENABLE=1
    export NCCL_LOCAL_REGISTER=0
fi
unset _linear_precision
