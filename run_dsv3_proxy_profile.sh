#!/bin/bash
###############################################################################
# Profiler test: run a TINY 2-layer DSv3 proxy on 1x gfx1250 with the PyTorch
# profiler enabled and shape collection on, so we can inspect the GEMM ops
# (hipBLASLt/rocBLAS kernels + aten::mm/bmm) with their shapes and timings.
#
# Trace is written to: output/proftest_dsv3_2L/torch_profile/rank-0.json.gz
###############################################################################
set -e

export DOCKER_IMAGE=registry-sc-harbor.amd.com/framework/therock-npi@sha256:feba897e2a32a2465b8b296ed2662b2ad6136b5f1cf6f6c2716a3674aafc30f3
SCRIPT_DIR=$(realpath -m "$(dirname "$0")")
export TE_DIR=${TE_DIR:-$(realpath -m "$SCRIPT_DIR/../../mi455/TransformerEngine")}
export TE_WHEEL_DIR=${TE_WHEEL_DIR:-$(realpath -m "$SCRIPT_DIR/../../mi455/dist")}

export NVTE_FUSED_ATTN=1
export NVTE_FUSED_ATTN_CK=0
export NVTE_FUSED_ATTN_AOTRITON=1
export NVTE_USE_CK_GEMM=0
# Set PRIMUS_ATTN_AITER=1 to profile with the gfx1250-patched aiter Triton fused
# flash-attn instead of AOTriton (which falls back to UNFUSED for DSv3 MLA).
export NVTE_FLASH_ATTN=0
export NVTE_FLASH_ATTN_AITER=0
export NVTE_AITER_MLA_FLASH_ATTN=0
export AITER_SRC=""
ATTN_TAG="aotriton"
if [[ "${PRIMUS_ATTN_AITER:-0}" == "1" ]]; then
    echo "[attn] PRIMUS_ATTN_AITER=1 -> using aiter Triton flash-attn (fused MLA)"
    export NVTE_FUSED_ATTN=0
    export NVTE_FUSED_ATTN_AOTRITON=0
    export NVTE_FLASH_ATTN=1
    export NVTE_FLASH_ATTN_AITER=1
    export NVTE_AITER_MLA_FLASH_ATTN=1
    export AITER_SRC="$TE_DIR/3rdparty/aiter"
    ATTN_TAG="aiter"
fi

# TUNED hipBLASLt 1.4 (gfx1250) catalog, LD_PRELOADed (default for DSv3 + Qwen3).
# Verified NaN-free for DSv3 after the MaxSgprPreload rebuild (~2.8x vs stock).
export HIPBLASLT_DIR=${HIPBLASLT_DIR:-$(realpath -m "$SCRIPT_DIR/../../hipblaslt")}
export HIPBLASLT_LD_PRELOAD=$HIPBLASLT_DIR/rocm-libraries/projects/hipblaslt/build/release/library/libhipblaslt.so.1.4
export HIPBLASLT_TENSILE_LIBPATH=$HIPBLASLT_DIR/rocm-libraries/projects/hipblaslt/build/release/Tensile/library/gfx1250

# RCCL workaround: all_reduce(op=AVG) hangs on this gfx1250 build at world_size=1
# (MoE aux-loss metric reduce). sitecustomize rewrites AVG -> SUM/ws.
export PYTHONPATH="$SCRIPT_DIR/rccl_avg_workaround:${PYTHONPATH:-}"

export HSA_NO_SCRATCH_RECLAIM=1
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_HCA=
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export RCCL_DISABLE_AMDSMI=1
export NCCL_AMDSMI_DISABLE=1

export GPUS_PER_NODE=1
export NNODES=1
export PRIMUS_SEQ_LENGTH=4096
export PRIMUS_VPP=1          # PP=1 proxy: disable VPP (interleaving needs PP>1)
export PYTHONUNBUFFERED=1

PRIMUS_PATH=$(realpath "$(dirname "$0")")
DATA_PATH="${PRIMUS_PATH}/data"
mkdir -p "$DATA_PATH"

EXP=examples/megatron/configs/MI355X/deepseek_v3-FP8-pretrain.yaml
LOG=dsv3-proxy-profile-2L-${ATTN_TAG}.log
TB_DIR=output/proftest_dsv3_2L_${ATTN_TAG}/tb

# Tiny 2-layer proxy + PyTorch profiler (capture steps 3-4, record shapes).
PROXY_OVERRIDES="\
    --train_iters 7 \
    --num_layers 2 \
    --micro_batch_size 1 \
    --global_batch_size 8 \
    --tensor_model_parallel_size 1 \
    --pipeline_model_parallel_size 1 \
    --expert_model_parallel_size 1 \
    --enable_primus_turbo=False \
    --num_experts 32 \
    --moe_router_topk 1 \
    --moe_router_group_topk 1 \
    --moe_router_pre_softmax True \
    --moe_layer_freq [1]*2 \
    --mtp_num_layers 0 \
    --use_distributed_optimizer False \
    --use_precision_aware_optimizer False \
    --main_grads_dtype fp32 \
    --exp_avg_dtype fp32 \
    --exp_avg_sq_dtype fp32 \
    --store_param_remainders False \
    --overlap_grad_reduce False \
    --overlap_param_gather False \
    --barrier_with_L1_time False \
    --distributed_backend nccl \
    --profile True \
    --use_pytorch_profiler True \
    --profile_step_start 3 \
    --profile_step_end 5 \
    --pytorch_profiler_collect_shapes True \
    --tensorboard_dir $TB_DIR"

ENV_ARGS=()
for v in DOCKER_IMAGE NVTE_FUSED_ATTN NVTE_FUSED_ATTN_CK NVTE_FUSED_ATTN_AOTRITON \
         NVTE_FLASH_ATTN NVTE_FLASH_ATTN_AITER NVTE_AITER_MLA_FLASH_ATTN AITER_SRC \
         NVTE_USE_CK_GEMM HSA_NO_SCRATCH_RECLAIM NCCL_IB_DISABLE NCCL_P2P_DISABLE \
         NCCL_IB_HCA NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME RCCL_DISABLE_AMDSMI \
         NCCL_AMDSMI_DISABLE GPUS_PER_NODE NNODES PRIMUS_SEQ_LENGTH PRIMUS_VPP \
         PYTHONPATH PYTHONUNBUFFERED TE_DIR TE_WHEEL_DIR HIPBLASLT_DIR HIPBLASLT_TENSILE_LIBPATH; do
    ENV_ARGS+=("--env" "$v")
done

VOLUME_ARGS=(-v "$PRIMUS_PATH":"$PRIMUS_PATH" -v "$DATA_PATH":"$DATA_PATH")
[[ -d "$TE_WHEEL_DIR" ]] && VOLUME_ARGS+=(-v "$TE_WHEEL_DIR":"$TE_WHEEL_DIR")
[[ -d "$TE_DIR" ]] && VOLUME_ARGS+=(-v "$TE_DIR":"$TE_DIR")
[[ -d "$HIPBLASLT_DIR" ]] && VOLUME_ARGS+=(-v "$HIPBLASLT_DIR":"$HIPBLASLT_DIR")

TE_INSTALL_PREFIX="\
    if ls ${TE_WHEEL_DIR}/transformer_engine-*.whl >/dev/null 2>&1; then \
        echo '[TE] installing prebuilt wheel from ${TE_WHEEL_DIR}' && \
        pip install --quiet --force-reinstall --no-deps ${TE_WHEEL_DIR}/transformer_engine-*.whl && \
        pip install --quiet einops nvdlfw-inspect onnxscript onnx pydantic importlib-metadata packaging transformers pybind11; \
    else \
        echo '[TE] WARNING: no TE wheel found; run will likely fail'; \
    fi && \
    echo '[deps] installing Primus requirements (nltk, etc.)' && \
    pip install --quiet -r requirements.txt && "

HBL_PREFIX=""
if [[ -n "${HIPBLASLT_LD_PRELOAD:-}" ]]; then
    HBL_PREFIX="export LD_PRELOAD=${HIPBLASLT_LD_PRELOAD}\${LD_PRELOAD:+:\$LD_PRELOAD} && "
fi

# aiter Triton flash-attn wiring (only when PRIMUS_ATTN_AITER=1 set AITER_SRC):
# install deps, overlay the MLA carve-out onto the installed TE utils.py, and put the
# gfx1250-patched aiter submodule on PYTHONPATH.
AITER_PREFIX=""
if [[ -n "${AITER_SRC:-}" ]]; then
    AITER_PREFIX="pip install --quiet psutil ninja pandas 2>/dev/null && \
        SP=\$(python -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null) && \
        cp ${TE_DIR}/transformer_engine/pytorch/attention/dot_product_attention/utils.py \
           \"\$SP/transformer_engine/pytorch/attention/dot_product_attention/utils.py\" && \
        echo '[TE] patched utils.py (aiter MLA carve-out)' && \
        export PYTHONPATH=${AITER_SRC}\${PYTHONPATH:+:\$PYTHONPATH} && "
fi

docker run --rm \
    "${ENV_ARGS[@]}" \
    --ipc=host --network=host \
    --device=/dev/kfd --device=/dev/dri \
    --cap-add=SYS_PTRACE --cap-add=CAP_SYS_ADMIN \
    --security-opt seccomp=unconfined --group-add video \
    --privileged --device=/dev/infiniband \
    --name primus-projection \
    "${VOLUME_ARGS[@]}" \
    "$DOCKER_IMAGE" /bin/bash -c "\
        set -e && cd $PRIMUS_PATH && \
        ${TE_INSTALL_PREFIX}${HBL_PREFIX}${AITER_PREFIX}\
        echo '==================== PROFILER TEST: 2-layer DSv3 proxy (tuned GEMM, attn=${ATTN_TAG}) ====================' && \
        EXP=$EXP GPUS_PER_NODE=1 NNODES=1 bash examples/run_pretrain.sh \
            ${PROXY_OVERRIDES}" \
    2>&1 | tee "$LOG"
