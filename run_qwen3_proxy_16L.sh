#!/bin/bash
###############################################################################
# Throughput test (NO profiler): 16-layer Qwen3-235B-A22B proxy on 1x gfx1250.
# Same per-GPU MoE/GEMM shapes as the 2-layer profile proxy (32 experts shard,
# topk=2 — the per-GPU slice of the real EP=4 / global-128 / top-8 layout),
# just 16 layers so transformer-block cost dominates the step.
#
# Uses the TUNED hipBLASLt 1.4 (gfx1250) catalog, LD_PRELOADed into container.
# Profiler is intentionally OFF.
###############################################################################
set -e

export DOCKER_IMAGE=registry-sc-harbor.amd.com/framework/therock-npi@sha256:feba897e2a32a2465b8b296ed2662b2ad6136b5f1cf6f6c2716a3674aafc30f3
SCRIPT_DIR=$(realpath -m "$(dirname "$0")")
export TE_DIR=${TE_DIR:-$(realpath -m "$SCRIPT_DIR/../../mi450/TransformerEngine")}
export TE_WHEEL_DIR=${TE_WHEEL_DIR:-$(realpath -m "$SCRIPT_DIR/../../mi450/dist")}

# Tuned hipBLASLt 1.4 (gfx1250) FP8 GEMM, LD_PRELOADed into the container.
export HIPBLASLT_DIR=${HIPBLASLT_DIR:-$(realpath -m "$SCRIPT_DIR/../../hipblaslt")}
export HIPBLASLT_LD_PRELOAD=$HIPBLASLT_DIR/rocm-libraries/projects/hipblaslt/build/release/library/libhipblaslt.so.1.4
export HIPBLASLT_TENSILE_LIBPATH=$HIPBLASLT_DIR/rocm-libraries/projects/hipblaslt/build/release/Tensile/library/gfx1250

# Attention backend. Default: AOTriton fused (falls back to UNFUSED on gfx1250 even
# for Qwen3's equal 128/128 dims -> exposed softmax). Set PRIMUS_ATTN_AITER=1 to use
# the gfx1250-patched aiter Triton flash-attn (Qwen3 is GQA 64/4, equal dims = the
# standard flash case, pe=0; no MLA carve-out needed).
export NVTE_FUSED_ATTN=1
export NVTE_FUSED_ATTN_CK=0
export NVTE_FUSED_ATTN_AOTRITON=1
export NVTE_USE_CK_GEMM=0
export NVTE_FLASH_ATTN=0
export NVTE_FLASH_ATTN_AITER=0
export NVTE_AITER_MLA_FLASH_ATTN=0
export AITER_SRC=""
if [[ "${PRIMUS_ATTN_AITER:-0}" == "1" ]]; then
    echo "[attn] PRIMUS_ATTN_AITER=1 -> using aiter Triton flash-attn"
    export NVTE_FUSED_ATTN=0
    export NVTE_FUSED_ATTN_AOTRITON=0
    export NVTE_FLASH_ATTN=1
    export NVTE_FLASH_ATTN_AITER=1
    export NVTE_AITER_MLA_FLASH_ATTN=1
    export AITER_SRC="$TE_DIR/3rdparty/aiter"
fi

export HSA_NO_SCRATCH_RECLAIM=1
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_HCA=
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export RCCL_DISABLE_AMDSMI=1
export NCCL_AMDSMI_DISABLE=1

# RCCL workaround: all_reduce(op=AVG) hangs on this gfx1250 build (MoE aux-loss /
# expert-bias reduce); sitecustomize on PYTHONPATH rewrites AVG -> SUM/world_size.
# (The aiter prefix below prepends AITER_SRC, keeping this on the path too.)
export PYTHONPATH="$SCRIPT_DIR/rccl_avg_workaround:${PYTHONPATH:-}"

export GPUS_PER_NODE=1
export NNODES=1
export PRIMUS_SEQ_LENGTH=4096
export PRIMUS_PP=1
export PRIMUS_EP=1
export PYTHONUNBUFFERED=1

PRIMUS_PATH=$(realpath "$(dirname "$0")")
DATA_PATH="${PRIMUS_PATH}/data"
mkdir -p "$DATA_PATH"

EXP=examples/megatron/configs/MI355X/qwen3_235B_A22B-FP8-pretrain.yaml
LOG=qwen3-proxy-16L.log

# 16-layer Qwen3 proxy, NO profiler.
# 32 experts = the per-GPU shard under the real EP=4 layout (128 experts / 4).
# topk=2 = local activations per token (global top-8 of 128 over EP=4: 8*32/128=2).
PROXY_OVERRIDES="\
    --train_iters 10 \
    --num_layers 16 \
    --micro_batch_size 1 \
    --global_batch_size 8 \
    --tensor_model_parallel_size 1 \
    --pipeline_model_parallel_size 1 \
    --expert_model_parallel_size 1 \
    --num_experts 32 \
    --moe_router_topk 2 \
    --enable_primus_turbo=False \
    --barrier_with_L1_time False \
    --distributed_backend nccl"

ENV_ARGS=()
for v in DOCKER_IMAGE NVTE_FUSED_ATTN NVTE_FUSED_ATTN_CK NVTE_FUSED_ATTN_AOTRITON \
         NVTE_FLASH_ATTN NVTE_FLASH_ATTN_AITER NVTE_AITER_MLA_FLASH_ATTN AITER_SRC \
         NVTE_USE_CK_GEMM PYTHONPATH HSA_NO_SCRATCH_RECLAIM NCCL_IB_DISABLE NCCL_P2P_DISABLE \
         NCCL_IB_HCA NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME RCCL_DISABLE_AMDSMI \
         NCCL_AMDSMI_DISABLE GPUS_PER_NODE NNODES PRIMUS_SEQ_LENGTH PRIMUS_PP PRIMUS_EP \
         PYTHONUNBUFFERED TE_DIR TE_WHEEL_DIR HIPBLASLT_DIR HIPBLASLT_TENSILE_LIBPATH; do
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
    --name primus-proxy-qwen3-16l \
    "${VOLUME_ARGS[@]}" \
    "$DOCKER_IMAGE" /bin/bash -c "\
        set -e && cd $PRIMUS_PATH && \
        ${TE_INSTALL_PREFIX}${HBL_PREFIX}${AITER_PREFIX}\
        echo '==================== THROUGHPUT TEST: 16-layer Qwen3 proxy (no profiler) ====================' && \
        EXP=$EXP GPUS_PER_NODE=1 NNODES=1 bash examples/run_pretrain.sh \
            ${PROXY_OVERRIDES}" \
    2>&1 | tee "$LOG"
