#!/bin/bash
###############################################################################
# DSv3 projection VALIDATION on a single gfx1250 GPU.
#
# Goal: check whether the Primus performance-projection engine reproduces a
# REAL training run on the SAME hardware. We do this on a small proxy DSv3
# config that actually fits on 1 GPU (4 layers, 32 experts, topk=1, PP1/EP1,
# GBS=8) — identical to the validated mi455 proxy run.
#
# In ONE container we:
#   1) install the prebuilt gfx1250 TE wheel (image ships none),
#   2) run the REAL pretrain (10 iters) and record tokens/s/GPU + TFLOP/s/GPU,
#   3) run the PROJECTION with the SAME proxy overrides, target = 1 GPU,
#      and record projected tokens/s/GPU + iteration time.
# Then we compare measured vs projected.
###############################################################################
set -e

export DOCKER_IMAGE=registry-sc-harbor.amd.com/framework/therock-npi@sha256:feba897e2a32a2465b8b296ed2662b2ad6136b5f1cf6f6c2716a3674aafc30f3
SCRIPT_DIR=$(realpath -m "$(dirname "$0")")
export TE_DIR=${TE_DIR:-$(realpath -m "$SCRIPT_DIR/../../mi455/TransformerEngine")}
export TE_WHEEL_DIR=${TE_WHEEL_DIR:-$(realpath -m "$SCRIPT_DIR/../../mi455/dist")}

# DSv3 uses STOCK hipBLASLt (tuned 1.4 is forward-only and Infs DSv3 backward).
export NVTE_FUSED_ATTN=1
export NVTE_FUSED_ATTN_CK=0
export NVTE_FUSED_ATTN_AOTRITON=1
export NVTE_USE_CK_GEMM=0

export HSA_NO_SCRATCH_RECLAIM=1
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_HCA=
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export RCCL_DISABLE_AMDSMI=1
export NCCL_AMDSMI_DISABLE=1

# Single GPU for BOTH the real run and the projection target.
export GPUS_PER_NODE=1
export NNODES=1
export PRIMUS_SEQ_LENGTH=4096
export PYTHONUNBUFFERED=1

PRIMUS_PATH=$(realpath "$(dirname "$0")")
DATA_PATH="${PRIMUS_PATH}/data"
mkdir -p "$DATA_PATH"

EXP=examples/megatron/configs/MI355X/deepseek_v3-FP8-pretrain.yaml
LOG=dsv3-proxy-validate.log

# Identical proxy overrides for the real run and the projection.
PROXY_OVERRIDES="\
    --num_layers 4 \
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
    --moe_layer_freq [1]*4 \
    --mtp_num_layers 0"

ENV_ARGS=()
for v in DOCKER_IMAGE NVTE_FUSED_ATTN NVTE_FUSED_ATTN_CK NVTE_FUSED_ATTN_AOTRITON \
         NVTE_USE_CK_GEMM HSA_NO_SCRATCH_RECLAIM NCCL_IB_DISABLE NCCL_P2P_DISABLE \
         NCCL_IB_HCA NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME RCCL_DISABLE_AMDSMI \
         NCCL_AMDSMI_DISABLE GPUS_PER_NODE NNODES PRIMUS_SEQ_LENGTH \
         PYTHONUNBUFFERED TE_DIR TE_WHEEL_DIR; do
    ENV_ARGS+=("--env" "$v")
done

VOLUME_ARGS=(-v "$PRIMUS_PATH":"$PRIMUS_PATH" -v "$DATA_PATH":"$DATA_PATH")
[[ -d "$TE_WHEEL_DIR" ]] && VOLUME_ARGS+=(-v "$TE_WHEEL_DIR":"$TE_WHEEL_DIR")
[[ -d "$TE_DIR" ]] && VOLUME_ARGS+=(-v "$TE_DIR":"$TE_DIR")

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
        ${TE_INSTALL_PREFIX}\
        echo '==================== REAL PROXY TRAINING ====================' && \
        EXP=$EXP GPUS_PER_NODE=1 NNODES=1 bash examples/run_pretrain.sh \
            --train_iters 10 \
            --distributed_backend gloo \
            ${PROXY_OVERRIDES} ; \
        echo '==================== PROJECTION (same proxy cfg) ====================' && \
        bash runner/primus-cli direct --single -- \
            projection performance \
            --config $EXP \
            --benchmark-gpus 1 \
            --target-nodes 1 \
            ${PROXY_OVERRIDES}" \
    2>&1 | tee "$LOG"
