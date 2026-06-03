#!/bin/bash
###############################################################################
# DSv3 (full model) performance projection: benchmark on 1x gfx1250 GPU,
# project to 64 GPUs / 8 nodes with TP=1, PP=16, EP=4.
#
# Uses the NEW unified Primus CLI (runner/primus-cli) projection tool, which
# benchmarks the model on a single GPU (--benchmark-gpus 1) and analytically
# restores the full PP=16/EP=4 target during projection. We run it with
# --single (one python3 process, NOT torchrun) so GPUS_PER_NODE=8 is used only
# for TARGET node sizing (8 nodes x 8 = 64 GPUs); the projection internally
# spawns its own nproc=1 benchmark subprocess pinned to the single GPU.
#
# The harbor therock-npi image ships no TransformerEngine, so we install the
# prebuilt gfx1250 TE wheel at container start (same as the validated training
# runs).
###############################################################################
set -e

# Validated harbor image (no TE shipped) + prebuilt gfx1250 TE wheel/source.
export DOCKER_IMAGE=registry-sc-harbor.amd.com/framework/therock-npi@sha256:feba897e2a32a2465b8b296ed2662b2ad6136b5f1cf6f6c2716a3674aafc30f3
SCRIPT_DIR=$(realpath -m "$(dirname "$0")")
export TE_DIR=${TE_DIR:-$(realpath -m "$SCRIPT_DIR/../../mi455/TransformerEngine")}
export TE_WHEEL_DIR=${TE_WHEEL_DIR:-$(realpath -m "$SCRIPT_DIR/../../mi455/dist")}

# gfx1250 fused attention: Triton (AOTriton) backend; CK has no gfx1250 support.
export NVTE_FUSED_ATTN=1
export NVTE_FUSED_ATTN_CK=0
export NVTE_FUSED_ATTN_AOTRITON=1
export NVTE_USE_CK_GEMM=0

# Single-node loopback so 1-rank distributed init does not touch IB.
export HSA_NO_SCRATCH_RECLAIM=1
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_HCA=
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export RCCL_DISABLE_AMDSMI=1
export NCCL_AMDSMI_DISABLE=1

# Projection target overrides (read by the projection python from env).
export GPUS_PER_NODE=8        # TARGET node size: 8 nodes x 8 = 64 GPUs
export NNODES=1
export PRIMUS_EP=4            # override EP 8 -> 4 (PP stays 16 from the YAML)
export PRIMUS_SEQ_LENGTH=4096
export PYTHONUNBUFFERED=1

PRIMUS_PATH=$(realpath "$(dirname "$0")")
DATA_PATH="${PRIMUS_PATH}/data"
mkdir -p "$DATA_PATH"

EXP=examples/megatron/configs/MI355X/deepseek_v3-FP8-pretrain.yaml
LOG=dsv3-FP8-projection-pp16ep4-64gpu.log

# Collect env to forward into the container.
ENV_ARGS=()
for v in DOCKER_IMAGE NVTE_FUSED_ATTN NVTE_FUSED_ATTN_CK NVTE_FUSED_ATTN_AOTRITON \
         NVTE_USE_CK_GEMM HSA_NO_SCRATCH_RECLAIM NCCL_IB_DISABLE NCCL_P2P_DISABLE \
         NCCL_IB_HCA NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME RCCL_DISABLE_AMDSMI \
         NCCL_AMDSMI_DISABLE GPUS_PER_NODE NNODES PRIMUS_EP PRIMUS_SEQ_LENGTH \
         PYTHONUNBUFFERED TE_DIR TE_WHEEL_DIR; do
    ENV_ARGS+=("--env" "$v")
done

VOLUME_ARGS=(-v "$PRIMUS_PATH":"$PRIMUS_PATH" -v "$DATA_PATH":"$DATA_PATH")
[[ -d "$TE_WHEEL_DIR" ]] && VOLUME_ARGS+=(-v "$TE_WHEEL_DIR":"$TE_WHEEL_DIR")
[[ -d "$TE_DIR" ]] && VOLUME_ARGS+=(-v "$TE_DIR":"$TE_DIR")

# Install the prebuilt TE wheel into the image (it ships none).
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
        bash runner/primus-cli direct --single -- \
            projection performance \
            --config $EXP \
            --benchmark-gpus 1 \
            --target-nodes 8" \
    2>&1 | tee "$LOG"
