#!/bin/bash
# Script to setup and start llama2_70b_lora training
# This script automates all the steps required to start training

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

echo_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Configuration variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# This repo on the host (e.g. /home/kgoginen/up_primus/Primus)
PRIMUS_ROOT="${SCRIPT_DIR}"
MEGATRON_BRIDGE_HOST="${PRIMUS_ROOT}/third_party/Megatron-Bridge"
PATCHES_HOST="${PRIMUS_ROOT}/primus/recipes/patches"
# start_container.sh mounts PRIMUS_PATH at this path inside the container
CONTAINER_PRIMUS_ROOT="${CONTAINER_PRIMUS_ROOT:-/workspace/Primus}"
MEGATRON_BRIDGE_CONTAINER="${CONTAINER_PRIMUS_ROOT}/third_party/Megatron-Bridge"
PATCH_DIR="${CONTAINER_PRIMUS_ROOT}/primus/recipes/patches"

# Data + optional Megatron checkpoints: under this repo by default (override with HOST_MLPERF_DATA).
# Same host path is mounted into the container as MOUNT_DATA_PATH (default /data); YAML uses
# PACKED_* / PRETRAINED_CHECKPOINT env vars resolved at config load time.
HOST_MLPERF_DATA="${HOST_MLPERF_DATA:-${PRIMUS_ROOT}/data/mlperf_llama2}"
# Typical layout (optional): ${HOST_MLPERF_DATA}/megatron_checkpoints/<model>/iter_0000000
MEGATRON_CKPT_SUBDIR="${MEGATRON_CKPT_SUBDIR:-megatron_checkpoints}"
SEQ_LENGTH=8192
HF_TOKEN="${HF_TOKEN:-}"
if [ -z "$HF_TOKEN" ]; then
    echo -e "${RED}[ERROR]${NC} HF_TOKEN is not set or is empty. Please set your Hugging Face token in the environment or in this script."
    exit 1
fi
WANDB_API_KEY="${WANDB_API_KEY:-}"
if [ -z "$WANDB_API_KEY" ]; then
    echo -e "${RED}[WARNING]${NC} WANDB_API_KEY is not set or is empty."
fi

# Docker configuration (for start_container.sh: -v DATA_PATH:MOUNT_DATA_PATH)
export DOCKER_IMAGE="${DOCKER_IMAGE:-rocm/primus:v26.2}"
export DATA_PATH="${HOST_MLPERF_DATA}"
export MOUNT_DATA_PATH="${MOUNT_DATA_PATH:-/data}"
export CONTAINER_NAME="${CONTAINER_NAME:-llama2_lora_26_2_primus_upstream}"

# Paths inside the container (must match MOUNT_DATA_PATH mount of HOST_MLPERF_DATA)
PACKED_TRAIN_DATA_PATH="${MOUNT_DATA_PATH}/train.npy"
PACKED_VAL_DATA_PATH="${MOUNT_DATA_PATH}/validation.npy"
PACKED_METADATA_PATH="${MOUNT_DATA_PATH}/packed_metadata.jsonl"

# Optional Megatron pretrained dir on host -> container path for PEFT base weights
PRETRAINED_CHECKPOINT=""
if [ -d "${HOST_MLPERF_DATA}/${MEGATRON_CKPT_SUBDIR}" ]; then
    _iter_dir="$(find "${HOST_MLPERF_DATA}/${MEGATRON_CKPT_SUBDIR}" -type d -name 'iter_*' 2>/dev/null | sort | head -n 1)"
    if [ -n "${_iter_dir}" ]; then
        PRETRAINED_CHECKPOINT="${MOUNT_DATA_PATH}${_iter_dir#"${HOST_MLPERF_DATA}"}"
    fi
fi

# Step 1: Update submodules
echo_info "Step 1: Updating submodules..."
git submodule update --init --recursive
echo_info "Submodules updated successfully"

# Step 2: Start Primus Docker container with data mount
echo_info "Step 2: Starting Primus Docker container..."
echo_info "Primus repo (host): ${PRIMUS_ROOT}"
echo_info "Data / checkpoints root (host): ${HOST_MLPERF_DATA}"
echo_info "Container mount: ${HOST_MLPERF_DATA} -> ${MOUNT_DATA_PATH}"
echo_info "Packed data paths (container): ${PACKED_TRAIN_DATA_PATH}, ${PACKED_VAL_DATA_PATH}, ${PACKED_METADATA_PATH}"
if [ -n "${PRETRAINED_CHECKPOINT}" ]; then
    echo_info "Megatron pretrained_checkpoint (container): ${PRETRAINED_CHECKPOINT}"
else
    echo_info "No iter_* under ${HOST_MLPERF_DATA}/${MEGATRON_CKPT_SUBDIR}; set pretrained_checkpoint in the posttrain yaml or add Megatron weights under that tree."
fi
echo_info "Container name: ${CONTAINER_NAME}"
echo_info "Docker image: ${DOCKER_IMAGE}"

mkdir -p "${HOST_MLPERF_DATA}"
mkdir -p "${HOST_MLPERF_DATA}/${MEGATRON_CKPT_SUBDIR}"

# Check if container is already running
if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo_info "Container ${CONTAINER_NAME} is already running."
elif docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo_info "Container ${CONTAINER_NAME} exists but is stopped. Starting it..."
    docker start "${CONTAINER_NAME}"
else
    echo_info "Creating and starting new container using start_container.sh..."
    bash "${SCRIPT_DIR}/tools/docker/start_container.sh"
    echo_info "Container started successfully"
fi

# Step 3: Download dataset and create tokenizer metadata
echo_info "Step 3: Download dataset and create tokenizer metadata..."
docker exec "${CONTAINER_NAME}" bash -c "
    cd ${CONTAINER_PRIMUS_ROOT}
    if [ ! -f ${MOUNT_DATA_PATH}/train.npy ]; then
        python download_dataset.py --data_dir ${MOUNT_DATA_PATH}
        python convert_dataset.py --data_dir ${MOUNT_DATA_PATH}
    else
        echo 'Dataset already exists: ${MOUNT_DATA_PATH}/train.npy'
    fi

    if [ ! -f ${MOUNT_DATA_PATH}/packed_metadata.jsonl ]; then
        python3 create_metadata.py ${SEQ_LENGTH} ${MOUNT_DATA_PATH}/packed_metadata.jsonl
    else
        echo 'Metadata already exists: ${MOUNT_DATA_PATH}/packed_metadata.jsonl'
    fi
"

# Patches are files under primus/recipes/patches on the host; git apply runs in Megatron-Bridge submodule (pinned HEAD).
echo_info "Megatron-Bridge (host): ${MEGATRON_BRIDGE_HOST}"
echo_info "Patch bundle (host): ${PATCHES_HOST}"

# Step 4: LoRA / NeMo-style iteration timing (lora.py, train_utils.py).
echo_info "Step 4: Patching Megatron-Bridge (LoRA fused controls + NeMo-style train log timing)..."
docker exec "${CONTAINER_NAME}" bash -c "
    # Volume mounts often make repo dirs owned by host UID; Git 2.35+ refuses that unless marked safe.
    git config --global --add safe.directory ${CONTAINER_PRIMUS_ROOT}
    git config --global --add safe.directory ${MEGATRON_BRIDGE_CONTAINER}
    cd ${MEGATRON_BRIDGE_CONTAINER}
    echo \"[patch] Megatron-Bridge HEAD: \$(git rev-parse --short HEAD)\"
    # Patches are diffs against this commit. A bind-mounted tree often already contains prior applies;
    # reset so the working tree matches HEAD and git apply can find the expected context.
    echo \"[patch] git reset --hard HEAD (drops uncommitted Megatron-Bridge edits under this submodule)\"
    git reset --hard HEAD
    git apply ${PATCH_DIR}/megatron_nemo_lora_only.patch || true"

# Step 5: Validation sample accounting and capped validation dataloader (loaders.py, samplers.py).
echo_info "Step 5: Patching Megatron-Bridge for llama2_70b_lora training (validation / consumed samples)..."
docker exec "${CONTAINER_NAME}" bash -c "
    git config --global --add safe.directory ${CONTAINER_PRIMUS_ROOT}
    git config --global --add safe.directory ${MEGATRON_BRIDGE_CONTAINER}
    cd ${MEGATRON_BRIDGE_CONTAINER}
    git apply ${PATCH_DIR}/megatron_bridge_validation_consumed_samples.patch || true"

# Step 5b: Deterministic evaluation (reset val iterator each eval; loaders.py, eval.py). Apply after Step 5.
echo_info "Step 5b: Patching Megatron-Bridge for deterministic evaluation..."
docker exec "${CONTAINER_NAME}" bash -c "
    git config --global --add safe.directory ${CONTAINER_PRIMUS_ROOT}
    git config --global --add safe.directory ${MEGATRON_BRIDGE_CONTAINER}
    cd ${MEGATRON_BRIDGE_CONTAINER}
    git apply ${PATCH_DIR}/megatron_bridge_deterministic_eval.patch || true"

# Step 6: Start training
echo_info "Step 6: Starting llama2_70b_lora training..."
echo_info "Training configuration: examples/megatron_bridge/configs/MI355X/llama2_70b_lora_posttrain.yaml"
if [ -z "${PRETRAINED_CHECKPOINT}" ]; then
    echo_warn "No Megatron iter_* checkpoint found under ${HOST_MLPERF_DATA}/${MEGATRON_CKPT_SUBDIR}. LoRA PEFT needs pretrained_checkpoint — place weights there or set it in the posttrain yaml."
fi
POSTTRAIN_CLI_EXTRA=()
if [ -n "${PRETRAINED_CHECKPOINT}" ]; then
    POSTTRAIN_CLI_EXTRA+=( "modules.post_trainer.overrides.pretrained_checkpoint=${PRETRAINED_CHECKPOINT}" )
fi
docker exec -it "${CONTAINER_NAME}" bash -c "
    cd ${CONTAINER_PRIMUS_ROOT} && \
    PACKED_TRAIN_DATA_PATH=\"${PACKED_TRAIN_DATA_PATH}\" \
    PACKED_VAL_DATA_PATH=\"${PACKED_VAL_DATA_PATH}\" \
    PACKED_METADATA_PATH=\"${PACKED_METADATA_PATH}\" \
    HF_TOKEN=${HF_TOKEN} \
    WANDB_API_KEY=${WANDB_API_KEY} \
    MEGATRON_BRIDGE_LOGGING_LEVEL=10 \
    NVTE_DEBUG=1 \
    NVTE_DEBUG_LEVEL=2 \
    NVTE_FUSED_ATTN_LOG_CONFIG=1 \
    NVTE_RS_STRIDED_ATOMIC=2 \
    NVTE_FP8_DPA_BWD=1 \
    NVTE_FUSED_ATTN=1 \
    NVTE_FUSED_ATTN_CK=1 \
    NVTE_FUSED_ATTN_AOTRITON=1 \
    NVTE_USE_HIPBLASLT=1 \
    NVTE_USE_CAST_TRANSPOSE_TRITON=1 \
    NVTE_USE_OPTIMIZED_HIPIFIED_CAST_TRANSPOSE=0 \
    NVTE_CK_USES_BWD_V3=1 \
    NVTE_CK_USES_FWD_V3=1 \
    NVTE_CK_IS_V3_ATOMIC_FP32=0 \
    ./runner/primus-cli direct train posttrain --config examples/megatron_bridge/configs/MI355X/llama2_70b_lora_posttrain.yaml $(printf '%q ' "${POSTTRAIN_CLI_EXTRA[@]}")"

    #TODO: TE_SWIGLU
