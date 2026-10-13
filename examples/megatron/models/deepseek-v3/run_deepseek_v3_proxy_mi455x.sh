#!/bin/bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# DeepSeek-V3 proxy pretraining on MI455X (gfx1250) with the Primus `direct`
# launcher. Run it inside a container made by start_container_mi455x.sh and
# prepared once by prepare_container_mi455x.sh (see README.md):
#
#   bash examples/megatron/models/deepseek-v3/run_deepseek_v3_proxy_mi455x.sh
#   PRESET=ep1_1gpu bash examples/megatron/models/deepseek-v3/run_deepseek_v3_proxy_mi455x.sh
#
#   PRESET               GPUs  layers  EP  experts  GBS  recompute
#   ep4_4gpu (default)      4       4   4      128   64  full, layers 0 and 3 (GFX1250 runbook v3 §3)
#   ep1_1gpu                1       1   1       32   16  off
#
# Both use topk 8, MBS 2, seq 4096, bf16, mock data, 10 iterations and TE
# attention. Arguments are appended to the Primus command line and override the
# preset, e.g. `--train_iters 20` or `--use_turbo_attention True`.
#
# Environment:
#   TURBO_DIR          Primus-Turbo checkout built by prepare_container_mi455x.sh,
#                      used instead of the image's primus_turbo
#   PROFILE=1          torch profiler on rank 0 for iterations 6-7
#                      (trace in <output>/tensorboard/)
#   EXTRA_PYTHONPATH   prepended to the PYTHONPATH of the training processes
#   PRIMUS_EXP_NAME    output directory name: output/<team>/<user>/<exp>/train.log
#   DRY_RUN=1          run the checks and print the launch command only
###############################################################################
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${REPO}"

PRESET=${PRESET:-ep4_4gpu}
case "${PRESET}" in
    ep4_4gpu) NUM_GPUS=4 LAYERS=4 EP=4 EXPERTS=128 GBS=64 RECOMPUTE_IDS="0,3" RECOMPUTE=full ;;
    ep1_1gpu) NUM_GPUS=1 LAYERS=1 EP=1 EXPERTS=32 GBS=16 RECOMPUTE_IDS=null RECOMPUTE=null ;;
    *)
        echo "[dsv3] unknown PRESET=${PRESET} (ep4_4gpu or ep1_1gpu)" >&2
        exit 2
        ;;
esac

# ---------- GFX1250 runbook v3 §1.2 environment ------------------------------
# Triton 3.8 compiles gfx1250 kernels with expert scheduling, which makes Triton
# kernels fault or return NaN at random. The env var misses torch inductor
# kernels, so prepare_container_mi455x.sh also patches Triton (checked below).
export PRIMUS_GPU_MODEL=MI455X
export TRITON_HIP_USE_EXPERT_SCHEDULING=0
export PRIMUS_TURBO_ATTN_BACKEND=${PRIMUS_TURBO_ATTN_BACKEND:-triton}
export HSA_COREDUMP_PATTERN=${HSA_COREDUMP_PATTERN:-/dev/null}
export NNODES=1 NODE_RANK=0 GPUS_PER_NODE=${NUM_GPUS}
export MASTER_ADDR=${MASTER_ADDR:-localhost} MASTER_PORT=${MASTER_PORT:-29537}
# `docker exec -u <uid>` leaves HOME=/ for a uid without a passwd entry.
if [[ ! -w "${HOME:-/}" || "${HOME:-/}" == / ]]; then
    HOME="/tmp/primus-home-$(id -u)"
    export HOME
    mkdir -p "${HOME}"
fi
# Keep kernels compiled with expert scheduling off in their own caches.
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-${HOME}/.cache/primus/triton_expert_sched_off}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-${HOME}/.cache/primus/torchinductor_expert_sched_off}

# ---------- GPUs ---------------------------------------------------------------
# Count the GPUs this container can open (sysfs lists every GPU of the host).
# Otherwise base_env.sh sets HIP_VISIBLE_DEVICES to 0..N-1 of whatever is
# visible, which on a shared node can be someone else's GPU.
VISIBLE=0
for n in /sys/class/kfd/kfd/topology/nodes/*; do
    [[ "$(cat "${n}/gpu_id" 2>/dev/null || echo 0)" != 0 ]] || continue
    minor=$(awk '$1 == "drm_render_minor" {print $2}' "${n}/properties")
    if [[ -r "/dev/dri/renderD${minor}" && -w "/dev/dri/renderD${minor}" ]]; then
        VISIBLE=$((VISIBLE + 1))
    fi
done
if [[ -n "${HIP_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a HVD <<< "${HIP_VISIBLE_DEVICES}"
    if [[ ${#HVD[@]} -ne ${NUM_GPUS} ]]; then
        echo "[dsv3] HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES} lists ${#HVD[@]} GPU(s); ${PRESET} needs ${NUM_GPUS}" >&2
        exit 2
    fi
elif [[ ${VISIBLE} -eq ${NUM_GPUS} ]]; then
    HIP_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS - 1)))
    export HIP_VISIBLE_DEVICES
elif [[ "${DRY_RUN:-0}" != 1 ]]; then
    echo "[dsv3] ${PRESET} needs ${NUM_GPUS} GPU(s) but this container can open ${VISIBLE}: use" \
        "start_container_mi455x.sh or set HIP_VISIBLE_DEVICES" >&2
    exit 2
fi

# ---------- PYTHONPATH ------------------------------------------------------------
# runner/helpers/envs/base_env.sh puts <repo>:<site-packages> in front of
# PYTHONPATH, so an exported PYTHONPATH cannot shadow the image's primus_turbo or
# flydsl. `primus-cli direct --env` is applied last, so the full path goes there.
PY_PREPEND=()
if [[ -n "${EXTRA_PYTHONPATH:-}" ]]; then
    PY_PREPEND+=("${EXTRA_PYTHONPATH}")
fi
if [[ -n "${TURBO_DIR:-}" ]]; then
    TURBO_DIR="$(cd "${TURBO_DIR}" && pwd)"
    PY_PREPEND+=("${TURBO_DIR}" "${TURBO_DIR}/build/pydeps")
fi
if [[ ${#PY_PREPEND[@]} -gt 0 ]]; then
    RUN_PYTHONPATH="$(IFS=:; echo "${PY_PREPEND[*]}"):${REPO}:${REPO}/third_party/Megatron-LM${PYTHONPATH:+:${PYTHONPATH}}"
else
    # what base_env.sh will build
    RUN_PYTHONPATH="${REPO}:$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')${PYTHONPATH:+:${PYTHONPATH}}"
fi

# Check with the run's PYTHONPATH that Triton's expert scheduling is off (with
# the env var unset, which the check honours) and which primus_turbo / flydsl load.
env -u TRITON_HIP_USE_EXPERT_SCHEDULING PYTHONPATH="${RUN_PYTHONPATH}" TURBO_DIR="${TURBO_DIR:-}" python3 - <<'PY'
import os
import re
import sys

from triton.backends.amd import compiler

if compiler.is_expert_scheduling_enabled("gfx1250"):
    sys.exit("[dsv3] Triton still enables gfx1250 expert scheduling: run prepare_container_mi455x.sh first")
import primus_turbo

try:
    import flydsl

    fly = getattr(flydsl, "__version__", "?")
except ImportError:
    fly = "not installed"
turbo_file, turbo_dir = os.path.realpath(primus_turbo.__file__), os.environ["TURBO_DIR"]
print(f"[dsv3] primus_turbo {getattr(primus_turbo, '__version__', '?')} from {os.path.dirname(turbo_file)}, flydsl {fly}")
if turbo_dir:
    if not turbo_file.startswith(os.path.realpath(turbo_dir) + os.sep):
        sys.exit(f"[dsv3] primus_turbo resolves to {turbo_file}, not to TURBO_DIR={turbo_dir}")
    pin = re.search(r"flydsl==([0-9][0-9.]*)", open(os.path.join(turbo_dir, "setup.py")).read())
    if pin and fly != pin.group(1):
        sys.exit(f"[dsv3] flydsl {fly} is not the {pin.group(1)} that {turbo_dir} pins: rerun prepare_container_mi455x.sh")
PY

# ---------- Primus command (runbook v3 §3, scaled by the preset) --------------
ARGS=(
    train pretrain --config examples/megatron/configs/MI355X/deepseek_v3-BF16-pretrain.yaml
    --num_layers "${LAYERS}" --train_iters 10 --micro_batch_size 2 --global_batch_size "${GBS}"
    --use_turbo_deepep False --use_turbo_grouped_gemm True --moe_use_legacy_grouped_gemm False
    --pipeline_model_parallel_size 1 --tensor_model_parallel_size 1 --expert_model_parallel_size "${EP}"
    --pipeline_model_parallel_layout null --num_layers_per_virtual_pipeline_stage null
    --num_experts "${EXPERTS}" --moe_router_topk 8 --gradient_accumulation_fusion False
    --cross_entropy_fusion_impl te --cross_entropy_loss_fusion True
    --recompute_layer_ids "${RECOMPUTE_IDS}" --recompute_granularity "${RECOMPUTE}"
    --disable_last_saving True --moe_layer_freq 1 --mock_data True --pp_warmup True
    --mtp_num_layers 0 --disable_wandb True --disable_tensorboard True
)
SUFFIX=""
if [[ "${PROFILE:-0}" == 1 ]]; then
    ARGS+=(--profile True --use_pytorch_profiler True --torch_profiler_with_stack False
        --profile_step_start 6 --profile_step_end 8)
    SUFFIX=_prof
fi
ARGS+=("$@")

# ---------- Output ------------------------------------------------------------
# Primus reads a numeric user name as an int and then fails to build the output
# path; GNU `id -un` prints the numeric uid for a uid without a passwd entry.
USER_NAME=$(id -un 2>/dev/null | head -n1 || true)
PRIMUS_USER=${PRIMUS_USER:-${USER:-${USER_NAME:-$(id -u)}}}
if [[ "${PRIMUS_USER}" =~ ^[0-9]+$ ]]; then
    PRIMUS_USER="uid${PRIMUS_USER}"
fi
export PRIMUS_USER PRIMUS_WORKSPACE=${PRIMUS_WORKSPACE:-${REPO}/output} PRIMUS_TEAM=${PRIMUS_TEAM:-amd}
export PRIMUS_EXP_NAME=${PRIMUS_EXP_NAME:-dsv3_proxy_${PRESET}${SUFFIX}_$(date +%Y%m%d_%H%M%S)}
LOG="${PRIMUS_WORKSPACE}/${PRIMUS_TEAM}/${PRIMUS_USER}/${PRIMUS_EXP_NAME}/train.log"

LAUNCH=(./primus-cli direct --log_file "${LOG}")
if [[ ${#PY_PREPEND[@]} -gt 0 ]]; then
    LAUNCH+=(--env "PYTHONPATH=${RUN_PYTHONPATH}")
fi
echo "[dsv3] preset ${PRESET}, HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-unset}, log ${LOG}"
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf '%q ' "${LAUNCH[@]}" -- "${ARGS[@]}"
    echo
    exit 0
fi

mkdir -p "$(dirname "${LOG}")"
rc=0
"${LAUNCH[@]}" -- "${ARGS[@]}" || rc=$?
# Runbook v3 §1.2: no kernel may have been compiled with expert scheduling on.
if grep -lq WAVE_SCHED_MODE "${TRITON_CACHE_DIR}"/*/*.amdgcn 2>/dev/null; then
    echo "[dsv3] kernels in ${TRITON_CACHE_DIR} were compiled with gfx1250 expert scheduling:" \
        "do not trust this run" >&2
    [[ ${rc} -ne 0 ]] || rc=3
fi
exit "${rc}"
