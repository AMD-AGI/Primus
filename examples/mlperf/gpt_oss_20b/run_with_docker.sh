#!/bin/bash
# Host-side MLPerf GPT-OSS-20B pretraining launcher (Primus + Docker).
#
# Orchestrates N timed training trials in a long-lived container, matching the
# MLPerf submission flow used by examples/mlperf/llama3.1_8b/run_with_docker.sh.
#
# Prerequisites on the host:
#   - Docker + ROCm devices (/dev/kfd, /dev/dri)
#   - Primus repo (this script lives under examples/mlperf/gpt_oss_20b/)
#   - Writable DATADIR (preprocessed C4 dataset; see README.md)
#   - Writable MODELDIR (Llama 3.1 tokenizer; see README.md)
#   - Writable LOGDIR (trial logs + /results artifacts inside the container)
#   - HF_TOKEN (Hugging Face hub access when MODELDIR is empty)
#
# Quick start:
#   export DATADIR=/data/gpt_oss_20b/data
#   export MODELDIR=/data/gpt_oss_20b/model
#   export LOGDIR=/data/gpt_oss_20b/results
#   export CONT=rocm/primus:v26.5
#   export DGXSYSTEM=MI355X_1x8x1
#   export HF_TOKEN=<your_token>
#   bash examples/mlperf/gpt_oss_20b/run_with_docker.sh
#
# MXFP4 recipe:
#   export EXP=/workspace/Primus/examples/mlperf/gpt_oss_20b/configs/MI355/gpt_oss_20B-MXFP4-deosc-mlperf-pretrain.yaml
#   export MLLOG_LOWEST_NUMERICAL_PRECISION_LINEAR=mxfp4
#
# Interactive (prompts for HF_TOKEN, image, NEXP, paths):
#   INTERACTIVE=1 bash examples/mlperf/gpt_oss_20b/run_with_docker.sh
#
# Quiet MLPerf submission logging (:::MLLOG + timing only):
#   SUBMISSION_QUIET=1 bash examples/mlperf/gpt_oss_20b/run_with_docker.sh
#
# Host runtime tunables before each trial (cpupower, THP, drop_caches; see runtime_tunables.sh):
#   RUN_RUNTIME_TUNABLES=1 bash examples/mlperf/gpt_oss_20b/run_with_docker.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRIMUS_HOST="${PRIMUS_HOST:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

# Defaults (override via env or INTERACTIVE prompts)
: "${DGXSYSTEM:=MI355X_1x8x1}"
: "${CONT:=rocm/primus:v26.5}"
: "${DATADIR:=${HOME}/data/gpt_oss_20b/data}"
: "${MODELDIR:=$(dirname "${DATADIR}")/model}"
: "${LOGDIR:=$(dirname "${DATADIR}")/results}"
: "${NEXP:=1}"
: "${CONT_NAME:=mlperf_gpt_oss_20b_pretrain_primus}"
: "${CLEAR_CACHES:=0}"
: "${CHECK_COMPLIANCE:=0}"
: "${MLPERF_RULESET:=6.0.0}"
: "${DATESTAMP:=$(date +'%y%m%d%H%M%S')}"
: "${INTERACTIVE:=0}"
: "${SUBMISSION_QUIET:=0}"
: "${RUN_RUNTIME_TUNABLES:=0}"

# Map DGXSYSTEM (e.g. MI355X_1x8x1) -> PRIMUS_GPU_MODEL for primus-env when rocm-smi is unavailable in Docker.
if [[ -z "${PRIMUS_GPU_MODEL:-}" && "${DGXSYSTEM}" =~ ^(MI[0-9]+[A-Z]*) ]]; then
    export PRIMUS_GPU_MODEL="${BASH_REMATCH[1]}"
fi

readonly _runtime_tunables="${SCRIPT_DIR}/runtime_tunables.sh"

prompt_if_interactive() {
    local var_name="$1"
    local prompt_text="$2"
    local default_val="${3:-}"
    local current_val="${!var_name:-}"

    if [[ "${INTERACTIVE}" != "1" ]]; then
        return 0
    fi
    if [[ -n "${current_val}" ]]; then
        read -r -p "${prompt_text} [${current_val}]: " _input || true
        if [[ -n "${_input}" ]]; then
            printf -v "${var_name}" '%s' "${_input}"
        fi
    else
        read -r -p "${prompt_text}${default_val:+ [${default_val}]}: " _input || true
        if [[ -n "${_input}" ]]; then
            printf -v "${var_name}" '%s' "${_input}"
        elif [[ -n "${default_val}" ]]; then
            printf -v "${var_name}" '%s' "${default_val}"
        fi
    fi
}

prompt_if_interactive "CONT" "Docker image (CONT)"
prompt_if_interactive "NEXP" "Number of experiments (NEXP)" "1"
prompt_if_interactive "DATADIR" "Data directory on host (mounted at /data)" "${DATADIR}"
prompt_if_interactive "MODELDIR" "Tokenizer directory on host (mounted at /model)" "${MODELDIR}"
prompt_if_interactive "LOGDIR" "Log/results directory on host (mounted at /results)" "${LOGDIR}"
prompt_if_interactive "DGXSYSTEM" "DGXSYSTEM (config file suffix)" "${DGXSYSTEM}"

readonly _config_file="${SCRIPT_DIR}/config_${DGXSYSTEM}.sh"

if [[ ! -f "${_config_file}" ]]; then
    echo "[ERROR] Config not found: ${_config_file} (set DGXSYSTEM correctly)" >&2
    exit 1
fi

if [[ ! -d "${PRIMUS_HOST}" ]]; then
    echo "[ERROR] Primus repo not found: ${PRIMUS_HOST} (set PRIMUS_HOST)" >&2
    exit 1
fi

mkdir -p "${DATADIR}" "${LOGDIR}" "${LOGDIR}/artifacts"

_use_model_mount=0
if [[ -d "${MODELDIR}" && -n "$(ls -A "${MODELDIR}" 2>/dev/null)" ]]; then
    _use_model_mount=1
else
    echo "[INFO] MODELDIR not found or empty (${MODELDIR}); skipping /model mount."
    echo "[INFO] Training will use the default HuggingFace tokenizer (meta-llama/Llama-3.1-8B)."
    if [[ -z "${HF_TOKEN:-}" ]]; then
        if [[ "${INTERACTIVE}" == "1" ]] || [[ -t 0 ]]; then
            read -rsp "Hugging Face HF_TOKEN (required without MODELDIR): " HF_TOKEN
            echo
            export HF_TOKEN
        fi
    fi
    if [[ -z "${HF_TOKEN:-}" ]]; then
        echo "[ERROR] HF_TOKEN is required when MODELDIR is missing or empty." >&2
        exit 1
    fi
fi

readonly _logfile_base="${LOGDIR}/${DATESTAMP}"
readonly _cont_name="${CONT_NAME}"
readonly _primus_mount="/workspace/Primus"
readonly _example_dir="${_primus_mount}/examples/mlperf/gpt_oss_20b"

_cont_mounts=(
    "--volume=${PRIMUS_HOST}:${_primus_mount}"
    "--volume=${DATADIR}:/data"
    "--volume=${LOGDIR}:/results"
)
if [[ "${_use_model_mount}" == "1" ]]; then
    _cont_mounts+=("--volume=${MODELDIR}:/model")
fi
if [[ -d /opt/rocm ]]; then
    _cont_mounts+=("--volume=/opt/rocm:/opt/rocm:ro")
fi

# Container Python/torchrun live in /opt/venv (rocm/primus image). Do not inject the host PATH.
_container_path="/opt/venv/bin:/opt/rocm/bin:/usr/local/bin:/usr/bin:/bin"

_extra_env=(
    "--env=PATH=${_container_path}"
    "--env=VENV_ACTIVATE=/opt/venv/bin/activate"
    "--env=PRIMUS_AUTO_INSTALL=0"
    "--env=PRIMUS_GPU_MODEL=${PRIMUS_GPU_MODEL:-MI355X}"
    "--env=PRIMUS_PATH=${_primus_mount}"
    "--env=DATADIR=/data"
    "--env=DATA_PATH=/data"
    "--env=HF_HOME=/data/.cache/huggingface"
    "--env=ENABLE_MLLOG=1"
    "--env=DGXSYSTEM=${DGXSYSTEM}"
    "--env=LOGDIR=/results"
)
if [[ -n "${HF_TOKEN:-}" ]]; then
    _extra_env+=("--env=HF_TOKEN=${HF_TOKEN}")
fi
# Host overrides win over the sourced config_${DGXSYSTEM}.sh defaults.
if [[ -n "${EXP:-}" ]]; then
    _extra_env+=("--env=EXP=${EXP}")
fi
if [[ -n "${MLLOG_LOWEST_NUMERICAL_PRECISION_LINEAR:-}" ]]; then
    _extra_env+=("--env=MLLOG_LOWEST_NUMERICAL_PRECISION_LINEAR=${MLLOG_LOWEST_NUMERICAL_PRECISION_LINEAR}")
fi
if [[ -n "${PRIMUS_TURBO_MXFP4_SCALE_ROUNDING:-}" ]]; then
    _extra_env+=("--env=PRIMUS_TURBO_MXFP4_SCALE_ROUNDING=${PRIMUS_TURBO_MXFP4_SCALE_ROUNDING}")
fi

if [[ "${SUBMISSION_QUIET}" == "1" ]]; then
    _extra_env+=(
        "--env=PRIMUS_LOG_SUPPRESSION=1"
        "--env=MLPERF_VERBOSE_LOGS=0"
        "--env=PRIMUS_LOG_GPU_MEM=0"
        "--env=VERBOSE_TRAINING_LOG=0"
    )
fi

# Export values from config_${DGXSYSTEM}.sh for docker exec.
mapfile -t _config_env < <(
    env -i HOME="${HOME}" PATH="${PATH}" bash -c "
        set -a
        source '${_config_file}'
        set +a
        env
    " | grep -E -v '^(PWD|SHLVL|_|OLDPWD)=' || true
)
_config_docker_env=()
for v in "${_config_env[@]}"; do
    # Never override container PATH/PYTHONHOME from a host-side config source.
    case "${v}" in PATH=*|PYTHONHOME=*|HOME=* ) continue ;; esac
    _config_docker_env+=("--env=${v}")
done

_base_exec_env=("${_config_docker_env[@]}" "${_extra_env[@]}")

cleanup_docker() {
    if docker ps -a --format '{{.Names}}' | grep -qx "${_cont_name}"; then
        docker container rm -f "${_cont_name}" >/dev/null 2>&1 || true
    fi
}
cleanup_docker
trap 'cleanup_docker' EXIT

echo "[INFO] Primus host path:  ${PRIMUS_HOST}"
echo "[INFO] Docker image:      ${CONT}"
echo "[INFO] Data mount:        ${DATADIR} -> /data"
if [[ "${_use_model_mount}" == "1" ]]; then
    echo "[INFO] Model mount:       ${MODELDIR} -> /model"
else
    echo "[INFO] Model mount:       (skipped; no local tokenizer)"
fi
echo "[INFO] Results mount:     ${LOGDIR} -> /results"
echo "[INFO] Experiments:       ${NEXP}"
echo "[INFO] Container name:    ${_cont_name}"

if [[ "${DGXSYSTEM}" == MI* ]]; then
    docker run --rm --init --detach \
        --net=host --uts=host --ipc=host \
        --device=/dev/dri --device=/dev/kfd --device=/dev/infiniband \
        --security-opt=seccomp=unconfined \
        --group-add=video \
        --cap-add=SYS_PTRACE \
        --cap-add=CAP_SYS_ADMIN \
        --privileged \
        --shm-size=128g \
        --ulimit=memlock=-1 \
        --ulimit=stack=67108864 \
        --name="${_cont_name}" "${_cont_mounts[@]}" \
        -e IMAGE_NAME="${CONT}" \
        "${CONT}" sleep infinity
else
    docker run --rm --init --detach \
        --net=host --uts=host --ipc=host \
        --gpus=all \
        --ulimit=memlock=-1 \
        --ulimit=stack=67108864 \
        --security-opt=seccomp=unconfined \
        --name="${_cont_name}" "${_cont_mounts[@]}" \
        -e IMAGE_NAME="${CONT}" \
        "${CONT}" sleep infinity
fi

sleep 3
docker exec "${_cont_name}" true

# Editable Primus from the host mount (ML deps including mlperf-logging come from the image venv).
_setup_env=("${_base_exec_env[@]}")
docker exec "${_setup_env[@]}" "${_cont_name}" bash -lc "
set -e
pip install -q -e '${_primus_mount}' --no-deps 2>/dev/null || true
if ! python -c 'import primus_mllog' 2>/dev/null; then
    echo '[INFO] primus_mllog not found; installing pinned mlperf-common package.'
    pip install -q \
        'git+https://github.com/AMD-AGI/mlperf-common.git@56d8ca70f60f866312add84adfea66f5447a4c86'
else
    echo '[INFO] primus_mllog is already installed; skipping installation.'
fi
python -c 'import primus_mllog'
echo '[INFO] Container Primus editable install complete.'
"

_run_cmd="bash ${_example_dir}/run_and_time.sh"

for _experiment_index in $(seq 1 "${NEXP}"); do
    echo "============================================"
    echo "Beginning trial ${_experiment_index} of ${NEXP}"
    echo "============================================"

    if [[ "${RUN_RUNTIME_TUNABLES}" == "1" && -x "${_runtime_tunables}" ]]; then
        bash "${_runtime_tunables}"
    fi

    if [[ "${CLEAR_CACHES}" == "1" ]]; then
        echo "[INFO] Clearing host page cache before trial ${_experiment_index}..."
        sync
        if [[ -w /proc/sys/vm/drop_caches ]]; then
            echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null || true
        else
            sudo sysctl vm.drop_caches=3 >/dev/null 2>&1 || true
        fi
    fi

    _run_env=("${_base_exec_env[@]}" --env="SEED=${RANDOM}")

    set +e
    docker exec "${_run_env[@]}" "${_cont_name}" bash -lc "${_run_cmd}" \
        2>&1 | grep --line-buffered -v "connected peer ranks" \
        | tee "${_logfile_base}_${_experiment_index}.log"
    _trial_exit=${PIPESTATUS[0]}
    set -e

    if [[ "${_trial_exit}" -ne 0 ]]; then
        echo "[ERROR] Trial ${_experiment_index} failed with exit code ${_trial_exit}" >&2
        exit "${_trial_exit}"
    fi

    if [[ -f "${LOGDIR}/mlperf_logging.out" ]]; then
        cp -f "${LOGDIR}/mlperf_logging.out" "${LOGDIR}/artifacts/mlperf_logging_${DATESTAMP}_${_experiment_index}.out" || true
    fi
    _train_log="${PRIMUS_HOST}/examples/mlperf/gpt_oss_20b/train.mlperfpretrain.exp.log"
    if [[ -f "${_train_log}" ]]; then
        cp -f "${_train_log}" "${LOGDIR}/artifacts/train_${DATESTAMP}_${_experiment_index}.log" || true
    fi

    if [[ "${CHECK_COMPLIANCE}" == "1" ]]; then
        docker exec "${_run_env[@]}" "${_cont_name}" bash -lc \
            "python3 -m mlperf_logging.compliance_checker --usage training \
             --ruleset '${MLPERF_RULESET}' \
             --log_output '/results/compliance_${DATESTAMP}_${_experiment_index}.out' \
             '/results/${DATESTAMP}_${_experiment_index}.log'" \
            || echo "[WARN] Compliance check failed for trial ${_experiment_index} (non-blocking)"
    fi
done

echo "[OK] Completed ${NEXP} trial(s). Logs: ${_logfile_base}_*.log"
echo "[OK] MLLOG (last trial): ${LOGDIR}/mlperf_logging.out"
