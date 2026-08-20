#!/bin/bash
# Host-side MLPerf Llama2-70B LoRA launcher (Primus + Docker).
#
# Orchestrates N timed training trials in a long-lived container, matching the
# MLPerf submission flow in training_results_v6.0 (run_with_docker.sh).
#
# Prerequisites on the host:
#   - Docker + ROCm devices (/dev/kfd, /dev/dri)
#   - Primus repo (this script lives under examples/mlperf/llama2_70b/)
#   - Writable DATADIR (dataset + checkpoint; hooks can populate on first run)
#   - Writable LOGDIR (trial logs + /results artifacts inside the container)
#
# Quick start:
#   export DATADIR=/data/mlperf_llama2
#   export LOGDIR=/data/mlperf_llama2/results
#   export CONT=rocm/primus:v26.5
#   export DGXSYSTEM=MI355X_1x8x1
#   bash examples/mlperf/llama2_70b/run_with_docker.sh
#
# Interactive (prompts for HF_TOKEN, image, NEXP, paths):
#   INTERACTIVE=1 bash examples/mlperf/llama2_70b/run_with_docker.sh
#
# Quiet MLPerf submission logging (:::MLLOG + timing only):
#   SUBMISSION_QUIET=1 bash examples/mlperf/llama2_70b/run_with_docker.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRIMUS_HOST="${PRIMUS_HOST:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

# Defaults (override via env or INTERACTIVE prompts)
: "${DGXSYSTEM:=MI355X_1x8x1}"
: "${CONT:=rocm/primus:v26.5}"
: "${DATADIR:=${HOME}/data/mlperf_llama2}"
: "${LOGDIR:=${DATADIR}/results}"
: "${NEXP:=1}"
: "${CONT_NAME:=mlperf_llama2_70b_lora_primus}"
: "${CLEAR_CACHES:=0}"
: "${CHECK_COMPLIANCE:=0}"
: "${MLPERF_RULESET:=6.0.0}"
: "${DATESTAMP:=$(date +'%y%m%d%H%M%S')}"
: "${INTERACTIVE:=0}"
: "${SUBMISSION_QUIET:=0}"
: "${RUN_RUNTIME_TUNABLES:=1}"

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
prompt_if_interactive "LOGDIR" "Log/results directory on host (mounted at /results)" "${LOGDIR}"
prompt_if_interactive "DGXSYSTEM" "DGXSYSTEM (config file suffix)" "${DGXSYSTEM}"

readonly _config_file="${SCRIPT_DIR}/config_${DGXSYSTEM}.sh"

if [[ -z "${HF_TOKEN:-}" ]]; then
    if [[ "${INTERACTIVE}" == "1" ]] || [[ -t 0 ]]; then
        read -rsp "Hugging Face HF_TOKEN (required): " HF_TOKEN
        echo
        export HF_TOKEN
    fi
fi

if [[ ! -f "${_config_file}" ]]; then
    echo "[ERROR] Config not found: ${_config_file} (set DGXSYSTEM correctly)" >&2
    exit 1
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "[ERROR] HF_TOKEN is required (meta-llama/Llama-2-70b-hf + MLPerf dataset)." >&2
    exit 1
fi

if [[ ! -d "${PRIMUS_HOST}" ]]; then
    echo "[ERROR] Primus repo not found: ${PRIMUS_HOST} (set PRIMUS_HOST)" >&2
    exit 1
fi

mkdir -p "${DATADIR}" "${LOGDIR}" "${LOGDIR}/artifacts"

readonly _logfile_base="${LOGDIR}/${DATESTAMP}"
readonly _cont_name="${CONT_NAME}"
readonly _primus_mount="/workspace/Primus"
readonly _code_mount="/workspace/code"

_cont_mounts=(
    "--volume=${PRIMUS_HOST}:${_primus_mount}"
    "--volume=${SCRIPT_DIR}:${_code_mount}"
    "--volume=${DATADIR}:/data"
    "--volume=${LOGDIR}:/results"
)
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
    "--env=HF_TOKEN=${HF_TOKEN}"
    "--env=PRIMUS_PATH=${_primus_mount}"
    "--env=DATADIR=/data"
    "--env=DATA_PATH=/data"
    "--env=PACKED_DATA_DIR=/data"
    "--env=PRETRAINED_CHECKPOINT=/data/megatron_checkpoints/Llama-2-70b-hf"
    "--env=HF_HOME=/data/.cache/huggingface"
    "--env=ENABLE_MLLOG=1"
    "--env=DGXSYSTEM=${DGXSYSTEM}"
    "--env=LOGDIR=/results"
)

if [[ "${SUBMISSION_QUIET}" == "1" ]]; then
    _extra_env+=(
        "--env=PRIMUS_LOG_SUPPRESSION=1"
        "--env=MLPERF_VERBOSE_LOGS=0"
        "--env=PRIMUS_LOG_GPU_MEM=0"
        "--env=VERBOSE_TRAINING_LOG=0"
    )
fi

# Export names from config_${DGXSYSTEM}.sh for docker exec (mirrors MLPerf reference).
mapfile -t _config_env < <(
    env -i HOME="${HOME}" PATH="${PATH}" bash -c "
        set -a
        source '${_config_file}'
        set +a
        compgen -e
    " | grep -E -v '^(PWD|SHLVL|_|OLDPWD)$' || true
)
_config_docker_env=()
for v in "${_config_env[@]}"; do
    # Never override container PATH/PYTHONHOME from a host-side config source.
    case "${v}" in PATH=*|PYTHONHOME=* ) continue ;; esac
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
echo "[INFO] Results mount:     ${LOGDIR} -> /results"
echo "[INFO] Experiments:       ${NEXP}"
echo "[INFO] Container name:    ${_cont_name}"

if [[ "${DGXSYSTEM}" == MI* ]]; then
    docker run --rm --init --detach \
        --net=host --uts=host --ipc=host \
        --device=/dev/dri --device=/dev/kfd \
        --security-opt=seccomp=unconfined \
        --group-add=video \
        --cap-add=SYS_PTRACE \
        --privileged \
        --shm-size=32g \
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
source /opt/venv/bin/activate
pip install -q -e '${_primus_mount}' --no-deps 2>/dev/null || true
echo '[INFO] Container Primus editable install complete.'
"

_run_cmd="bash ${_code_mount}/run_and_time.sh"

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
    if [[ -f "${LOGDIR}/train.mlperfposttrain.exp.log" ]]; then
        cp -f "${LOGDIR}/train.mlperfposttrain.exp.log" \
            "${LOGDIR}/artifacts/train_${DATESTAMP}_${_experiment_index}.log" || true
    fi

    if [[ "${CHECK_COMPLIANCE}" == "1" ]]; then
        docker exec "${_run_env[@]}" "${_cont_name}" bash -lc \
            "python3 -m mlperf_logging.compliance_checker --usage training \
             --ruleset '${MLPERF_RULESET}' \
             --log_output '/results/compliance_${DATESTAMP}_${_experiment_index}.out' \
             '/results/train.mlperfposttrain.exp.log'" \
            || echo "[WARN] Compliance check failed for trial ${_experiment_index} (non-blocking)"
    fi
done

echo "[OK] Completed ${NEXP} trial(s). Logs: ${_logfile_base}_*.log"
echo "[OK] MLLOG (last trial): ${LOGDIR}/mlperf_logging.out"
