#!/bin/bash
# Host-side MLPerf DLRMv4 launcher (Primus + Docker, one node).
#
# Starts one long-lived rocm/primus container, then runs NEXP timed trials in it
# (examples/mlperf/dlrm_v4/run_and_time.sh), each with its own MLPerf log.
#
# Prerequisites on the host:
#   - Docker + ROCm devices (/dev/kfd, /dev/dri)
#   - DATADIR: the yambda-5b dataset (processed_5b/hstu_cache_L4086/ and
#     shared_metadata/; see README.md). Put it on local NVMe: the run reads ~230 GB.
#   - LOGDIR: writable; trial logs and MLPerf logs land here (/results inside).
#
# Quick start:
#   export DATADIR=/mnt/nvme/dlrm_data
#   export LOGDIR=/mnt/nvme/dlrm_results
#   bash examples/mlperf/dlrm_v4/run_with_docker.sh
#
# Options (environment):
#   CONT=rocm/primus:v26.9  DGXSYSTEM=MI355X_1x8x1  NEXP=1
#   SEED=<n>                 the same seed for every trial (default: random per trial)
#   CHECK_COMPLIANCE=1       run the MLPerf compliance checker on each trial's log
#   PRIVILEGED=1             --privileged (also flushes the global page cache;
#                            without it the dataset's own pages are evicted and verified)
#   EXTRA_ENV="K=V ..."      extra environment for the trials (e.g. NUM_TRAIN_TS=31)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRIMUS_HOST="${PRIMUS_HOST:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

: "${DGXSYSTEM:=MI355X_1x8x1}"
: "${CONT:=rocm/primus:v26.9}"
: "${DATADIR:?Set DATADIR to the yambda-5b dataset root}"
: "${LOGDIR:?Set LOGDIR to a writable results directory}"
: "${NEXP:=1}"
: "${CONT_NAME:=mlperf_dlrm_v4_primus}"
: "${CHECK_COMPLIANCE:=0}"
: "${MLPERF_RULESET:=6.1.0}"
: "${PRIVILEGED:=0}"
: "${EXTRA_ENV:=}"
: "${DATESTAMP:=$(date +'%y%m%d%H%M%S')}"

readonly _config_file="${SCRIPT_DIR}/config_${DGXSYSTEM}.sh"
if [[ ! -f "${_config_file}" ]]; then
    echo "[ERROR] Config not found: ${_config_file} (set DGXSYSTEM correctly)" >&2
    exit 1
fi
if [[ ! -d "${DATADIR}/processed_5b" ]]; then
    echo "[ERROR] ${DATADIR} has no processed_5b/ (see README.md for the dataset)" >&2
    exit 1
fi
mkdir -p "${LOGDIR}/artifacts"

readonly _primus_mount="/workspace/Primus"
readonly _example_dir="${_primus_mount}/examples/mlperf/dlrm_v4"

# config_${DGXSYSTEM}.sh, exported into every docker exec.
mapfile -t _config_env < <(
    env -i HOME="${HOME}" PATH="${PATH}" bash -c "
        set -a
        source '${_config_file}'
        set +a
        env
    " | grep -E -v '^(PWD|SHLVL|_|OLDPWD|HOME|PATH)=' || true
)
_exec_env=()
for v in "${_config_env[@]}"; do
    _exec_env+=("--env=${v}")
done
for v in ${EXTRA_ENV}; do
    _exec_env+=("--env=${v}")
done

_run_opts=(
    --rm --init --detach
    --net=host --uts=host --ipc=host
    --device=/dev/dri --device=/dev/kfd
    --group-add=video
    --cap-add=SYS_PTRACE --cap-add=IPC_LOCK
    --security-opt=seccomp=unconfined
    --ulimit=memlock=-1:-1 --ulimit=stack=67108864:67108864 --ulimit=core=0:0
    --ulimit=nofile=1048576:1048576
    "--volume=${PRIMUS_HOST}:${_primus_mount}"
    "--volume=${DATADIR}:/data:ro"
    "--volume=${LOGDIR}:/results"
)
if [[ -e /dev/infiniband ]]; then
    _run_opts+=(--device=/dev/infiniband)
fi
if [[ "${PRIVILEGED}" == "1" ]]; then
    _run_opts+=(--privileged)
fi

cleanup_docker() {
    docker container rm -f "${CONT_NAME}" >/dev/null 2>&1 || true
}
cleanup_docker
trap cleanup_docker EXIT

echo "[INFO] Primus host path: ${PRIMUS_HOST}"
echo "[INFO] Docker image:     ${CONT}"
echo "[INFO] Data:             ${DATADIR} -> /data (read-only)"
echo "[INFO] Results:          ${LOGDIR} -> /results"
echo "[INFO] Experiments:      ${NEXP}"

docker run "${_run_opts[@]}" --name="${CONT_NAME}" "${CONT}" sleep infinity
docker exec "${CONT_NAME}" true

for _i in $(seq 1 "${NEXP}"); do
    echo "============================================"
    echo "Beginning trial ${_i} of ${NEXP}"
    echo "============================================"
    _seed="${SEED:-$(( (RANDOM << 15) | RANDOM ))}"
    _mllog="/results/mlperf_${DATESTAMP}_${_i}.log"
    set +e
    docker exec "${_exec_env[@]}" --env="SEED=${_seed}" --env="MLPERF_LOG_PATH=${_mllog}" \
        --env="TRAIN_LOG_FILE=/results/train_${DATESTAMP}_${_i}.log" \
        "${CONT_NAME}" bash -lc "bash ${_example_dir}/run_and_time.sh" \
        2>&1 | tee "${LOGDIR}/${DATESTAMP}_${_i}.log"
    _rc=${PIPESTATUS[0]}
    set -e
    if [[ "${_rc}" -ne 0 ]]; then
        echo "[ERROR] Trial ${_i} failed with exit code ${_rc}" >&2
        exit "${_rc}"
    fi
    if [[ "${CHECK_COMPLIANCE}" == "1" ]]; then
        docker exec "${CONT_NAME}" bash -lc \
            "python3 -m mlperf_logging.compliance_checker --usage training \
             --ruleset '${MLPERF_RULESET}' \
             --log_output '/results/compliance_${DATESTAMP}_${_i}.out' '${_mllog}'" \
            || echo "[WARN] Compliance check failed for trial ${_i}"
    fi
done

echo "[OK] Completed ${NEXP} trial(s). Logs: ${LOGDIR}/${DATESTAMP}_*.log, MLPerf logs: ${LOGDIR}/mlperf_${DATESTAMP}_*.log"
