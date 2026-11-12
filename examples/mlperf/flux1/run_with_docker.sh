#!/bin/bash
# Host-side MLPerf Flux.1-Schnell launcher (Primus + Docker).
#
# Same trial loop as mlperf-training-dev/flux1/primus/run_with_docker.sh.
# The image already has Primus and Primus-Turbo under /workspace, so this
# script does not bind-mount either tree. The example directory is mounted
# at /workspace/code, which is where run_and_time.sh and config_*.sh live.
#
# Quick start (from the Primus repo root, on the compute node):
#   export CONT=rocm/primus:v26.9
#   export DGXSYSTEM=MI355X_1x8x32
#   export DATADIR=/path/to/flux1/data
#   export LOGDIR=/path/to/results
#   export NEXP=1
#   bash examples/mlperf/flux1/run_with_docker.sh

set -euxo pipefail

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
cd "$SCRIPT_DIR"

: "${DGXSYSTEM:?DGXSYSTEM not set}"
: "${CONT:?CONT not set}"
: "${DATADIR:?DATADIR not set}"
: "${LOGDIR:?LOGDIR not set}"

: "${NEXP:=1}"
: "${DATESTAMP:=$(date +'%y%m%d%H%M%S%N')}"
: "${CLEAR_CACHES:=1}"
: "${CHECK_COMPLIANCE:=1}"
: "${MLPERF_RULESET:=6.1.0}"

: "${ROOT_SEED:=$(od -An -N4 -tu4 /dev/urandom | tr -d ' ')}"
echo "ROOT SEED IS: ${ROOT_SEED}"

: "${CONT_NAME:=mlperf_flux1_primus}"
: "${HF_TOKEN:=""}"

readonly _config_file="./config_${DGXSYSTEM}.sh"
readonly _logfile_base="${LOGDIR}/${DATESTAMP}"
readonly _cont_name="${CONT_NAME}"
_cont_mounts=(
    "--volume=${DATADIR}:/dataset"
    "--volume=${SCRIPT_DIR}:/workspace/code"
    "--volume=${LOGDIR}:/results"
)

mkdir -p "${LOGDIR}"
mkdir -p "${LOGDIR}/artifacts/"

if [[ -f "${_config_file}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${_config_file}"
    set +a
else
    echo "[ERROR] Config not found: ${_config_file}" >&2
    exit 1
fi

# Optional library overrides. Unset uses the copies in the image.
if [[ -n "${RCCL_LIB:-}" ]]; then
    [[ -f "${RCCL_LIB}" ]] || { echo "RCCL_LIB=${RCCL_LIB} not found" >&2; exit 1; }
    echo "[flux1] overriding image librccl with ${RCCL_LIB}"
    _cont_mounts+=("--volume=${RCCL_LIB}:/opt/venv/lib/python3.12/site-packages/_rocm_sdk_libraries/lib/librccl.so.1:ro")
fi
if [[ -n "${ROCR_LIB:-}" ]]; then
    [[ -f "${ROCR_LIB}" ]] || { echo "ROCR_LIB=${ROCR_LIB} not found" >&2; exit 1; }
    echo "[flux1] overriding image libhsa-runtime64 with ${ROCR_LIB}"
    _cont_mounts+=("--volume=${ROCR_LIB}:/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib/libhsa-runtime64.so.1:ro")
fi
if [[ -n "${A6W4_KERNEL:-}" ]]; then
    [[ -f "${A6W4_KERNEL}" ]] || { echo "A6W4_KERNEL=${A6W4_KERNEL} not found" >&2; exit 1; }
    echo "[flux1] overriding image gemm_a6w4_kernel_tuned with ${A6W4_KERNEL}"
    _cont_mounts+=("--volume=${A6W4_KERNEL}:/opt/venv/lib/python3.12/site-packages/primus_turbo/flydsl/gemm/gemm_a6w4_kernel_tuned.py:ro")
fi

mapfile -t _config_env < <(env -i bash -c ". ${_config_file} && compgen -e" | grep -E -v '^(PWD|SHLVL)')
_config_env+=(DATADIR)
_config_env+=(DGXSYSTEM)
_config_env+=(LOGDIR)
_config_env+=(HF_TOKEN)
_config_env+=(MAX_STEPS)

for _extra_env in FLUX_FP4_FORWARD FLUX_FP4_ALL_GATHER FLUX_MXFP4_PATCH FLUX_FP4_H16_STOCK \
                  FLUX_PRECOMPUTE_GATHER_PACKS FLUX_SDMA_ATTACH FLUX_AG_COLPACK_KMAJOR_CHECK \
                  FLUX_A6W4_QUANT FLUX_A6W4_FP4_ALL_GATHER FLUX_A6W4_FUSE_BIAS_EPILOGUE FLUX_A6W4_ACT_DUAL \
                  FLUX_A6W4_GELU_PACK_DOUBLE FLUX_A6W4_GEMM_V0 \
                  PRIMUS_TURBO_DGATE_ACC_ATOMIC \
                  FSDP_ALL_GATHER_BACKEND NCCL_RMA_DISABLE \
                  TORCH_NCCL_ALLOCATOR_HOOK_OVERRIDE NCCL_DEBUG NCCL_DEBUG_SUBSYS \
                  FSDP_SDMA_MODULES NCCL_CROSS_NIC NCCL_CUMEM_ENABLE NCCL_CTA_POLICY NCCL_SYM_GIN_KERNELS_ENABLE \
                  NCCL_NUM_RMA_INT_CTX NCCL_HIER_CE_COLL_NUM_CTX NCCL_RMA_MULTI_CTX_THRESHOLD \
                  HSA_SDMA_LINEAR_B2B HSA_ENABLE_SDMA_GANG HSA_ENABLE_SDMA_RECOMMENDED_ENG HSA_ENABLE_PEER_SDMA; do
    if [[ -n "${!_extra_env:-}" ]]; then
        _config_env+=("${_extra_env}")
    fi
done

_base_config_env=()
for v in "${_config_env[@]}"; do
    [[ -n "$v" && "$v" != SEED ]] || continue
    _base_config_env+=("--env=$v=${!v-}")
done

cleanup_docker() {
    if docker ps -a --format '{{.Names}}' | grep -q "^${_cont_name}$"; then
        docker container rm -f "${_cont_name}" || true
    else
        echo "Container ${_cont_name} does not exist. Skipping removal."
    fi
}
cleanup_docker
trap 'set -eux; cleanup_docker' EXIT

ib_args=()
if [[ -e /dev/infiniband ]]; then
    ib_args=(--device=/dev/infiniband)
fi

# No --privileged and no CAP_SYS_ADMIN: the cluster docker-authz plugin rejects both.
if [[ "${DGXSYSTEM}" == MI* ]]; then
    echo "Using AMD/ROCm container flags"
    docker run --rm --init --detach \
        --net=host --uts=host --ipc=host \
        --device /dev/dri --device /dev/kfd "${ib_args[@]}" \
        --cap-add=SYS_PTRACE \
        --security-opt=seccomp=unconfined \
        --group-add video \
        --ulimit memlock=-1:-1 \
        --ulimit nofile=1048576:1048576 \
        --name="${_cont_name}" "${_cont_mounts[@]}" \
        -e IMAGE_NAME="${CONT}" \
        "${CONT}" sleep infinity
else
    echo "Using NVIDIA container flags"
    docker run --rm --init --detach \
        --net=host --uts=host \
        --ipc=host --gpus all \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        --device=/dev/infiniband \
        --security-opt=seccomp=unconfined \
        --name="${_cont_name}" "${_cont_mounts[@]}" \
        -e IMAGE_NAME="${CONT}" \
        "${CONT}" sleep infinity
fi

sleep 5
if [[ -f "${SCRIPT_DIR}/runtime_tunables.sh" ]]; then
    echo "[flux1] sourcing ${SCRIPT_DIR}/runtime_tunables.sh"
    set +e
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/runtime_tunables.sh"
    set -euxo pipefail
fi
docker exec "${_cont_name}" true

RANDOM=$(( ROOT_SEED % 2147483648 ))
_seeds=()
for (( _i = 0; _i < NEXP; _i++ )); do
  _seeds+=( $(( (RANDOM << 15) | RANDOM )) )
done
echo "TRIAL SEEDS: ${_seeds[*]}"

for _experiment_index in $(seq 0 $((NEXP-1))); do
  SEED=${_seeds[_experiment_index]}
  _run_config_env=("${_base_config_env[@]}" --env=SEED="${SEED}")
  if [[ -n "${FLUX_ARMS:-}" ]]; then
    IFS=';' read -r -a _arms <<< "${FLUX_ARMS}"
    _this_arm="${_arms[${_experiment_index}]:-}"
    if [[ -n "${_this_arm}" ]]; then
      IFS=',' read -r -a _arm_kv <<< "${_this_arm}"
      for _kv in "${_arm_kv[@]}"; do
        [[ -n "${_kv}" ]] && _run_config_env+=(--env="${_kv}")
      done
    fi
    echo "[flux1] trial ${_experiment_index} arm: ${_this_arm:-<control, no overrides>}"
  fi
  _cache_cleared=false
  if [[ $CLEAR_CACHES == 1 ]]; then
    if bash -c "echo -n 'Clearing cache on ' && hostname && sync && sudo /sbin/sysctl vm.drop_caches=3"; then
      _cache_cleared=true
    else
      echo "WARNING: vm.drop_caches failed; MLPerf cache_clear will be false" >&2
    fi
  fi
  _run_config_env+=(--env=MLPERF_CACHE_CLEARED="${_cache_cleared}")
  if [[ "${_cache_cleared}" != "true" ]]; then
    _run_config_env+=(--env=MLPERF_CLEAR_CACHES=false)
  fi
  (
    echo "Beginning trial ${_experiment_index} of ${NEXP}"
    if [[ $CLEAR_CACHES == 1 ]]; then
      docker exec "${_cont_name}" python -c "
from mlperf_logging import mllog
mllogger = mllog.get_mllogger()
mllogger.event(key=mllog.constants.CACHE_CLEAR, value=${_cache_cleared^})"
    fi
    echo "launching experiment using: ${_run_config_env[*]} ${_cont_name} /workspace/code/run_and_time.sh"
    docker exec "${_run_config_env[@]}" "${_cont_name}" bash /workspace/code/run_and_time.sh
  ) | grep --line-buffered -v "connected peer ranks" | tee "${_logfile_base}_${_experiment_index}.log"

  if [ "${CHECK_COMPLIANCE}" -eq 1 ]; then
      docker exec "${_run_config_env[@]}" "${_cont_name}" \
           python3 -m mlperf_logging.compliance_checker --usage training \
           --ruleset "${MLPERF_RULESET}"                                 \
           --log_output "/results/compliance_${DATESTAMP}.out"           \
           "/results/${DATESTAMP}_${_experiment_index}.log" \
      || echo "WARNING: Compliance check failed for experiment ${_experiment_index} (non-blocking)"
  fi

done

echo "Number of experiments $NEXP"
