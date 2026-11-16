#!/bin/bash
set -euxo pipefail

if docker info >/dev/null 2>&1; then
    DOCKER="docker"
else
    echo "Docker not accessible directly, using sudo docker"
    DOCKER="sudo docker"
fi

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
cd "$SCRIPT_DIR"

: "${DGXSYSTEM:?DGXSYSTEM not set}"
: "${CONT:?CONT not set}"
: "${DATADIR:?DATADIR not set}"
: "${LOGDIR:?LOGDIR not set}"

: "${NEXP:=1}"
: "${DATESTAMP:=$(date +'%y%m%d%H%M%S%N')}"
: "${CLEAR_CACHES:=1}"
: "${CHECK_COMPLIANCE:=0}"
: "${CONT_NAME:=mlperf_flux1_primus}"
: "${LOG_FREQ:=999999}"
: "${ROOT_SEED:=$(od -An -N4 -tu4 /dev/urandom | tr -d ' ')}"
: "${NGPU:=8}"

: "${PROFILER:=""}"
: "${TORCHPROF_OUTPUT_DIR:=""}"

: "${PROF_WARMUP_STEPS:=5}"
: "${PROF_ACTIVE_STEPS:=2}"
: "${ROCPROF_PROFILER:='none'}"
: "${ROCPROF:='/opt/rocm/bin/rocprofv3'}"
: "${HIPBLASLT_LOG:=0}"
: "${GEMM_OFFLINE_TUNING:=0}"
: "${GEMM_USE_TUNING_RESULTS:=0}"
: "${RUN_OPTUNA:=0}"
: "${OPTUNA_CONFIG:=''}"
: "${ENABLE_MEMORY_PROFILING:=0}"
: "${POWER_PROFILING:=0}"

: "${ENABLE_TRAINING_METRICS_COLLECTION:=0}"

echo "ROOT SEED IS: $ROOT_SEED"

readonly _config_file="./config_${DGXSYSTEM}.sh"
readonly _cont_name="${CONT_NAME}"

# srun exports the submitting node's HOSTNAME to every task, so use the real one.
_this_host="$(hostname -s)"
echo "Running on host: ${_this_host}"

_local_data="${FLUX_LOCAL_DATADIR:-/data/mlperf_flux1}"
_staged_from=""
for _src in "${DATADIR}" ${FLUX_LOCAL_DATA_ALT_SOURCES:-}; do
    if grep -qxF "source=${_src}" "${_local_data}/.stage_complete" 2>/dev/null; then
        _staged_from="${_src}"
        break
    fi
done
if [[ -z "${_staged_from}" ]]; then
    echo "[flux1] ${_this_host}: ${_local_data} is not staged from ${DATADIR} or any of FLUX_LOCAL_DATA_ALT_SOURCES=${FLUX_LOCAL_DATA_ALT_SOURCES:-}; run scripts/stage_dataset_local.sh on this node." | tee /dev/stderr
    exit 1
fi
echo "[flux1] ${_this_host}: using node-local dataset ${_local_data} (staged from ${_staged_from})"
export DATADIR="${_local_data}"

mkdir -p "${LOGDIR}" && chmod 777 "${LOGDIR}"
_cont_mounts=(
    "--volume=${DATADIR}:/dataset"
    "--volume=${LOGDIR}:/results"
    "--volume=$(pwd):/workspace/code"
)

if [[ -f "${_config_file}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${_config_file}"
    set +a
fi

if [[ -n "${RCCL_LIB:-}" ]]; then
    [[ -f "${RCCL_LIB}" ]] || { echo "RCCL_LIB=${RCCL_LIB} not found on ${HOSTNAME}" >&2; exit 1; }
    echo "[flux1] ${HOSTNAME}: overriding image librccl with ${RCCL_LIB}"
    _cont_mounts+=("--volume=${RCCL_LIB}:/opt/venv/lib/python3.12/site-packages/_rocm_sdk_libraries/lib/librccl.so.1:ro")
fi
if [[ "${FSDP_ALL_GATHER_BACKEND:-}" == "rccl_sdma" ]]; then
    _amdgpu=$(cat /sys/module/amdgpu/version 2>/dev/null || echo unknown)
    echo "[flux1] ${HOSTNAME}: amdgpu=${_amdgpu}"
    if [[ "$(printf '%s\n' 6.16 "${_amdgpu}" | sort -V | head -1)" != 6.16 ]]; then
        echo "[flux1] ${HOSTNAME}: WARNING rccl_sdma wants amdgpu >= 6.16, have ${_amdgpu}" >&2
    fi
fi
if [[ -n "${ROCR_LIB:-}" ]]; then
    [[ -f "${ROCR_LIB}" ]] || { echo "ROCR_LIB=${ROCR_LIB} not found on ${HOSTNAME}" >&2; exit 1; }
    echo "[flux1] ${HOSTNAME}: overriding image libhsa-runtime64 with ${ROCR_LIB}"
    _cont_mounts+=("--volume=${ROCR_LIB}:/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib/libhsa-runtime64.so.1:ro")
fi
# Optional override of the image A6W4 kernel. Unset uses the pinned Primus-Turbo file.
if [[ -n "${A6W4_KERNEL:-}" ]]; then
    [[ -f "${A6W4_KERNEL}" ]] || { echo "A6W4_KERNEL=${A6W4_KERNEL} not found on ${HOSTNAME}" >&2; exit 1; }
    echo "[flux1] ${HOSTNAME}: overriding image gemm_a6w4_kernel_tuned with ${A6W4_KERNEL}"
    _cont_mounts+=("--volume=${A6W4_KERNEL}:/opt/venv/lib/python3.12/site-packages/primus_turbo/flydsl/gemm/gemm_a6w4_kernel_tuned.py:ro")
fi

mapfile -t _config_env < <(env -i bash -c ". ${_config_file} && compgen -e" | grep -E -v '^(PWD|SHLVL)')
_config_env+=(DATADIR)
_config_env+=(DGXSYSTEM)
_config_env+=(LOGDIR)
_config_env+=(HF_TOKEN)
_config_env+=(SEED)
_config_env+=(MAX_STEPS)
for _extra_env in TORCHINDUCTOR_CACHE_SEED TORCHINDUCTOR_CACHE_DIR TORCHINDUCTOR_CACHE_EXPORT \
                  TORCH_COMPILE_MODE NCCL_NET_PLUGIN NCCL_DMABUF_ENABLE \
                  NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME DP_REPLICATE \
                  FLUX_FP4_FORWARD FLUX_FP4_ALL_GATHER FLUX_MXFP4_PATCH FLUX_FP4_H16_STOCK \
                  FLUX_PRECOMPUTE_GATHER_PACKS FLUX_SDMA_ATTACH FLUX_AG_COLPACK_KMAJOR_CHECK \
                  FLUX_A6W4_QUANT FLUX_A6W4_FP4_ALL_GATHER FLUX_A6W4_FUSE_BIAS_EPILOGUE FLUX_A6W4_ACT_DUAL \
                  FLUX_A6W4_GELU_PACK_DOUBLE FLUX_A6W4_GEMM_V0 \
                  TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK \
                  NCCL_MAX_NCHANNELS NCCL_MIN_NCHANNELS ATTENTION_BACKEND \
                  NCCL_MAX_CTAS NCCL_ALGO NCCL_PROTO \
                  NCCL_MIN_P2P_NCHANNELS NCCL_MIN_CTAS NCCL_NCHANNELS_PER_NET_PEER \
                  GPU_FORCE_BLIT_COPY_SIZE PRIMUS_TURBO_DGATE_ACC_ATOMIC \
                  NCCL_DEBUG NCCL_DEBUG_SUBSYS NCCL_DEBUG_FILE \
                  FSDP_ALL_GATHER_BACKEND NCCL_RMA_DISABLE TORCH_NCCL_ALLOCATOR_HOOK_OVERRIDE \
                  FSDP_SDMA_MODULES NCCL_CROSS_NIC NCCL_CUMEM_ENABLE NCCL_CTA_POLICY NCCL_SYM_GIN_KERNELS_ENABLE \
                  NCCL_NUM_RMA_INT_CTX NCCL_HIER_CE_COLL_NUM_CTX NCCL_RMA_MULTI_CTX_THRESHOLD \
                  HSA_SDMA_LINEAR_B2B HSA_ENABLE_SDMA_GANG HSA_ENABLE_SDMA_RECOMMENDED_ENG HSA_ENABLE_PEER_SDMA \
                  PRIMUS_MXFP4_BIAS_FOLD; do
    if [[ -n "${!_extra_env:-}" ]]; then
        _config_env+=("${_extra_env}")
    fi
done

readonly _sbatch_file="./sbatch.sh"
_sbatch_env=()
if [[ -f "${_sbatch_file}" ]]; then
    mapfile -t _sbatch_env < <(
        grep '^export ' "${_sbatch_file}" > /tmp/sbatch_exports.sh
        env -i bash -c ". /tmp/sbatch_exports.sh && compgen -e" | grep -E -v '^(PWD|SHLVL)'
        rm -f /tmp/sbatch_exports.sh
    )
fi

_all_env=()
for v in "${_config_env[@]}" "${_sbatch_env[@]}"; do
    [[ -n "$v" ]] || continue
    _all_env+=("--env=$v=${!v-}")
done
if [[ -n "${TORCHINDUCTOR_CACHE_DIR:-}" ]]; then
    _all_env+=("--env=TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR}")
fi

cleanup_docker() {
    ${DOCKER} container rm -f "${_cont_name}" || true
}
trap 'set -eux; cleanup_docker' EXIT

start_container() {
cleanup_docker
${DOCKER} run --rm --init --detach \
      --net=host --uts=host \
      --ipc=host \
      --device /dev/dri \
      --device /dev/kfd \
      --device /dev/infiniband \
      --group-add video \
      --cap-add SYS_PTRACE \
      --cap-add IPC_LOCK \
      --ulimit memlock=-1:-1 \
      --ulimit nofile=1048576:1048576 \
      --security-opt seccomp=unconfined \
      --name="${_cont_name}" "${_cont_mounts[@]}" \
      -e IMAGE_NAME="${CONT}" \
      "${CONT}" sleep infinity

sleep 5

if [[ ${RUNTIME_TUNABLES:-0} == 1 && -z "${_node_setup_done:-}" ]]; then
  echo "Setting runtime tunables"
  bash runtime_tunables.sh
fi

${DOCKER} exec "${_cont_name}" true

if [[ "${FLUX_CLEAN_STALE_SHM:-1}" == "1" ]]; then
    _shm_min=${FLUX_CLEAN_STALE_SHM_MINUTES:-}
    if [[ -z "${_shm_min}" ]]; then
        _others=$(squeue -w "$(hostname -s)" -h -o '%i' 2>/dev/null \
                  | grep -cvx "${SLURM_JOB_ID%%.*}" || true)
        if [[ "${_others}" == "0" ]]; then _shm_min=0; else _shm_min=30; fi
    fi
    # ${M} expands inside the container from the M env passed above.
    # shellcheck disable=SC2016
    ${DOCKER} exec -e M="${_shm_min}" "${_cont_name}" bash -c '
        n=$(find /dev/shm -maxdepth 1 -name "nccl-*" -mmin +${M} 2>/dev/null | wc -l)
        find /dev/shm -maxdepth 1 -name "nccl-*" -mmin +${M} -delete 2>/dev/null
        echo "floor=${M}min removed ${n}, $(ls /dev/shm 2>/dev/null | grep -c nccl) left"
    ' | sed "s/^/[flux1] $(hostname -s): \/dev\/shm /" || true
fi

if [[ -n "${PRIMUS_LIBIONIC_SRC_ABI4_SO:-}" ]]; then
    set +e
    ${DOCKER} exec \
        -e PRIMUS_LIBIONIC_SRC_ABI4_SO="${PRIMUS_LIBIONIC_SRC_ABI4_SO}" \
        "${_cont_name}" \
        bash /workspace/utilities/ainic/fix_libionic_abi4.sh
    _ionic_patch_rc=$?
    set -e
    if [[ ${_ionic_patch_rc} -ne 0 && ${_ionic_patch_rc} -ne 2 ]]; then
        echo "[$(hostname)] ERROR: Ionic ABI-4 patch failed with exit ${_ionic_patch_rc}" >&2
        exit "${_ionic_patch_rc}"
    fi
fi

# Without a usable RDMA device RCCL falls back to TCP sockets and the 4-node job runs
# orders of magnitude slower until Slurm cancels it. AINIC_REQUIRE=0 skips this check.
: "${AINIC_REQUIRE:=1}"
if [[ "${SLURM_NNODES:-1}" -gt 1 && "${AINIC_REQUIRE}" == "1" ]]; then
    _ibv_rc=0
    _ibv_out=$(${DOCKER} exec "${_cont_name}" bash -c \
        'command -v ibv_devices >/dev/null || exit 127; ibv_devices 2>&1') || _ibv_rc=$?
    if [[ ${_ibv_rc} -eq 127 ]]; then
        echo "[flux1] $(hostname): WARNING ibv_devices is not in the image; skipping the RDMA device check" >&2
    else
        if grep -q "does not support the kernel ABI" <<<"${_ibv_out}"; then
            echo "[flux1] $(hostname): ERROR the container's RDMA userspace provider does not match" \
                 "the host kernel driver ABI, so no RDMA device can be opened." >&2
            while IFS= read -r _line; do printf '         %s\n' "${_line}"; done <<<"${_ibv_out}" >&2
            exit 42
        fi
        _ibv_devs=$(awk '$2 ~ /^[0-9a-f]+$/ && length($2) == 16 {print $1}' <<<"${_ibv_out}")
        _wanted=()
        if [[ -n "${NCCL_IB_HCA:-}" && "${NCCL_IB_HCA}" != ^* ]]; then
            IFS=',' read -r -a _wanted <<<"${NCCL_IB_HCA#=}"
        fi
        _found=0
        _missing=()
        for _w in "${_wanted[@]}"; do
            _w=${_w%%:*}
            if grep -q "^${_w}" <<<"${_ibv_devs}"; then
                _found=$((_found + 1))
            else
                _missing+=("${_w}")
            fi
        done
        if [[ ${#_wanted[@]} -eq 0 && -n "${_ibv_devs}" ]]; then
            _found=1
        fi
        if [[ ${_found} -eq 0 ]]; then
            echo "[flux1] $(hostname): ERROR none of NCCL_IB_HCA=${NCCL_IB_HCA:-<unset>} is visible" \
                 "inside the container. Set NCCL_IB_HCA to this site's RDMA devices, or" \
                 "AINIC_REQUIRE=0 to run over sockets. ibv_devices output was:" >&2
            while IFS= read -r _line; do printf '         %s\n' "${_line}"; done <<<"${_ibv_out}" >&2
            exit 42
        fi
        if [[ ${#_missing[@]} -gt 0 ]]; then
            echo "[flux1] $(hostname): WARNING NCCL_IB_HCA entries not visible: ${_missing[*]}" >&2
        fi
        echo "[flux1] $(hostname): RDMA devices OK:" "${_ibv_devs}"
    fi
fi

if [[ -z "${_node_setup_done:-}" ]]; then
echo "=== Checking for InfiniBand/RDMA devices in container ==="
${DOCKER} exec "${_cont_name}" bash -c "rdma link show 2>&1 || echo 'rdma link show: No InfiniBand devices found'" || true
${DOCKER} exec "${_cont_name}" bash -c "ibstat 2>&1 || echo 'ibstat: Command not found or no InfiniBand devices'" || true
${DOCKER} exec "${_cont_name}" bash -c "ls -la /dev/infiniband/ 2>&1 || echo 'No /dev/infiniband directory found'" || true
${DOCKER} exec "${_cont_name}" bash -c "ip link show | grep -i ib || echo 'No InfiniBand interfaces found'" || true
_node_setup_done=1
fi
}

_node_rank="${SLURM_NODEID:-0}"
if [[ -z "${NCCL_SOCKET_IFNAME:-}" ]]; then
  NCCL_SOCKET_IFNAME=$(route | grep '^default' | grep -o '[^ ]*$')
  export NCCL_SOCKET_IFNAME
fi
if [[ -z "${GLOO_SOCKET_IFNAME:-}" ]]; then
  export GLOO_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}
fi
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  export MASTER_PORT=$((20000 + (SLURM_JOB_ID % 20000)))
else
  export MASTER_PORT=${MASTER_PORT:-29501}
fi
for _rdma_var in NCCL_IB_HCA NCCL_IB_GID_INDEX NCCL_DMABUF_ENABLE NCCL_NET_PLUGIN \
                 NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME \
                 TORCHINDUCTOR_CACHE_SEED TORCHINDUCTOR_CACHE_DIR TORCH_COMPILE_MODE; do
  if [[ -n "${!_rdma_var:-}" ]]; then
    _all_env+=("--env=${_rdma_var}=${!_rdma_var}")
  fi
done

echo "=== Node $(hostname) starting ==="
echo "SLURM_PROCID: ${SLURM_PROCID:-}"
echo "SLURM_NODEID: ${_node_rank}"
echo "SLURM_NNODES: ${SLURM_NNODES:-1}"
echo "Master: $MASTER_ADDR:$MASTER_PORT"

RANDOM=$(( ROOT_SEED % 2147483648 ))
_seeds=()
for (( _i = 0; _i < NEXP; _i++ )); do
  _seeds+=( $(( (RANDOM << 15) | RANDOM )) )
done
echo "TRIAL SEEDS: ${_seeds[*]}"

_mllog_file="${MLLOG_OUTPUT_FILE:-/results/mlperf_logging.out}"
_mllog_host="${LOGDIR}/${_mllog_file#/results/}"

for _experiment_index in $(seq 0 $((NEXP-1))); do
{
  echo "Beginning trial ${_experiment_index} of ${NEXP}"
  start_container
  touch "${LOGDIR}/.ready_${SLURM_JOB_ID}_${_experiment_index}_${_node_rank}"
  until [[ $(find "${LOGDIR}" -maxdepth 1 -name ".ready_${SLURM_JOB_ID}_${_experiment_index}_*" 2>/dev/null | wc -l) -ge ${SLURM_NNODES:-1} ]]; do
    sleep 5
  done
  _mllog_offset=$(stat -c %s "${_mllog_host}" 2>/dev/null || echo 0)
  _cache_cleared=false
  if [[ $CLEAR_CACHES == 1 ]]; then
    if bash -c "echo -n 'Clearing cache on ' && hostname && sync && sudo /sbin/sysctl vm.drop_caches=3"; then
      _cache_cleared=true
    else
      echo "WARNING: vm.drop_caches failed; MLPerf cache_clear will be false" >&2
    fi
    ${DOCKER} exec "${_cont_name}" python -c "
from mlperf_logging import mllog
mllogger = mllog.get_mllogger()
mllogger.event(key=mllog.constants.CACHE_CLEAR, value=${_cache_cleared^})"
  fi

  export SEED=${_seeds[_experiment_index]}
  _all_env+=("--env=SEED=${SEED}")
  echo "launching experiment using:" "${_all_env[@]}" "${_cont_name}" /workspace/code/run_and_time.sh
  _run_env=(
    -e MASTER_ADDR="${MASTER_ADDR}"
    -e MASTER_PORT="${MASTER_PORT}"
    -e SLURM_NNODES="${SLURM_NNODES:-1}"
    -e SLURM_NODEID="${_node_rank}"
    -e NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}"
    -e GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME}"
    "${_all_env[@]}"
  )
  # After _all_env, so these win over the config's MLPERF_CLEAR_CACHES=true.
  _run_env+=(-e "MLPERF_CACHE_CLEARED=${_cache_cleared}")
  if [[ "${_cache_cleared}" != "true" ]]; then
    _run_env+=(-e MLPERF_CLEAR_CACHES=false)
  fi
  _arm_env=()
  if [[ -n "${FLUX_ARMS:-}" ]]; then
    IFS=';' read -r -a _arms <<< "${FLUX_ARMS}"
    _this_arm="${_arms[${_experiment_index}]:-}"
    if [[ -n "${_this_arm}" ]]; then
      IFS=',' read -r -a _arm_kv <<< "${_this_arm}"
      for _kv in "${_arm_kv[@]}"; do
        [[ -n "${_kv}" ]] && _arm_env+=(-e "${_kv}")
      done
    fi
    echo "[flux1] trial ${_experiment_index} arm: ${_this_arm:-<control, no overrides>}"
  fi

  if ! ${DOCKER} exec \
      "${_run_env[@]}" \
      "${_arm_env[@]}" \
      "${_cont_name}" \
      bash /workspace/code/run_and_time.sh; then
    echo "Trial ${_experiment_index} of ${NEXP} FAILED; continuing with the next trial"
  fi

  if [[ "${_node_rank}" == "0" && -f "${_mllog_host}" ]]; then
    _trial_log="${DATESTAMP}_${_experiment_index}.log"
    tail -c +$((_mllog_offset + 1)) "${_mllog_host}" > "${LOGDIR}/${_trial_log}"
    if [[ "${CHECK_COMPLIANCE}" == "1" ]]; then
      ${DOCKER} exec "${_cont_name}" \
        python3 -m mlperf_logging.compliance_checker --usage training \
        --ruleset "${MLPERF_RULESET:-6.1.0}" \
        --log_output "/results/compliance_${DATESTAMP}.out" \
        "/results/${_trial_log}" \
        || echo "WARNING: Compliance check failed for experiment ${_experiment_index} (non-blocking)"
    fi
  fi
}
done
