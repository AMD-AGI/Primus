#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)

: "${DATA_ROOT:?Set DATA_ROOT to the directory containing the MLPerf datasets}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to the output directory}"

DOCKER_IMAGE=${DOCKER_IMAGE:-zirui3/primus-v26.3-flux:v0.1}
CONFIG=${CONFIG:-examples/mlperf/flux1/flux.1_schnell_t2i-pretrain.yaml}
CONTAINER_NAME=${CONTAINER_NAME:-primus-mlperf-flux1-${SLURM_JOB_ID:-local}-${SLURM_PROCID:-0}}
NNODES=${NNODES:-${SLURM_NNODES:-1}}
NODE_RANK=${NODE_RANK:-${SLURM_NODEID:-0}}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}

mkdir -p "$OUTPUT_ROOT"

rdma_args=()
if [[ -e /dev/infiniband ]]; then
  rdma_args=(--device=/dev/infiniband)
fi

PRIMUS_SRC=${PRIMUS_SRC:-$REPO_ROOT}
if [[ ! -x "${PRIMUS_SRC}/primus-cli" ]]; then
  echo "PRIMUS_SRC=${PRIMUS_SRC} has no primus-cli" >&2
  exit 1
fi

turbo_mounts=()
if [[ -n "${PRIMUS_TURBO_SRC:-}" ]]; then
  if [[ ! -d "${PRIMUS_TURBO_SRC}/primus_turbo" ]]; then
    echo "PRIMUS_TURBO_SRC=${PRIMUS_TURBO_SRC} has no primus_turbo package" >&2
    exit 1
  fi
  turbo_mounts=(-v "${PRIMUS_TURBO_SRC}:/workspace/Primus-Turbo:ro")
fi
# Host tunables from the MLPerf Flux launcher. sudo -n so a missing password
# cannot block the run; each line is independent.
apply_tunable() {
  local label="$1"
  shift
  if "$@"; then
    echo "runtime tunable ${label}: applied"
  else
    echo "runtime tunable ${label}: skipped"
  fi
}
echo "Applying runtime tunables"
apply_tunable modprobe sudo -n modprobe amdgpu
if command -v rocm_agent_enumerator >/dev/null 2>&1; then
  apply_tunable rocm_agent_enumerator rocm_agent_enumerator
else
  echo "runtime tunable rocm_agent_enumerator: skipped"
fi
apply_tunable cpupower_idle sudo -n cpupower idle-set -d 2
apply_tunable cpupower_performance sudo -n cpupower frequency-set -g performance
apply_tunable drop_caches bash -c 'echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null'
apply_tunable nmi_watchdog bash -c 'echo 0 | sudo -n tee /proc/sys/kernel/nmi_watchdog >/dev/null'
apply_tunable numa_balancing bash -c 'echo 0 | sudo -n tee /proc/sys/kernel/numa_balancing >/dev/null'
apply_tunable randomize_va_space bash -c 'echo 0 | sudo -n tee /proc/sys/kernel/randomize_va_space >/dev/null'
apply_tunable thp_enabled bash -c 'echo always | sudo -n tee /sys/kernel/mm/transparent_hugepage/enabled >/dev/null'
apply_tunable thp_defrag bash -c 'echo always | sudo -n tee /sys/kernel/mm/transparent_hugepage/defrag >/dev/null'
echo "Primus mount: ${PRIMUS_SRC} -> /workspace/Primus"
if [[ -n "${PRIMUS_TURBO_SRC:-}" ]]; then
  echo "Primus-Turbo mount: ${PRIMUS_TURBO_SRC} -> /workspace/Primus-Turbo"
fi

# Forward caller overrides of config_common.sh. Unset names keep that file's defaults,
# which the container sources before primus-cli.
override_args=()
while IFS= read -r name; do
  if [[ -v $name ]]; then
    override_args+=(-e "$name=${!name}")
  fi
done < <(sed -n 's/^[[:space:]]*export \([A-Za-z_][A-Za-z0-9_]*\)=.*/\1/p' "$SCRIPT_DIR/config_common.sh" | sort -u)

docker run --rm --init \
  --name "$CONTAINER_NAME" \
  --device=/dev/kfd --device=/dev/dri "${rdma_args[@]}" --group-add video \
  --cap-add=SYS_PTRACE --security-opt=seccomp=unconfined \
  --ipc=host --network=host --shm-size=20G --ulimit memlock=-1:-1 \
  -v "$PRIMUS_SRC:/workspace/Primus" \
  "${turbo_mounts[@]}" \
  -v "$DATA_ROOT:/data" \
  -v "$OUTPUT_ROOT:/output" \
  -w /workspace/Primus \
  -e NNODES="$NNODES" -e NODE_RANK="$NODE_RANK" \
  -e MASTER_ADDR="$MASTER_ADDR" -e MASTER_PORT="$MASTER_PORT" \
  -e GPUS_PER_NODE="$GPUS_PER_NODE" \
  -e CONFIG="$CONFIG" \
  -e PRIMUS_WORKSPACE="${PRIMUS_WORKSPACE:-/output/primus_workspace}" \
  -e DATASET_PATH="${DATASET_PATH:-/data/cc12m_preprocessed}" \
  -e EVAL_DATASET_PATH="${EVAL_DATASET_PATH:-/data/coco_preprocessed}" \
  -e EMPTY_ENCODINGS_PATH="${EMPTY_ENCODINGS_PATH:-/data/empty_encodings}" \
  -e OUTPUT_DIR="${OUTPUT_DIR:-/output/flux_mlperf}" \
  -e LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-64}" \
  -e GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}" \
  -e LR="${LR:-0.0002}" \
  -e WARMUP_STEPS="${WARMUP_STEPS:-1600}" \
  -e LOG_FREQ="${LOG_FREQ:-999999}" \
  -e MAX_STEPS="${MAX_STEPS:-30000}" \
  -e SAVE_STEPS="${SAVE_STEPS:-100}" \
  -e SAVE_STRATEGY="${SAVE_STRATEGY:-dtcp_full}" \
  -e CHECKPOINT_KEEP_LATEST="${CHECKPOINT_KEEP_LATEST:-3}" \
  -e RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-latest}" \
  -e MLPERF_CLEAR_CACHES="${MLPERF_CLEAR_CACHES:-true}" \
  -e MLLOG_OUTPUT_FILE="${MLLOG_OUTPUT_FILE:-/output/flux_mlperf/mlperf_compliance.log}" \
  -e TARGET_ACCURACY="${TARGET_ACCURACY:-0.586}" \
  -e VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-262144}" \
  -e SEED="${SEED:-10007}" \
  -e PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}" \
  "${override_args[@]}" \
  "$DOCKER_IMAGE" bash -c '
    set -euo pipefail
    # Outputs are written as root; make them readable to the host user afterwards.
    trap "chmod -R a+rX /output 2>/dev/null || true" EXIT
    set -a
    # shellcheck disable=SC1091
    source examples/mlperf/flux1/config_common.sh
    set +a
    if [[ -d /workspace/Primus-Turbo/primus_turbo ]]; then
      python3 - <<'"'"'PY'"'"'
import os
import pathlib
import primus_turbo

image = pathlib.Path(primus_turbo.__file__).resolve().parent
local = pathlib.Path("/workspace/Primus-Turbo/primus_turbo")
merged = pathlib.Path("/opt/mounted-turbo/primus_turbo")

def link_tree(src, dest, overwrite):
    for path in src.rglob("*"):
        rel = path.relative_to(src)
        target = dest / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        if target.exists() or target.is_symlink():
            if not overwrite:
                continue
            target.unlink()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(path)

merged.mkdir(parents=True, exist_ok=True)
link_tree(image, merged, overwrite=False)
link_tree(local, merged, overwrite=True)
print(f"using mounted Primus-Turbo via {merged}", flush=True)
PY
      # base_env.sh puts site-packages ahead of PYTHONPATH; PRIMUS_TURBO_PATH is the hook it
      # provides for a Primus-Turbo tree that must shadow the image install.
      export PRIMUS_TURBO_PATH=/opt/mounted-turbo
      export PYTHONPATH="/opt/mounted-turbo${PYTHONPATH:+:$PYTHONPATH}"
      python3 -c "import primus_turbo, primus_turbo.pytorch._C; print(\"using\", primus_turbo.__file__)"
    fi
    export MLLOG_OUTPUT_FILE=${MLLOG_OUTPUT_FILE:-$OUTPUT_DIR/mlperf_compliance.log}
    mkdir -p "$OUTPUT_DIR"
    if [[ "${MLPERF_CLEAR_CACHES:-true}" == "true" ]]; then
      sync
      echo 3 > /proc/sys/vm/drop_caches || true
    fi
    ./primus-cli direct -- train pretrain \
      --config "$CONFIG"
  '
