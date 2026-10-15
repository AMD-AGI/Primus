#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

: "${SLURM_JOB_ID:?Run inside a Slurm/Spur allocation}"
: "${SLURM_NNODES:?Missing SLURM_NNODES}"
: "${DATA_ROOT:?Set DATA_ROOT}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT}"

export FLUX_CONFIG=${FLUX_CONFIG:-config_4n_gbs1024.sh}
[[ -f "$SCRIPT_DIR/$FLUX_CONFIG" ]] || { echo "Unknown FLUX_CONFIG: $FLUX_CONFIG" >&2; exit 2; }
source "$SCRIPT_DIR/$FLUX_CONFIG"
[[ "$NNODES" == "$SLURM_NNODES" ]] || {
    echo "$FLUX_CONFIG requires $NNODES nodes; allocation has $SLURM_NNODES" >&2
    exit 2
}
if [[ -z "${MASTER_ADDR:-}" ]]; then
    if [[ "$NNODES" == "1" ]]; then
        MASTER_ADDR=${SLURMD_NODENAME:-$SLURM_JOB_NODELIST}
    elif command -v scontrol >/dev/null; then
        MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
    else
        MASTER_ADDR=${SPUR_PEER_NODES:?Set MASTER_ADDR}
        MASTER_ADDR=${MASTER_ADDR%%,*}
        MASTER_ADDR=${MASTER_ADDR%%:*}
    fi
fi
export MASTER_ADDR MASTER_PORT=${MASTER_PORT:-29500}

srun_args=(--nodes="$NNODES" --ntasks="$NNODES" --ntasks-per-node=1)
command -v spur >/dev/null && srun_args+=(--jobid="$SLURM_JOB_ID" --overlap)

# Forward the selected recipe's exported surface without duplicating it here.
mapfile -t forward_vars < <(
    env -i PATH="$PATH" bash -c 'source "$1"; compgen -e' bash "$SCRIPT_DIR/$FLUX_CONFIG" |
        grep -Ev '^(PATH|PWD|SHLVL|_)$'
)
# Preserve bounded recipe variables created only by the current overrides.
mapfile -t current_flux_env < <(compgen -e | grep '^FLUX_' || true)
forward_vars+=("${current_flux_env[@]}")
forward_vars+=(
    FLUX_CONFIG LAUNCH_MODE DOCKER_IMAGE CONFIG PRIMUS_WORKSPACE DATASET_PATH
    EVAL_DATASET_PATH EMPTY_ENCODINGS_PATH OUTPUT_DIR MLLOG_OUTPUT_FILE EVAL_BATCH_SIZE
    PROFILE_OUTPUT_DIR TORCHINDUCTOR_CACHE_DIR TORCHINDUCTOR_CACHE_SEED
    TORCHINDUCTOR_CACHE_EXPORT FSDP2_HSDP_FP8_ALL_REDUCE FSDP2_HSDP_FP8_BLOCK_SIZE
)
env_args=(DATA_ROOT="$DATA_ROOT" OUTPUT_ROOT="$OUTPUT_ROOT" NNODES="$NNODES" MASTER_ADDR="$MASTER_ADDR" MASTER_PORT="$MASTER_PORT")
declare -A seen_env=()
for name in "${forward_vars[@]}"; do
    if [[ -v "$name" && -z "${seen_env[$name]:-}" ]]; then
        env_args+=("$name=${!name}")
        seen_env[$name]=1
    fi
done

exec srun "${srun_args[@]}" env "${env_args[@]}" bash "$SCRIPT_DIR/run_with_docker.sh"
