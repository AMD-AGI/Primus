#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)

usage() {
    cat <<'EOF'
Build one exact max-autotune cache archive per node before a FLUX run.

Usage:
  ALLOCATION_JOB_ID=<job-id> \
  DATA_ROOT=/path/to/data \
  OUTPUT_ROOT=/shared/path/to/output \
  bash examples/mlperf/flux1/setup_max_autotune_cache.sh

Run this from a login node with an existing idle Slurm allocation.
OUTPUT_ROOT must be shared by all nodes and must not already contain cache archives.
The training launcher reuses the same OUTPUT_ROOT with:
  TORCHINDUCTOR_CACHE_SEED=/output/cache-node%r.tar.zst

Optional environment variables:
  DOCKER_IMAGE       Container image (default: zirui3/primus-v26.3-flux:v0.4)
  CACHE_BUILD_STEPS  Short-run steps used to finalize each cache (default: 20)
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi

: "${DATA_ROOT:?Set DATA_ROOT to the MLPerf dataset root}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to an empty shared directory}"

DOCKER_IMAGE=${DOCKER_IMAGE:-zirui3/primus-v26.3-flux:v0.4}
CACHE_BUILD_STEPS=${CACHE_BUILD_STEPS:-20}

if [[ -n "${ALLOCATION_JOB_ID:-}" ]]; then
    JOB_ID=$ALLOCATION_JOB_ID
    if ! node_expr=$(squeue -j "$JOB_ID" -h -o '%N' 2>/dev/null); then
        echo "Cannot query allocation $JOB_ID" >&2
        exit 1
    fi
else
    JOB_ID=${SLURM_JOB_ID:-}
    : "${JOB_ID:?Set ALLOCATION_JOB_ID, or run inside a Slurm allocation}"
    node_expr=${SLURM_JOB_NODELIST:-}
    if [[ -z "$node_expr" ]] && ! node_expr=$(squeue -j "$JOB_ID" -h -o '%N' 2>/dev/null); then
        echo "Cannot query allocation $JOB_ID" >&2
        exit 1
    fi
fi
[[ -n "$node_expr" ]] || { echo "Allocation $JOB_ID is not running" >&2; exit 1; }
mapfile -t nodes < <(scontrol show hostnames "$node_expr")
[[ ${#nodes[@]} -gt 0 ]] || {
    echo "Allocation $JOB_ID has no nodes" >&2
    exit 1
}
if ! step_ids=$(squeue -s -j "$JOB_ID" -h -o '%i' 2>/dev/null); then
    echo "Cannot inspect steps for allocation $JOB_ID" >&2
    exit 1
fi
active_steps=$(grep -Ev "^${JOB_ID}\\.(batch|extern)$" <<<"$step_ids" || true)
[[ -z "$active_steps" ]] || {
    echo "Allocation $JOB_ID is busy with steps: $active_steps" >&2
    exit 1
}

mkdir -p "$OUTPUT_ROOT"
exec 9>"$OUTPUT_ROOT/.setup.lock"
flock -n 9 || { echo "Another cache setup is using $OUTPUT_ROOT" >&2; exit 1; }
cache_paths=("$OUTPUT_ROOT/cache-seed.tar.zst")
for rank in "${!nodes[@]}"; do
    cache_paths+=("$OUTPUT_ROOT/cache-node${rank}.tar.zst")
done
for path in "${cache_paths[@]}"; do
    [[ ! -e "$path" ]] || {
        echo "Refusing to overwrite existing cache: $path" >&2
        exit 1
    }
done

printf '%s\n' "[flux1-cache] allocation=$JOB_ID nodes=${nodes[*]}"
printf '%s\n' "[flux1-cache] building generic seed on ${nodes[0]}"

srun --jobid="$JOB_ID" --overlap --nodes=1 --ntasks=1 --nodelist="${nodes[0]}" \
    env REPO_ROOT="$REPO_ROOT" OUTPUT_ROOT="$OUTPUT_ROOT" DOCKER_IMAGE="$DOCKER_IMAGE" \
    bash -lc '
        set -euo pipefail
        docker run --rm --init --privileged \
            --device=/dev/kfd --device=/dev/dri --group-add video \
            --ipc=host --network=host --shm-size=20G \
            -v "$REPO_ROOT:/workspace/Primus" -v "$OUTPUT_ROOT:/output" \
            -w /workspace/Primus \
            -e TORCHINDUCTOR_CACHE_DIR=/output/.cache-seed \
            -e EVAL_BATCH_SIZE=32 \
            -e FLUX_FP8_GEMM_BACKEND=selective_flydsl \
            "$DOCKER_IMAGE" bash -lc "
                set -euo pipefail
                rm -rf /output/.cache-seed
                mkdir -p /output/.cache-seed
                python examples/mlperf/flux1/prewarm_inductor_cache.py
                tar --zstd -cf /output/cache-seed.tar.zst -C /output/.cache-seed .
                rm -rf /output/.cache-seed
            "
    '

if ! step_ids=$(squeue -s -j "$JOB_ID" -h -o '%i' 2>/dev/null); then
    echo "Cannot recheck steps for allocation $JOB_ID" >&2
    exit 1
fi
active_steps=$(grep -Ev "^${JOB_ID}\\.(batch|extern)$" <<<"$step_ids" || true)
[[ -z "$active_steps" ]] || {
    echo "Allocation $JOB_ID became busy with steps: $active_steps" >&2
    exit 1
}

pids=()
for rank in "${!nodes[@]}"; do
    node=${nodes[$rank]}
    log="$OUTPUT_ROOT/cache-node${rank}.log"
    printf '%s\n' "[flux1-cache] building cache-node${rank}.tar.zst on $node"
    srun --jobid="$JOB_ID" --overlap --nodes=1 --ntasks=1 --nodelist="$node" \
        --output="$log" --error="$log" \
        env DATA_ROOT="$DATA_ROOT" OUTPUT_ROOT="$OUTPUT_ROOT" \
        FLUX_CONFIG=config_4n_gbs1024.sh NNODES=1 NODE_RANK=0 \
        DP_REPLICATE=1 GLOBAL_BATCH_SIZE=256 MASTER_ADDR=127.0.0.1 MASTER_PORT="$((29510 + rank))" \
        MAX_STEPS="$CACHE_BUILD_STEPS" MLPERF_ENABLE=false MLPERF_CLEAR_CACHES=false \
        FLUX_FP8_ALL_GATHER=1 TORCH_COMPILE_MODE=max-autotune-no-cudagraphs \
        TORCHINDUCTOR_CACHE_SEED=/output/cache-seed.tar.zst \
        TORCHINDUCTOR_CACHE_EXPORT="/output/cache-node${rank}.tar.zst" \
        PRIMUS_WORKSPACE="/output/workspace-node${rank}" \
        OUTPUT_DIR="/output/cache-build-node${rank}" SAVE_STRATEGY=none \
        bash "$SCRIPT_DIR/run_with_docker.sh" &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
[[ $status -eq 0 ]] || { echo "One or more cache builds failed; inspect cache-node*.log" >&2; exit 1; }

manifest="$OUTPUT_ROOT/cache-manifest.txt"
worktree_dirty=false
[[ -z "$(git -C "$REPO_ROOT" status --porcelain)" ]] || worktree_dirty=true
{
    printf 'commit=%s\n' "$(git -C "$REPO_ROOT" rev-parse HEAD)"
    printf 'worktree_dirty=%s\n' "$worktree_dirty"
    printf 'tracked_diff_sha256=%s\n' "$(git -C "$REPO_ROOT" diff --binary HEAD | sha256sum | awk '{print $1}')"
    printf 'image=%s\n' "$DOCKER_IMAGE"
    printf 'generated_at=%s\n' "$(date -u +%FT%TZ)"
    for rank in "${!nodes[@]}"; do
        archive="$OUTPUT_ROOT/cache-node${rank}.tar.zst"
        [[ -s "$archive" ]] || { echo "Missing cache archive: $archive" >&2; exit 1; }
        printf 'rank%s=%s %s\n' "$rank" "${nodes[$rank]}" "$(sha256sum "$archive" | awk '{print $1}')"
    done
} >"$manifest"

printf '%s\n' "[flux1-cache] ready: $OUTPUT_ROOT"
cat "$manifest"
