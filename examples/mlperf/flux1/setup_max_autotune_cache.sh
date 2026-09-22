#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)

usage() {
    cat <<'EOF'
Build one synthetic-input max-autotune cache archive per node before a FLUX run.

Usage:
  ALLOCATION_JOB_ID=<job-id> \
  OUTPUT_ROOT=/shared/path/to/output \
  bash examples/mlperf/flux1/setup_max_autotune_cache.sh

Run this from a login node with an existing idle Slurm allocation.
OUTPUT_ROOT must be shared by all nodes and must not already contain cache archives.
The training launcher reuses the same OUTPUT_ROOT with:
  TORCHINDUCTOR_CACHE_SEED=/output/cache-node%r.tar.zst

Optional environment variables:
  DOCKER_IMAGE  Container image (default: zirui3/primus-v26.3-flux:v0.4)
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi

: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to an empty shared directory}"

DOCKER_IMAGE=${DOCKER_IMAGE:-zirui3/primus-v26.3-flux:v0.4}

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
cache_paths=()
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

pids=()
for rank in "${!nodes[@]}"; do
    node=${nodes[$rank]}
    log="$OUTPUT_ROOT/cache-node${rank}.log"
    printf '%s\n' "[flux1-cache] building cache-node${rank}.tar.zst on $node with synthetic inputs"
    srun --jobid="$JOB_ID" --overlap --nodes=1 --ntasks=1 --nodelist="$node" \
        --output="$log" --error="$log" \
        env REPO_ROOT="$REPO_ROOT" OUTPUT_ROOT="$OUTPUT_ROOT" DOCKER_IMAGE="$DOCKER_IMAGE" RANK="$rank" \
        bash -lc '
            set -euo pipefail
            cache_dir="/output/.cache-node${RANK}"
            archive="/output/cache-node${RANK}.tar.zst"
            docker run --rm --init --privileged \
                --device=/dev/kfd --device=/dev/dri --group-add video \
                --ipc=host --network=host --shm-size=20G \
                -v "$REPO_ROOT:/workspace/Primus" -v "$OUTPUT_ROOT:/output" \
                -w /workspace/Primus \
                -e TORCHINDUCTOR_CACHE_DIR="$cache_dir" \
                -e EVAL_BATCH_SIZE=32 \
                -e FLUX_FP8_GEMM_BACKEND=selective_flydsl \
                -e FLUX_FP8_ALL_GATHER=1 \
                -e TORCHINDUCTOR_BENCHMARK_FUSION=1 \
                -e PRIMUS_FLUX_AITER_ATOMIC_FP32=0 \
                -e PRIMUS_FLUX_REUSE_FP8_INPUT=1 \
                "$DOCKER_IMAGE" bash -lc "
                    set -euo pipefail
                    rm -rf \"$cache_dir\"
                    mkdir -p \"$cache_dir\"
                    python -m examples.mlperf.flux1.prewarm_inductor_cache
                    tar --zstd -cf \"$archive\" -C \"$cache_dir\" .
                    rm -rf \"$cache_dir\"
                "
        ' &
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
