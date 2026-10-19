#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# End-to-end runner for AMDiffusionBenchmark + DLRMBenchmark on MI300 / MI355 nodes.
#
# All settings (GPU, image, models, batch sizes, train args, HF cache, DLRM) come from a YAML
# config (default: config.yaml next to this script). Command-line options override the config
# for a single run; see `./run_benchmarks.sh --help` and README.md.
#
#   export HF_TOKEN=hf_xxx
#   ./run_benchmarks.sh                       # uses ./config.yaml
#   ./run_benchmarks.sh -c my_config.yaml
#   ./run_benchmarks.sh --bs flux=12 --models flux,sdxl

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONTAINER=""
RUN_DIR=""
FAILED_STEPS=0

log() { echo "[diffusion_bench $(date '+%F %T')] $*"; }
die() { log "ERROR: $*"; exit 1; }

# Everything written to the results folder passes through this, so HF_TOKEN
# never lands in a log there.
redact() {
    if [[ -n "${HF_TOKEN:-}" ]]; then
        DIFFBENCH_SECRET="$HF_TOKEN" awk 'BEGIN { s = ENVIRON["DIFFBENCH_SECRET"]; n = length(s) }
            { while ((i = index($0, s)) > 0) $0 = substr($0, 1, i - 1) "<redacted>" substr($0, i + n)
              print; fflush() }'
    else
        cat
    fi
}

for arg in "$@"; do
    if [[ "$arg" == -h || "$arg" == --help ]]; then
        exec python3 "$SCRIPT_DIR/config.py" --help
    fi
done

# ----------------------------------------------------------------------------- helpers

detect_gpu() {
    local v
    v=$(grep -h gfx_target_version /sys/class/kfd/kfd/topology/nodes/*/properties 2>/dev/null \
        | awk '$2 > 0 { print $2; exit }')
    case "$v" in
        9040*) echo mi300 ;;   # gfx940/941/942: MI300X / MI325X
        9050*) echo mi355 ;;   # gfx950: MI350X / MI355X
        *)     echo unknown ;;
    esac
}

gpu_count() {
    grep -h gfx_target_version /sys/class/kfd/kfd/topology/nodes/*/properties 2>/dev/null \
        | awk '$2 > 0' | wc -l
}

gpu_product() {
    cat /sys/class/drm/card*/device/product_name 2>/dev/null | sort -u | paste -sd ';'
}

start_container() {
    local image=$1 role=$2
    CONTAINER="diffusion-bench-${role}-$(date +%Y%m%d%H%M%S)-$$"

    if ! docker image inspect "$image" >/dev/null 2>&1; then
        log "Image $image not found locally, pulling..."
        docker pull "$image" || die "Failed to pull $image (docker login needed?)"
    fi

    log "Starting container $CONTAINER from $image"
    local extra=() label env
    # HF_TOKEN_PATH keeps `hf auth login` from writing the token into the
    # mounted cache, which may be shared (the default is under /tmp).
    [[ -n "${HF_TOKEN:-}" ]] && extra+=(-e HF_TOKEN -e HF_TOKEN_PATH=/root/.hf_token)
    for label in ${LABELS[@]+"${LABELS[@]}"}; do
        extra+=(--label "$label")
    done
    for env in ${ENVS[@]+"${ENVS[@]}"}; do
        extra+=(-e "$env")
    done
    local hf_mount="$HF_CACHE_DIR:$HF_MOUNT_TARGET"
    if (( HF_READONLY )); then
        hf_mount+=":ro"
        extra+=(-e HF_HUB_OFFLINE=1)
    fi
    # --rm: a container stopped from outside (e.g. by the perf batch runner's
    # cleanup) is removed too, rather than left behind stopped.
    docker run -d --rm --name "$CONTAINER" \
        --privileged --network host --ipc host \
        --device /dev/dri --device /dev/kfd \
        --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
        --group-add video --ulimit memlock=-1:-1 \
        ${extra[@]+"${extra[@]}"} \
        -v "$hf_mount" \
        -v "$RUN_DIR":/results \
        -v "$SCRIPT_DIR":/diffusion_bench:ro \
        "$image" sleep infinity >/dev/null \
        || die "Failed to start container"
    docker image inspect "$image" --format '{{.Id}}' > "$RUN_DIR/env/image_id_${role}.txt" 2>/dev/null
}

stop_container() {
    [[ -z "$CONTAINER" ]] && return
    docker exec "$CONTAINER" bash /diffusion_bench/in_container.sh cleanup >/dev/null 2>&1
    docker exec "$CONTAINER" bash /diffusion_bench/in_container.sh finalize "$(id -u)" "$(id -g)" >/dev/null 2>&1
    if (( KEEP_CONTAINER )); then
        log "Keeping container $CONTAINER (docker exec -it $CONTAINER bash; docker rm -f $CONTAINER when done)"
    else
        log "Removing container $CONTAINER"
        docker rm -f "$CONTAINER" >/dev/null 2>&1
    fi
    CONTAINER=""
}

record_status() {
    printf '%s\t%s\t%s\t%s\t%s\n' "$@" >> "$RUN_DIR/status.tsv"
}

# run_step <name> <timeout> <in_container.sh args...>
run_step() {
    local name=$1 tmo=$2
    shift 2
    local logf="$RUN_DIR/logs/${name}.log" limit="no timeout"
    (( tmo > 0 )) && limit="timeout: ${tmo}s"
    log "================ START ${name} (${limit}) ================"
    local t0=$SECONDS
    docker exec "$CONTAINER" timeout -k 60 "$tmo" bash /diffusion_bench/in_container.sh "$@" 2>&1 | redact | tee "$logf"
    local rc=${PIPESTATUS[0]}
    local dur=$((SECONDS - t0)) status
    case $rc in
        0)       status=success ;;
        2)       status=oom ;;
        3)       status=failed ;;
        4)       status=not_run ;;
        124)     status=timeout ;;
        # timeout's SIGKILL after its grace period, unless the step was killed
        # earlier from outside (host OOM killer, docker kill).
        137)     if (( tmo > 0 && dur >= tmo )); then status=timeout; else status=killed; fi ;;
        *)       status=error ;;
    esac
    record_status "$name" "$status" "$rc" "$dur" "logs/${name}.log"
    log "================ END ${name}: ${status} (rc=${rc}, $((dur / 60))m$((dur % 60))s) ================"
    docker exec "$CONTAINER" bash /diffusion_bench/in_container.sh cleanup >/dev/null 2>&1
    [[ $status == success || ( $status == not_run && $DRY_RUN == 1 ) ]] || FAILED_STEPS=$((FAILED_STEPS + 1))
    return "$rc"
}

finish() {
    local rc=$?
    trap - EXIT INT TERM
    stop_container
    if [[ -n "$RUN_DIR" && -d "$RUN_DIR" ]]; then
        python3 "$SCRIPT_DIR/parse_results.py" "$RUN_DIR" || log "WARNING: result parsing failed"
        log "Results directory: $RUN_DIR"
    fi
    (( rc == 0 && FAILED_STEPS > 0 )) && rc=1
    exit "$rc"
}

# ----------------------------------------------------------------------------- config

command -v python3 >/dev/null || die "python3 is required on the host"

DETECTED_GPU=$(detect_gpu)
PLAN=$(python3 "$SCRIPT_DIR/config.py" "$@" --detected-gpu "$DETECTED_GPU" --emit shell) || exit 1
PARSE_ONLY=""
eval "$PLAN"

if [[ -n "$PARSE_ONLY" ]]; then
    exec python3 "$SCRIPT_DIR/parse_results.py" "$PARSE_ONLY"
fi

# ----------------------------------------------------------------------------- preflight

command -v docker >/dev/null || die "docker not found"
docker info >/dev/null 2>&1 || die "Cannot talk to the docker daemon (are you in the docker group?)"

if (( HF_READONLY )); then
    [[ -d "$HF_CACHE_DIR" ]] || die "Read-only HF cache does not exist: $HF_CACHE_DIR"
else
    mkdir -p "$HF_CACHE_DIR" || die "Cannot create HF cache dir $HF_CACHE_DIR"
fi
if (( DOWNLOAD )) && (( ${#DIFF_MODELS[@]} > 0 )) && [[ -z "${HF_TOKEN:-}" ]] \
   && [[ ! -f "$HF_CACHE_DIR/token" && ! -f "$(dirname "$HF_CACHE_DIR")/token" ]]; then
    die "HF_TOKEN is not set and no token found in the HF cache. Run: export HF_TOKEN=hf_xxx (or set download: false)"
fi

# shellcheck disable=SC2153  # GPU is set by eval "$PLAN"
RUN_DIR="$OUTPUT_ROOT/${GPU}_$(hostname -s)_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/env" || die "Cannot create $RUN_DIR"
printf 'step\tstatus\texit_code\tduration_s\tlog\n' > "$RUN_DIR/status.tsv"
cp "$CONFIG_FILE" "$RUN_DIR/config.yaml"
python3 "$SCRIPT_DIR/config.py" "$@" --detected-gpu "$DETECTED_GPU" --emit yaml > "$RUN_DIR/resolved_config.yaml"

# The logger ignores INT/TERM, so cleanup after a signal can still be logged;
# it ends when the script's output closes.
exec > >(trap '' INT TERM; redact | tee -a "$RUN_DIR/run.log") 2>&1
trap finish EXIT
trap 'log "Interrupted, cleaning up..."; exit 130' INT TERM

NGPU=$(gpu_count)
# shellcheck disable=SC2153  # IMAGE is set by eval "$PLAN"
{
    echo "gpu_target: $GPU"
    echo "gpu_product: $(gpu_product)"
    echo "gpu_count: $NGPU"
    echo "hostname: $(hostname)"
    echo "date: $(date -Is)"
    echo "config: $CONFIG_FILE"
    echo "image: $IMAGE"
    echo "dlrm_image: $DLRM_IMAGE"
    echo "hf_cache: $HF_CACHE_DIR -> $HF_MOUNT_TARGET$( (( HF_READONLY )) && echo ' (read-only)')"
    echo "dry_run: $DRY_RUN"
    echo "kernel: $(uname -r)"
    echo "amdgpu_driver: $(cat /sys/module/amdgpu/version 2>/dev/null || echo unknown)"
} > "$RUN_DIR/run_info.txt"

log "Config   : $CONFIG_FILE"
log "Results  : $RUN_DIR"
log "GPU      : $GPU (detected: $DETECTED_GPU; $(gpu_product), ${NGPU} GPUs)"
log "Image    : $IMAGE"
log "HF cache : $HF_CACHE_DIR -> $HF_MOUNT_TARGET$( (( HF_READONLY )) && echo ' (read-only)')"
# Guarded expansions throughout: bash < 4.4 treats an empty array as unset
# under set -u, and the plan arrays are empty when a suite is disabled.
for run in ${DIFF_RUNS[@]+"${DIFF_RUNS[@]}"}; do
    IFS=$'\t' read -ra parts <<< "$run"
    log "Diffusion: ${parts[0]}  batch size ${parts[1]}  ${parts[*]:2}"
done
for p in ${DLRM_PRECS[@]+"${DLRM_PRECS[@]}"}; do log "DLRM     : $p${DLRM_BATCH_SIZE:+  batch size $DLRM_BATCH_SIZE}"; done

[[ "$DETECTED_GPU" != unknown && "$DETECTED_GPU" != "$GPU" ]] \
    && log "WARNING: config targets $GPU but this node has $DETECTED_GPU GPUs"
(( NGPU == 8 )) || log "WARNING: expected 8 GPUs, found ${NGPU}. Both benchmarks assume 8 GPUs per node."

others=$(docker ps --format '{{.Names}}' | grep -v '^diffusion-bench-' | paste -sd ' ')
[[ -n "$others" ]] && log "WARNING: other running containers may be using the GPUs: $others"

if (( DOWNLOAD )) && (( ${#DIFF_MODELS[@]} > 0 )); then
    avail_gb=$(df -BG --output=avail "$HF_CACHE_DIR" | tail -1 | tr -dc '0-9')
    log "Free space in HF cache filesystem: ${avail_gb} GB (a full first-time download is ~370 GB)"
fi

# ----------------------------------------------------------------------------- diffusion

DLRM_IN_SAME_CONTAINER=1
[[ "$DLRM_IMAGE" != "$IMAGE" ]] && DLRM_IN_SAME_CONTAINER=0

if (( ${#DIFF_RUNS[@]} > 0 )) || (( ${#DLRM_PRECS[@]} > 0 && DLRM_IN_SAME_CONTAINER )); then
    start_container "$IMAGE" main
    run_step setup 600 setup main || die "Container setup failed"
fi

if (( ${#DIFF_RUNS[@]} > 0 )); then
    if (( DOWNLOAD )); then
        download_targets=(all)
        (( DOWNLOAD_ALL )) || download_targets=(${DIFF_MODELS[@]+"${DIFF_MODELS[@]}"})
        run_step download "$DOWNLOAD_TIMEOUT" download ${download_targets[@]+"${download_targets[@]}"} \
            || log "WARNING: asset download failed; continuing with whatever is cached"
    fi

    for run in ${DIFF_RUNS[@]+"${DIFF_RUNS[@]}"}; do
        IFS=$'\t' read -ra parts <<< "$run"
        model=${parts[0]} bs=${parts[1]}
        overrides=("${parts[@]:2}")
        (( DRY_RUN )) && overrides+=("launcher.dry_run=true")
        run_step "diffusion_${model}_bs${bs}" "$STEP_TIMEOUT" diffusion "$model" "$bs" ${overrides[@]+"${overrides[@]}"}
    done
fi

# ----------------------------------------------------------------------------- DLRM

if (( ${#DLRM_PRECS[@]} > 0 )); then
    if (( DRY_RUN )); then
        for p in "${DLRM_PRECS[@]}"; do
            log "DRY RUN: would run ./launch_training_single_node.sh -p $p in /workspace/DLRMBenchmark"
            record_status "dlrm_${p}" skipped 0 0 "-"
        done
    else
        if (( ! DLRM_IN_SAME_CONTAINER )); then
            stop_container
            start_container "$DLRM_IMAGE" dlrm
            run_step setup_dlrm 600 setup dlrm || die "DLRM container setup failed"
        fi
        for p in "${DLRM_PRECS[@]}"; do
            run_step "dlrm_${p}" "$STEP_TIMEOUT" dlrm "$p" "$DLRM_BATCH_SIZE"
        done
    fi
fi

log "All steps finished (${FAILED_STEPS} step(s) not successful)."
