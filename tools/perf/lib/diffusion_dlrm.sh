#!/bin/bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# The diffusion_dlrm backend of run_batch.sh. It is not in the catalog:
# BACKEND=...,diffusion_dlrm adds one run per repetition, after the catalog
# configs, that calls diffusion_bench/run_benchmarks.sh for every diffusion and
# DLRM model its config enables. diffusion_bench writes its results folder
# under $RESULT_DIR/diffusion_dlrm.
#
# The runner sets no time limit on this run: diffusion_bench stops a step that
# exceeds its own step_timeout and moves on to the next one.
#
# Callers source lib/common.sh, lib/run.sh and lib/control.sh first.
###############################################################################

PERF_DD_DIR="$SCRIPT_DIR/diffusion_bench"
DIFFUSION_DLRM_CONFIG="${DIFFUSION_DLRM_CONFIG:-$PERF_DD_DIR/config.yaml}"
DIFFUSION_DLRM_HF_CACHE="${DIFFUSION_DLRM_HF_CACHE:-}"

# Read by run_batch.sh and lib/control.sh.
# shellcheck disable=SC2034
PERF_DD_RUNS=0        # runs this backend adds to the batch: NUM_REPS, or 0
# shellcheck disable=SC2034
PERF_DD_ACTIVE=0      # 1 while its run is in flight
# shellcheck disable=SC2034
PERF_DD_LOG=""        # that run's log, for the interrupt handler
PERF_DD_OUTPUT_DIR=""
PERF_DD_GPU=""
PERF_DD_PLAN=""       # diffusion_bench's resolved plan (config.py --emit yaml)
PERF_DD_ARGS=()

# diffusion_bench's GPU setting (mi300 | mi355) for the runner's GPU key, or
# else for the GPU architecture as diffusion_bench reads it (gfx942 / gfx950);
# empty when unknown, which leaves it to diffusion_bench's config.
perf_dd_gpu() {
    local gfx
    case "${GPU:-$(perf_detect_gpu)}" in
        MI300X|MI300|MI308X|MI308|MI325X|MI325) echo mi300; return ;;
        MI350X|MI350|MI355X|MI355) echo mi355; return ;;
    esac
    # </dev/null: with nullglob and no GPU nodes, grep would read stdin.
    gfx=$(grep -h gfx_target_version /sys/class/kfd/kfd/topology/nodes/*/properties 2>/dev/null </dev/null |
        awk '$2 > 0 { print $2; exit }')
    case "$gfx" in
        9040*) echo mi300 ;;
        9050*) echo mi355 ;;
    esac
}

# Read a field of the resolved plan.
perf_dd_plan() {
    printf '%s\n' "$PERF_DD_PLAN" | yq -r "$1" - 2>/dev/null
}

# One line naming what the run covers, e.g.
#   "flux-dev bs16, stable-diffusion-xl bs12; DLRM TF32 FP32 bs32768"
perf_dd_summary() {
    local diffusion dlrm bs
    diffusion=$(perf_dd_plan '[.diffusion_runs[] | .model + " bs" + (.batch_size | tostring)] | join(", ")')
    dlrm=$(perf_dd_plan '.dlrm_precisions | join(" ")')
    bs=$(perf_dd_plan '.dlrm_batch_size')
    printf '%s' "${diffusion:-no diffusion models}"
    if [ -z "$dlrm" ]; then
        printf '; no DLRM\n'
    elif [ "$bs" = null ]; then
        printf '; DLRM %s (batch size from DLRMBenchmark)\n' "$dlrm"
    else
        printf '; DLRM %s bs%s\n' "$dlrm" "$bs"
    fi
}

perf_dd_config_name() {
    local name="${DIFFUSION_DLRM_CONFIG#"$PRIMUS_PATH"/}"
    name="${name%.yaml}"
    echo "${name%.yml}"
}

# Checked before anything launches, so a bad diffusion_bench config fails now
# rather than after hours of catalog runs. Sets the PERF_DD_* globals.
perf_dd_prepare() {
    local out errors line kv env_args=()
    [ "$PERF_RUN_DIFFUSION_DLRM" = 1 ] || return 0
    # shellcheck disable=SC2034
    PERF_DD_RUNS=$NUM_REPS
    PERF_DD_OUTPUT_DIR="$RESULT_DIR/diffusion_dlrm"

    if [ ! -f "$PERF_DD_DIR/run_benchmarks.sh" ]; then
        echo "[ERROR] diffusion_dlrm: $PERF_DD_DIR/run_benchmarks.sh not found." >&2
        return 1
    fi
    if ! perf_uses_containers; then
        echo "[ERROR] diffusion_dlrm starts its own docker container, so it needs PRIMUS_MODE=container (got '$PRIMUS_MODE')." >&2
        return 1
    fi
    if ! command -v python3 >/dev/null 2>&1; then
        echo "[ERROR] diffusion_dlrm: python3 is required on the host (diffusion_bench/config.py)." >&2
        return 1
    fi
    # Absolute, because the batch runs from the repo root.
    DIFFUSION_DLRM_CONFIG=$(realpath -m "$DIFFUSION_DLRM_CONFIG")
    if [ ! -f "$DIFFUSION_DLRM_CONFIG" ]; then
        echo "[ERROR] DIFFUSION_DLRM_CONFIG does not exist: $DIFFUSION_DLRM_CONFIG" >&2
        return 1
    fi
    [ -z "$DIFFUSION_DLRM_HF_CACHE" ] || DIFFUSION_DLRM_HF_CACHE=$(realpath -m "$DIFFUSION_DLRM_HF_CACHE")

    # HF_TOKEN reaches diffusion_bench through the environment; as an option
    # it would show in `ps` and in the logged launch command. EXTRA_ENV goes
    # into its container like into primus-cli's; the runner's own Primus
    # defaults (HSA_NO_SCRATCH_RECLAIM, ...) do not. A secret-looking entry
    # goes by name only: perf_parse_extra_env exported it, and docker reads
    # the value from there instead of its command line.
    for kv in ${PERF_EXTRA_ENV_ARGS[@]+"${PERF_EXTRA_ENV_ARGS[@]}"}; do
        [ "$kv" = --env ] && continue
        if [[ "${kv%%=*}" =~ $PERF_SECRET_RE ]]; then
            env_args+=( --env "${kv%%=*}" )
        else
            env_args+=( --env "$kv" )
        fi
    done
    PERF_DD_GPU=$(perf_dd_gpu)
    PERF_DD_ARGS=( -c "$DIFFUSION_DLRM_CONFIG" --output-dir "$PERF_DD_OUTPUT_DIR"
                   --label "${PERF_LABEL_KEY}=${PERF_LABEL_VALUE}"
                   ${env_args[@]+"${env_args[@]}"} )
    [ -n "$PERF_DD_GPU" ] && PERF_DD_ARGS+=( --gpu "$PERF_DD_GPU" )
    [ -n "${DOCKER_IMAGE:-}" ] && PERF_DD_ARGS+=( --image "$DOCKER_IMAGE" )
    [ -n "$DIFFUSION_DLRM_HF_CACHE" ] && PERF_DD_ARGS+=( --hf-cache "$DIFFUSION_DLRM_HF_CACHE" )
    [ -n "$TRAIN_STEPS" ] && PERF_DD_ARGS+=( --num-iterations "$TRAIN_STEPS" )

    # stdout is the plan; a warning on stderr must not end up in it.
    errors=$(mktemp)
    if ! out=$(python3 "$PERF_DD_DIR/config.py" "${PERF_DD_ARGS[@]}" \
            --detected-gpu "${PERF_DD_GPU:-unknown}" --emit yaml 2>"$errors"); then
        echo "[ERROR] diffusion_dlrm: diffusion_bench rejected its settings:" >&2
        while IFS= read -r line; do
            echo "        $line" >&2
        done < "$errors"
        if [ -z "$PERF_DD_GPU" ]; then
            echo "        The runner could not tell which GPU settings to use (GPU='${GPU:-}');" >&2
            echo "        set GPU=MI300X / MI325X / MI355X, or gpu: mi300 | mi355 in $DIFFUSION_DLRM_CONFIG." >&2
        fi
        rm -f "$errors"
        return 1
    fi
    rm -f "$errors"
    PERF_DD_PLAN="$out"
}

perf_dd_launch_argv() {
    LAUNCH_ARGV=( bash "$PERF_DD_DIR/run_benchmarks.sh" "${PERF_DD_ARGS[@]}" )
}

perf_dd_log_name() {
    local name
    name=$(basename "$DIFFUSION_DLRM_CONFIG")
    name="${name%.yaml}"
    name="${name%.yml}"
    echo "${RESULT_DIR}/${PERF_DIFFUSION_DLRM_BACKEND}-$(sanitize "$name")-$(config_fingerprint "$DIFFUSION_DLRM_CONFIG")-rep${1}_${BATCH_TS}.log"
}

perf_dd_limits_text() {
    echo "none from the runner; diffusion_bench stops a step after step_timeout=$(perf_dd_plan '.step_timeout')s (0 = no limit)"
}

# Pre-run check: the image carries the benchmark repos the plan needs, and the
# Hugging Face cache is usable. Returns 1 to refuse the batch.
perf_dd_preflight() {
    local need=() missing rc hf image
    [ "$PERF_RUN_DIFFUSION_DLRM" = 1 ] || return 0
    image=$(perf_dd_plan '.image')
    [ "$(perf_dd_plan '.diffusion_runs | length')" -gt 0 ] && need+=(/workspace/AMDiffusionBenchmark)
    # A separate DLRM image is diffusion_bench's to pull and check.
    if [ "$(perf_dd_plan '.dlrm_precisions | length')" -gt 0 ] && [ "$(perf_dd_plan '.dlrm_image')" = "$image" ]; then
        need+=(/workspace/DLRMBenchmark)
    fi
    if [ ${#need[@]} -gt 0 ]; then
        # shellcheck disable=SC2016
        missing=$(timeout -k 10 120 docker run --rm --entrypoint /bin/bash "$image" \
            -c 'for d; do [ -d "$d" ] || echo "$d"; done' _ "${need[@]}" 2>&1)
        rc=$?
        if [ "$rc" -ne 0 ]; then
            perf_log "[ERROR] diffusion_dlrm: could not look inside $image (exit $rc): ${missing:0:300}"
            return 1
        fi
        if [ -n "$missing" ]; then
            perf_log "[ERROR] diffusion_dlrm: $image has no $(echo "$missing" | paste -sd ' '); it cannot run these benchmarks."
            return 1
        fi
    fi

    hf=$(perf_dd_plan '.hf_cache_dir')
    if [ "$(perf_dd_plan '.hf_cache_readonly')" = true ]; then
        if [ ! -d "$hf" ]; then
            perf_log "[ERROR] diffusion_dlrm: the read-only Hugging Face cache does not exist: $hf"
            return 1
        fi
    elif ! mkdir -p "$hf" 2>/dev/null || [ ! -w "$hf" ]; then
        perf_log "[ERROR] diffusion_dlrm: the Hugging Face cache is not writable: $hf"
        perf_log "        Point DIFFUSION_DLRM_HF_CACHE (or hf_cache_dir in $DIFFUSION_DLRM_CONFIG) at a cache you own."
        return 1
    fi
    perf_log "[INFO] diffusion_dlrm: $image has ${need[*]:-what the plan needs}; HF cache $hf"
}

# The results folder diffusion_bench created for the run logged in <log>.
perf_dd_run_dir() {
    sed -n 's/^\[diffusion_bench [^]]*\] Results  *: //p' "$1" 2>/dev/null | head -1
}

# PERF_REASON_FN for this run: the steps that did not succeed, e.g.
#   steps: 2 of 7 diffusion_bench steps not successful: diffusion_hunyuan-video_bs3 oom, ...
perf_dd_failure_reason() {
    local dir
    dir=$(perf_dd_run_dir "$1")
    [ -n "$dir" ] && [ -f "$dir/status.tsv" ] || return 0
    awk -F'\t' 'NR > 1 { n++; if ($2 != "success") { bad++; list = list (list ? ", " : "") $1 " " $2 } }
        END { if (bad) printf "steps: %d of %d diffusion_bench steps not successful: %s\n", bad, n, list }' \
        "$dir/status.tsv"
}

# One summary line per diffusion_bench step.
perf_dd_report_steps() {
    local dir
    dir=$(perf_dd_run_dir "$1")
    {
        if [ -z "$dir" ] || [ ! -f "$dir/status.tsv" ]; then
            echo "                    diffusion_bench recorded no steps (see the run log)"
        else
            echo "                    diffusion_bench results: $dir"
            awk -F'\t' 'NR > 1 { printf "                      %s %-36s %-9s exit=%s %ss\n",
                ($2 == "success" ? "[ OK ]" : "[FAIL]"), $1, $2, $3, $4 }' "$dir/status.tsv"
        fi
    } | perf_log_stream
}

# diffusion_bench's root-running container writes the results (and new HF
# cache files) as root, and hands them back to the user when it finishes.
# When it was stopped first, run its finalize step here for the run logged in
# <log>, or the user cannot delete those files. Only root-owned files change
# hands, as in diffusion_bench, so a shared directory's other files are kept.
perf_dd_finalize() {
    local dir hf mounts=() root_owned=""
    dir=$(perf_dd_run_dir "${1:-}")
    hf=$(perf_dd_plan '.hf_cache_dir')
    [ "$(perf_dd_plan '.hf_cache_readonly')" = true ] && hf=""
    [ -n "$dir" ] && [ -d "$dir" ] && root_owned=$(find "$dir" -user 0 -print -quit 2>/dev/null)
    [ -z "$root_owned" ] && [ -n "$hf" ] && [ -d "$hf" ] && root_owned=$(find "$hf" -user 0 -print -quit 2>/dev/null)
    [ -n "$root_owned" ] || return 0
    if [ -n "$dir" ] && [ -d "$dir" ]; then
        mounts+=( -v "$dir:/results" )
    fi
    if [ -n "$hf" ] && [ -d "$hf" ]; then
        mounts+=( -v "$hf:$(perf_dd_plan '.hf_mount_target')" )
    fi
    perf_log "[INFO] diffusion_dlrm: handing root-owned files of ${dir:-the run} and the HF cache back to $(id -un)."
    if ! timeout -k 10 120 docker run --rm --label "${PERF_LABEL_KEY}=${PERF_LABEL_VALUE}" \
            --entrypoint /bin/bash "${mounts[@]}" -v "$PERF_DD_DIR:/diffusion_bench:ro" "$(perf_dd_plan '.image')" \
            /diffusion_bench/in_container.sh finalize "$(id -u)" "$(id -g)" >/dev/null 2>&1; then
        perf_log "[WARN] diffusion_dlrm: could not do it; some files stay root-owned."
    fi
}

# The batch configuration lines for this backend.
perf_dd_config_lines() {
    echo "# diffusion_dlrm: $(perf_dd_config_name) [$(config_fingerprint "$DIFFUSION_DLRM_CONFIG")], after the configs above"
    echo "#   Models     : $(perf_dd_summary)"
    echo "#   GPU setting: $(perf_dd_plan '.gpu')"
    echo "#   HF cache   : $(perf_dd_plan '.hf_cache_dir')"
    echo "#   Output     : $PERF_DD_OUTPUT_DIR"
    echo "#   Run limits : $(perf_dd_limits_text)"
}

# DRY_RUN block, printed after the catalog configs.
perf_dd_print_plan() {
    perf_dd_launch_argv
    echo "[diffusion_dlrm] $(perf_dd_config_name) [$(config_fingerprint "$DIFFUSION_DLRM_CONFIG")]  (runs after the configs above)"
    echo "      $(perf_dd_summary)"
    echo "      gpu=$(perf_dd_plan '.gpu') reps=${NUM_REPS}; $(perf_dd_limits_text)"
    echo "      output: $PERF_DD_OUTPUT_DIR/<gpu>_<host>_<time>/"
    echo "      log: $(perf_dd_log_name 1)"
    echo "      cmd: $(perf_display_argv "${LAUNCH_ARGV[@]}")"
}
