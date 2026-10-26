#!/bin/bash
# shellcheck disable=SC2034,SC2153,SC2154  # variables are read by the sourced lib/*.sh
###############################################################################
# Primus Perf Batch Runner (multi-node, with retry on failure)
#
# Purpose:
#   Drive a batch of Primus *multi-node* benchmarks via the
#   `primus-cli slurm srun` launcher, collect the results with enough
#   metadata that a reviewer can reproduce/audit each run from the logs
#   alone, and retry training that crashes mid-step.
#
# This is the multi-node sibling of run_batch.sh: configs, banners, logs and
# the summary are shared (lib/), and on top it retries failed attempts and
# watches the output for failure text.
#
# Usage:
#   1. Allocate the nodes with SLURM first, e.g.:
#          salloc -N 8 -p mi325x -t 24:00:00
#      and run this script from inside the allocation.
#   2. Then:
#          export HF_TOKEN=hf_...                       # required
#          export DOCKER_IMAGE=<image:tag>              # required
#          export BACKEND=maxtext                       # required
#          export RESULT_DIR=~/primus-bench/mi325x-8N
#          bash tools/perf/run_batch_multinode.sh
#
#      Nothing in this file needs editing: every setting reads from the
#      environment, including the cluster networking defaults (NCCL_*),
#      which are Fremont values and should be overridden elsewhere.
#
#      GPU is optional: when unset it is read from the amd-smi / rocm-smi
#      product name. BACKEND is required because an image only supports some
#      backends.
#
# Choosing configs -- two modes, no paths to type:
#   Catalog mode (default)
#       Reads tools/perf/configs.yaml, keyed by GPU then backend. GPU picks
#       the top-level key, BACKEND the second-level keys (comma-separated).
#       Comment a line out to drop a model. CONFIG_FILE points at your own
#       copy for an ad-hoc set.
#   Directory mode
#       CONFIG_DIR=<dir> runs every *.yaml under <dir>, recursively. Renaming
#       one to *.yaml.done removes it from future runs.
#
# Optional: DRY_RUN=1 prints the selected configs and launch commands and
# exits without touching docker, srun or the GPUs. TRAIN_STEPS=20 caps
# training length for the whole batch; the CLI override flag is chosen from
# the config's framework (see below). Unset keeps whatever the YAML specifies.
# NUM_REPS defaults to 1.
#
# EXTRA_ENV="KEY=VALUE KEY2=VALUE2" forwards variables into the container as
# explicit `--env` flags. A plain export is NOT enough: primus-cli only
# forwards names listed in runner/.primus.yaml's container.options.env plus the
# PRIMUS_/NCCL_/RCCL_/GLOO_/IONIC_/HIPBLASLT_ prefixes, and silently drops the
# rest -- which is why an exported DEBUG_HIP_DYNAMIC_QUEUES never arrived.
#
# TRAIN_STEPS CLI flags (Primus deep-merges unknown tokens after the train verb):
#   megatron / megatron_bridge : --train_iters N
#   torchtitan                 : --training.steps N   (NOT --steps)
#   maxtext                    : --steps N
#   maxdiffusion               : --max_train_steps N  (NOT --steps; ignored if used)
#
# Per-config behaviour:
#   We keep launching the config until NUM_REPS successful runs are
#   collected, or until MAX_ATTEMPTS_PER_CONFIG total attempts have been
#   spent. Each *successful* attempt produces a log file in $RESULT_DIR:
#       <framework>-<config>-<hash>-MBS<m>-GBS<g>-rep<r>_<stamp>.log
#   Each *failed* attempt is preserved as
#       <framework>-<config>-<hash>-MBS<m>-GBS<g>-rep<r>-failed-attempt<n>_<stamp>.log
#   so nothing is silently dropped. Both contain the same structured
#   banner, the full YAML dump, raw stdout/stderr and a footer that records
#   the failure reason.
#
# Failure detection (lib/control.sh):
#   The training process is launched in the background in its own process
#   group and its log is checked every POLL_INTERVAL seconds. An attempt is
#   stopped and retried when FAIL_PATTERN appears in its output, when it runs
#   longer than MAX_RUN_SECONDS (default 3600), or when its log stops growing
#   for MAX_IDLE_SECONDS (default 1200); 0 disables either limit. Stopping an
#   attempt kills the launcher's process group, including the local srun, and
#   stops this batch's training containers on every node.
#
# Before the first attempt (and the GPU part before every attempt), on every
# node: the image is pulled if missing (always with PULL_IMAGE=1) and must be
# the same image on every node, containers that earlier batches of this user
# left running are stopped, and the GPUs must be idle (GPU_BUSY_MIB,
# ALLOW_BUSY_GPUS=1 to override); NCCL/Gloo must not use a bridge interface.
#
# Exit codes: 0 every config reached NUM_REPS good runs, 1 some did not,
# 2 bad input or a failed pre-run check (nothing launched), 130 interrupted.
#
# Batch-level outputs in $RESULT_DIR:
#     batch_summary_<stamp>.txt     - rolling summary, system info, per-run status.
#     batch_env_<stamp>.txt         - shell environment, secrets redacted.
#     batch_submodules_<stamp>.txt  - `git submodule status` (what code ran).
#     batch_gpu_<stamp>.txt         - GPU state of the node running this script.
#     batch_stack_<stamp>.txt       - image digest and ID, ROCm/HIP and library
#                                     versions from inside the container.
#     batch_host_<stamp>.txt        - every node's platform: server model and
#                                     BIOS, CPU, memory, GPU and driver, RDMA
#                                     NICs, OS and host settings, and the raw
#                                     data behind them; then the settings that
#                                     differ between nodes, if any.
#     batch_manifest_<stamp>/       - the image's /workspace/.manifest, if any.
#     run_batch_multinode_<stamp>.sh - a copy of THIS script as it was run.
#     results.csv, results.json     - every log in RESULT_DIR parsed by
#                                     extract_results.py, refreshed at the end
#                                     (EXTRACT=0 skips this).
#     report.xlsx                   - the Excel report of RESULT_DIR (report.py),
#                                     refreshed at the end when pandas and
#                                     XlsxWriter are installed (REPORT=0 skips it).
#   <stamp> is YYYYmmdd-HHMMSS, captured once when the script starts, so every
#   artefact of one batch shares it and repeated runs in the same RESULT_DIR
#   accumulate side by side instead of overwriting each other. Export BATCH_TS
#   to force a specific stamp (this WILL overwrite files that already use it).
###############################################################################

# Strict-ish mode: catch typos / unset vars, keep pipe exit codes accurate.
# We intentionally do NOT use `set -e`: a single failing benchmark must not
# abort the rest of the batch. No job control (+m): under it a background
# setsid forks, and the launcher's PID would no longer be its process group.
set -uo pipefail
set +m

# ----------------------------- User Settings ---------------------------------
# Every knob below is overridable from the environment, so a run never needs
# this file edited. HF_TOKEN is validated in the sanity-check block; it is
# deliberately not defaulted here.
export HSA_NO_SCRATCH_RECLAIM="${HSA_NO_SCRATCH_RECLAIM:-1}"
export PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32="${PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32:-1}"
export NVTE_CK_IS_V3_ATOMIC_FP32="${NVTE_CK_IS_V3_ATOMIC_FP32:-1}"
export REBUILD_BNXT="${REBUILD_BNXT:-1}"

# Cluster-specific networking. These defaults come from the Fremont cluster;
# override them for any other site.
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eno0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eno0}"
# On Fremont, NCCL_CROSS_NIC=1 with NCCL_PXN_DISABLE=0 caused JAX training to hang.
export NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-0}"
export NCCL_PXN_DISABLE="${NCCL_PXN_DISABLE:-1}"
# Node count: honor an explicit NNODES export, otherwise use the size of
# the current Slurm allocation (SLURM_NNODES / SLURM_JOB_NUM_NODES).
export NNODES="${NNODES:-${SLURM_NNODES:-${SLURM_JOB_NUM_NODES:-}}}"

# NCCL_* is auto-forwarded into the container by primus-cli, so a plain export
# works for those:
#   export NCCL_DEBUG=INFO
#
# Anything outside that passthrough set -- DEBUG_HIP_DYNAMIC_QUEUES is the
# usual example -- must go through EXTRA_ENV or it never reaches the container:
#   EXTRA_ENV="DEBUG_HIP_DYNAMIC_QUEUES=0" bash tools/perf/run_batch_multinode.sh

# Required: which image to benchmark is the whole point of a run, and a stale
# default would silently measure the wrong build. Validated below.
export DOCKER_IMAGE="${DOCKER_IMAGE:-}"

# Where this batch writes its logs. Created if missing.
RESULT_DIR="${RESULT_DIR:-$PWD/primus-perf-$(date +%Y%m%d)-${NNODES:-N}N}"

# Which configs to run. See the header: GPU/BACKEND filter the catalog,
# CONFIG_DIR switches to directory mode.
GPU="${GPU:-}"
BACKEND="${BACKEND:-}"
CONFIG_FILE="${CONFIG_FILE:-}"
CONFIG_DIR="${CONFIG_DIR:-}"

# How many *successful* runs we want per config and the cap on total
# attempts (good + failed) we are willing to spend chasing them.
# Matches the single-node default so the two behave the same unless asked.
NUM_REPS=${NUM_REPS:-1}
MAX_ATTEMPTS_PER_CONFIG=${MAX_ATTEMPTS_PER_CONFIG:-6}

# Optional training-length override for every config in this batch.
# Empty means "use the YAML". Must be a positive integer when set.
# Backend flag is selected from the config's framework (see header).
TRAIN_STEPS="${TRAIN_STEPS:-}"

DRY_RUN="${DRY_RUN:-0}"

# Failure-detection knobs.
#
# POLL_INTERVAL : how often we look at the training log.
# FAIL_PATTERN  : extended-regex (ERE). If ANY of these alternatives is seen
#                 in the training output (post-banner), the attempt is killed
#                 and retried. We default to a *broad* set of indicators
#                 because some failures (e.g. NCCL "remote process exited"
#                 followed by an endless HeartbeatMonitor / TCPStore "Broken
#                 pipe" / "shut down too early" spam) never produce
#                 'primus/cli/main.py FAILED' but still need to be killed.
# MAX_RUN_SECONDS : hard wall-clock cap per attempt; 0 disables it.
# MAX_IDLE_SECONDS: cap on how long the log may go without new output, which
#                 is how a hang shows up; 0 disables it.
POLL_INTERVAL=${POLL_INTERVAL:-15}
FAIL_PATTERN=${FAIL_PATTERN:-'primus/cli/main\.py FAILED|ChildFailedError|ncclRemoteError|Did the remote server shutdown|TCPStore.*Broken pipe|srun: error:.*Exited with exit code'}
MAX_RUN_SECONDS=${MAX_RUN_SECONDS:-3600}
MAX_IDLE_SECONDS=${MAX_IDLE_SECONDS:-1200}

# Path to Primus repo root and the CLI entry point. The repo root comes from
# git rather than this script's location, because the script lives in
# tools/perf/ rather than at the repo root.
PRIMUS_PATH=${PRIMUS_PATH:-"$(git -C "$(dirname "$(realpath "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel 2>/dev/null || (cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/../.." && pwd))"}
PRIMUS_CLI=${PRIMUS_CLI:-"./primus-cli"}

# Multi-node launcher mode. The CLI runs the container on each of the
# $NNODES allocated nodes via `slurm srun`.
PRIMUS_MODE="${PRIMUS_MODE:-slurm srun}"

# ----------------------------- Derived paths ---------------------------------
SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"
SCRIPT_NAME="$(basename "$SCRIPT_PATH")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"

# Sourced at runtime from the script's own directory; shellcheck cannot
# resolve the paths statically.
# shellcheck source=/dev/null
source "$SCRIPT_DIR/lib/common.sh"
# Multi-node layout for the shared banner and system snapshot.
PERF_MULTINODE=1
# shellcheck source=/dev/null
source "$SCRIPT_DIR/lib/run.sh"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/lib/control.sh"

# Absolute from here on. The batch later does `cd "$PRIMUS_PATH"`, so a
# relative RESULT_DIR would put the summary and the per-run logs in two
# different directories depending on where the script was invoked from.
# The directory itself is only created once DRY_RUN is ruled out.
RESULT_DIR="$(realpath -m "$RESULT_DIR")"

# Every artefact this batch writes is stamped with one timestamp captured here at
# start-up, so a second batch in the same RESULT_DIR cannot overwrite the first
# batch's records. The human-readable start time is derived from the same epoch
# value so the two can never disagree across a second boundary.
BATCH_START_TS=$(date +%s)
BATCH_TS="${BATCH_TS:-$(date -d "@$BATCH_START_TS" '+%Y%m%d-%H%M%S')}"
BATCH_START=$(date -d "@$BATCH_START_TS" '+%Y-%m-%d %H:%M:%S %Z')

SUMMARY_FILE="$RESULT_DIR/batch_summary_${BATCH_TS}.txt"
ENV_FILE="$RESULT_DIR/batch_env_${BATCH_TS}.txt"
SCRIPT_COPY="$RESULT_DIR/${SCRIPT_NAME%.sh}_${BATCH_TS}.sh"

perf_init_container_label

# ----------------------------- Sanity Checks ---------------------------------
# Required, never defaulted: a token baked into a shared script leaks the
# moment the script is copied next to the results.
if [ -z "${HF_TOKEN:-}" ] && [ "$DRY_RUN" != 1 ]; then
    echo "[ERROR] HF_TOKEN is required; export your Hugging Face token before running." >&2
    exit "$PERF_EXIT_REFUSED"
fi
export HF_TOKEN="${HF_TOKEN:-}"

# Required: benchmarking the wrong build because a default went stale is worse
# than being told to name the image.
if [ -z "$DOCKER_IMAGE" ] && [ "$DRY_RUN" != 1 ]; then
    echo "[ERROR] DOCKER_IMAGE is required; export the image:tag to benchmark." >&2
    exit "$PERF_EXIT_REFUSED"
fi

# yq is required. When it is missing (or is the unrelated Python `yq`), a
# private copy is fetched into ~/.cache/primus-perf/bin -- no root needed and
# no change to the user's PATH beyond this run.
if ! perf_ensure_yq; then
    echo "[ERROR] 'yq' is required and could not be installed automatically." >&2
    echo "        See https://github.com/mikefarah/yq for manual installation." >&2
    exit "$PERF_EXIT_REFUSED"
fi

if [ "$DRY_RUN" != 1 ] && ! command -v srun >/dev/null 2>&1; then
    echo "[ERROR] 'srun' not found in PATH. Allocate nodes (salloc) first." >&2
    exit "$PERF_EXIT_REFUSED"
fi

if [ -z "${SLURM_JOB_ID:-}" ] && [ "$DRY_RUN" != 1 ]; then
    echo "[WARN] SLURM_JOB_ID is not set; are you inside a salloc/sbatch allocation?" >&2
fi

if ! [[ "${NNODES:-}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] NNODES is unset or not a positive integer: '${NNODES:-}'" >&2
    echo "        Run inside a Slurm allocation (SLURM_NNODES is set) or export NNODES=<n>." >&2
    exit "$PERF_EXIT_REFUSED"
fi

if [ -n "${SLURM_NNODES:-}" ] && [ "$NNODES" -ne "$SLURM_NNODES" ]; then
    echo "[WARN] NNODES=$NNODES differs from SLURM_NNODES=$SLURM_NNODES (using NNODES)." >&2
fi

shopt -s nullglob

# Select the configs to run: catalog (default, filtered by GPU/BACKEND) or a
# directory of YAMLs (CONFIG_DIR). EXTRA_ENV, missing paths and duplicates are
# all reported here, before anything launches.
if ! check_batch_inputs || ! perf_check_run_limits; then
    exit "$PERF_EXIT_REFUSED"
fi
if [ "$PERF_RUN_DIFFUSION_DLRM" = 1 ]; then
    echo "[ERROR] diffusion_dlrm runs single-node benchmarks; use run_batch.sh for it." >&2
    exit "$PERF_EXIT_REFUSED"
fi

# ---------------------------- Helper functions -------------------------------

# Build the primus-cli argv for one config. Extra tokens after --config are
# deep-merged into the overrides of the module being trained (CLI > YAML).
build_launch_argv() {
    local config_file="$1"
    local framework="$2"
    local suite="$3"

    if [ -z "$suite" ]; then
        echo "[ERROR] no modules.pre_trainer or modules.post_trainer found in: $config_file" >&2
        echo "        A Primus experiment YAML must declare one of them." >&2
        return 1
    fi

    # The explicit `-- container <entry args> -- <primus args>` form documented
    # by primus-cli-slurm.sh carries the container label, and EXTRA_ENV as
    # entry-level --env flags.
    LAUNCH_ARGV=(
        "$PRIMUS_CLI" slurm srun -N "$NNODES"
        -- container --label "${PERF_LABEL_KEY}=${PERF_LABEL_VALUE}"
        ${PERF_EXTRA_ENV_ARGS[@]+"${PERF_EXTRA_ENV_ARGS[@]}"}
        -- train "$suite" --config "$config_file"
    )

    append_train_steps_override "$config_file" "$framework"
}

if [ "$DRY_RUN" = 1 ]; then
    perf_print_plan
    exit "$PERF_EXIT_OK"
fi

if ! mkdir -p "$RESULT_DIR" 2>/dev/null; then
    echo "[ERROR] RESULT_DIR does not exist and could not be created: $RESULT_DIR" >&2
    exit "$PERF_EXIT_REFUSED"
fi

trap perf_on_interrupt INT TERM HUP

# ---------------------------- Batch Execution --------------------------------
# Preserve the exact script that drove this batch before running anything, so
# an interrupted batch still records what it was doing.
cp -f "$SCRIPT_PATH" "$SCRIPT_COPY"

# Initialise summary and env files.
: > "$SUMMARY_FILE"
collect_system_info | tee -a "$SUMMARY_FILE"

# Image, leftover containers and idle GPUs on every node, then the NCCL
# interface. Done before the batch configuration is written so the image
# digest below is already known.
if ! perf_preflight; then
    {
        echo ""
        echo "################################################################################"
        echo "# Primus Multi-Node Benchmark Batch REFUSED: a pre-run check failed (see above)"
        echo "# Nothing was launched."
        echo "################################################################################"
    } | tee -a "$SUMMARY_FILE"
    exit "$PERF_EXIT_REFUSED"
fi

# Batch-level provenance: redacted env, submodule pins, GPU state, host
# settings, and the library versions from inside the container. Taken before
# the batch configuration below, which shows the container's ROCm and HIP.
perf_write_batch_snapshots "$RESULT_DIR" "$BATCH_TS" "${DOCKER_IMAGE:-}"

{
    echo ""
    echo "################################################################################"
    echo "# Batch Configuration"
    echo "################################################################################"
    echo "# Start time            : $BATCH_START"
    echo "# Batch stamp           : $BATCH_TS"
    echo "# RESULT_DIR            : $RESULT_DIR"
    echo "# Config source         : $CONFIG_SOURCE"
    echo "# PRIMUS_MODE           : $PRIMUS_MODE"
    echo "# Docker image          : ${DOCKER_IMAGE:-<unset; container mode uses runner/.primus.yaml>}"
    echo "# Image digest          : $(perf_image_digest "${DOCKER_IMAGE:-}")"
    echo "# Image ID              : ${PERF_IMAGE_ID:-$(perf_image_id "${DOCKER_IMAGE:-}")}  ($PERF_IMAGE_CHECK)"
    echo "# ROCm (host)           : $(perf_rocm_version)"
    echo "# ROCm (image)          : ${PERF_CONTAINER_ROCM:-unknown}  (HIP ${PERF_CONTAINER_HIP:-unknown})"
    echo "# System                : ${PERF_SYSTEM_DESCRIPTION:-unknown}"
    echo "# Host check            : ${PERF_HOST_CHECK:-not checked}"
    echo "# Launcher mode         : multi-node (primus-cli slurm srun -N $NNODES)"
    echo "# Container label       : ${PERF_LABEL_KEY}=${PERF_LABEL_VALUE}"
    echo "# TRAIN_STEPS           : ${TRAIN_STEPS:-<unset; use YAML>}"
    echo "# NUM_REPS (target)     : $NUM_REPS"
    echo "# MAX_ATTEMPTS_PER_CFG  : $MAX_ATTEMPTS_PER_CONFIG"
    echo "# Run limits            : $(perf_run_limits_text)"
    echo "# FAIL_PATTERN          : ${FAIL_PATTERN:-<none>}"
    echo "# GPU check             : $PERF_GPU_CHECK"
    echo "# NCCL iface            : $PERF_NIC_CHECK"
    # CONFIG_FILES is populated by perf_select_configs in lib/common.sh.
    # shellcheck disable=SC2153
    echo "# Total configs         : ${#CONFIG_FILES[@]}"
    echo "# Configs to run (resolved path + content hash):"
    for f in "${CONFIG_FILES[@]}"; do
        echo "#   - $(sha256sum "$f" | cut -c1-8)  $f"
    done
    echo "################################################################################"
    echo ""
} | tee -a "$SUMMARY_FILE"

cd "$PRIMUS_PATH" || { echo "[ERROR] cannot cd into $PRIMUS_PATH" >&2; exit "$PERF_EXIT_REFUSED"; }

RUNS_STARTED=0
RUNS_GOOD=0
RUNS_FAILED=0
CONFIGS_SKIPPED=0
BATCH_STOPPED=""

for config_file in "${CONFIG_FILES[@]}"; do
    [ -z "$BATCH_STOPPED" ] || break

    prepare_config "$config_file"

    if ! build_launch_argv "$config_file" "$FRAMEWORK" "$TRAIN_SUITE"; then
        CONFIGS_SKIPPED=$((CONFIGS_SKIPPED + 1))
        echo "[SKIP/FAIL] ${FRAMEWORK}/${config_name}: cannot build launch command" \
            | tee -a "$SUMMARY_FILE"
        continue
    fi
    launcher_cmd=$(perf_display_argv "${LAUNCH_ARGV[@]}")

    good_runs=0
    attempts=0

    echo "" | tee -a "$SUMMARY_FILE"
    echo "==============================================================" | tee -a "$SUMMARY_FILE"
    echo "[$(date '+%H:%M:%S')] Model: ${FRAMEWORK}/${config_name} [${config_hash}]" | tee -a "$SUMMARY_FILE"
    echo "  target=${NUM_REPS} good runs   max_attempts=${MAX_ATTEMPTS_PER_CONFIG}   steps=${STEPS}" | tee -a "$SUMMARY_FILE"
    echo "==============================================================" | tee -a "$SUMMARY_FILE"

    while [[ $good_runs -lt $NUM_REPS && $attempts -lt $MAX_ATTEMPTS_PER_CONFIG ]]; do
        # Stop this batch's containers from a failed attempt, stop leftovers,
        # and make sure nothing else holds the GPUs. The first attempt was
        # checked by perf_preflight a moment ago.
        if [ "$RUNS_STARTED" -gt 0 ] && ! perf_prepare_gpus; then
            BATCH_STOPPED="the GPUs were busy before attempt $((RUNS_STARTED + 1))"
            break
        fi

        attempts=$((attempts + 1))
        rep=$((good_runs + 1))

        # "Active" log file: written into during the run, renamed afterwards
        # if the attempt failed so we never lose a failed-attempt log either.
        result_file=$(run_log_name "$rep")

        echo "" | tee -a "$SUMMARY_FILE"
        echo "[$(date '+%H:%M:%S')] [attempt ${attempts}/${MAX_ATTEMPTS_PER_CONFIG}] ${FRAMEWORK}/${config_name} steps=${STEPS} rep ${rep}/${NUM_REPS} (good so far: ${good_runs}/${NUM_REPS})" \
            | tee -a "$SUMMARY_FILE"
        echo "                    -> $result_file" | tee -a "$SUMMARY_FILE"

        print_per_run_banner "$result_file" "$config_file" "$config_name" "$config_hash" \
            "$FRAMEWORK" "$MODEL" "$MBS" "$GBS" "$SEQ_LEN" "$STEPS" \
            "$rep" "$NUM_REPS" "$launcher_cmd" "$MODULE" "$TRAIN_SUITE" \
            "$attempts" "$MAX_ATTEMPTS_PER_CONFIG"

        RUNS_STARTED=$((RUNS_STARTED + 1))
        perf_launch_with_watchdog "$result_file" "$(( $(wc -l < "$result_file") + 1 ))"

        if [ "$RUN_FAILURE_REASON" != "none" ]; then
            RUNS_FAILED=$((RUNS_FAILED + 1))
            failed_file=$(run_log_name "$rep" "-failed-attempt${attempts}")
            mv "$result_file" "$failed_file"
            echo "                    [FAIL] ${RUN_FAILURE_REASON} exit=${RUN_EXIT_CODE} elapsed=${RUN_ELAPSED}s" \
                | tee -a "$SUMMARY_FILE"
            echo "                           -> $failed_file" | tee -a "$SUMMARY_FILE"
        else
            good_runs=$((good_runs + 1))
            RUNS_GOOD=$((RUNS_GOOD + 1))
            echo "                    [ OK ] rep ${rep}/${NUM_REPS} done in ${RUN_ELAPSED}s (good ${good_runs}/${NUM_REPS})" \
                | tee -a "$SUMMARY_FILE"
        fi
    done

    if [ -n "$BATCH_STOPPED" ]; then
        echo "[${FRAMEWORK}/${config_name}] STOPPED -- $BATCH_STOPPED" | tee -a "$SUMMARY_FILE"
    elif [[ $good_runs -ge $NUM_REPS ]]; then
        echo "[${FRAMEWORK}/${config_name}] COMPLETE — $good_runs successful runs in $attempts attempts" \
            | tee -a "$SUMMARY_FILE"
    else
        echo "[${FRAMEWORK}/${config_name}] GAVE UP — only $good_runs/$NUM_REPS successful runs after $attempts attempts" \
            | tee -a "$SUMMARY_FILE"
    fi
done

BATCH_END=$(date '+%Y-%m-%d %H:%M:%S %Z')
BATCH_END_TS=$(date +%s)
BATCH_ELAPSED=$((BATCH_END_TS - BATCH_START_TS))

EXPECTED_GOOD=$(( ${#CONFIG_FILES[@]} * NUM_REPS ))

{
    echo ""
    echo "################################################################################"
    if [ -n "$BATCH_STOPPED" ]; then
        echo "# Primus Multi-Node Benchmark Batch STOPPED: $BATCH_STOPPED"
    else
        echo "# Primus Multi-Node Benchmark Batch Complete"
    fi
    echo "################################################################################"
    echo "# End time      : $BATCH_END"
    echo "# Elapsed (sec) : $BATCH_ELAPSED"
    echo "# Configs       : ${#CONFIG_FILES[@]}"
    echo "# Target good   : $EXPECTED_GOOD  ($NUM_REPS x ${#CONFIG_FILES[@]})"
    echo "# Good runs     : $RUNS_GOOD"
    echo "# Failed runs   : $RUNS_FAILED"
    echo "# Total attempts: $RUNS_STARTED"
    echo "# Skipped cfgs  : $CONFIGS_SKIPPED"
    echo "################################################################################"
} | tee -a "$SUMMARY_FILE"

# The batch driver was copied before the first run, so an interrupted batch
# still has it; just report the artefacts here.
echo "Saved batch driver -> $SCRIPT_COPY" | tee -a "$SUMMARY_FILE"
echo "Saved env snapshot -> $ENV_FILE"    | tee -a "$SUMMARY_FILE"
echo "Saved provenance   -> $RESULT_DIR/batch_{submodules,gpu,stack,host}_${BATCH_TS}.txt" | tee -a "$SUMMARY_FILE"
perf_extract_results
echo "Saved summary      -> $SUMMARY_FILE"

if [ "$RUNS_GOOD" -ge "$EXPECTED_GOOD" ]; then
    exit "$PERF_EXIT_OK"
else
    exit "$PERF_EXIT_RUN_FAILED"
fi
