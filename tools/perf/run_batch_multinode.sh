#!/bin/bash
###############################################################################
# Primus Perf Batch Runner (multi-node, with retry on failure)
#
# Purpose:
#   Drive a batch of Primus *multi-node* pretrain benchmarks via the
#   `primus-cli slurm srun` launcher, collect the results with enough
#   metadata that a reviewer can reproduce/audit each run from the logs
#   alone, and retry training that crashes mid-step.
#
# This script is the multi-node sibling of `primus-bench-batch.sh`:
#   - structure, banners and summary format follow that script;
#   - retry-on-failure logic (poll log for failure pattern, kill, restart,
#     keep going until we have NUM_REPS successful runs) is borrowed from
#     `mi325x-8n-batch-run-retry.sh`.
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
#      GPU is optional: when unset it is read from the rocm-smi product name.
#      BACKEND is required because an image only supports some backends.
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
# Optional: TRAIN_STEPS=20 caps training length for the whole batch; the CLI
# override flag is chosen from the config's framework (see below). Unset keeps
# whatever the YAML specifies. NUM_REPS defaults to 1.
#
# EXTRA_ENV="KEY=VALUE KEY2=VALUE2" forwards variables into the container as
# explicit `--env` flags. A plain export is NOT enough: primus-cli only
# forwards names listed in runner/.primus.yaml's container.options.env plus the
# PRIMUS_/NCCL_/RCCL_/GLOO_/IONIC_/HIPBLASLT_ prefixes, and silently drops the
# rest -- which is why an exported DEBUG_HIP_DYNAMIC_QUEUES never arrived.
#
# TRAIN_STEPS CLI flags (Primus deep-merges unknown tokens after `train pretrain`):
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
#   banner, the full YAML dump, raw stdout/stderr and a footer.
#
# Failure detection:
#   The training process is launched in the background and we tail its log.
#   Every POLL_INTERVAL seconds we also grep the log for FAIL_PATTERN.
#   If we hit the pattern we kill the launcher and retry; on every failure
#   we also `srun docker stop` the Primus container on all nodes so a
#   stale process doesn't hold the GPUs for the next attempt.
#
# Batch-level outputs in $RESULT_DIR:
#     batch_summary_<stamp>.txt - rolling summary, system info, per-run status.
#     batch_env_<stamp>.txt     - snapshot of the shell environment.
#     primus-bench-batch-multinode_<stamp>.sh - a copy of THIS script as run.
#     configs/                  - the configs you provided (left untouched).
#   <stamp> is YYYYmmdd-HHMMSS, captured once when the script starts, so every
#   artefact of one batch shares it and repeated runs in the same RESULT_DIR
#   accumulate side by side instead of overwriting each other. Export BATCH_TS
#   to force a specific stamp (this WILL overwrite files that already use it).
###############################################################################

# Strict-ish mode: catch typos / unset vars, keep pipe exit codes accurate.
# We intentionally do NOT use `set -e`: a single failing benchmark must not
# abort the rest of the batch.
set -uo pipefail

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
# Backend flag is selected from modules.pre_trainer.framework (see header).
TRAIN_STEPS="${TRAIN_STEPS:-}"

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
# MAX_RUN_SECONDS: hard wall-clock cap per attempt. Even if no pattern
#                 matches, any attempt running longer than this is killed.
#                 This is the script's last-resort defense against hangs.
POLL_INTERVAL=${POLL_INTERVAL:-15}
FAIL_PATTERN=${FAIL_PATTERN:-'primus/cli/main\.py FAILED|ChildFailedError|ncclRemoteError|Did the remote server shutdown|TCPStore.*Broken pipe|srun: error:.*Exited with exit code'}
MAX_RUN_SECONDS=${MAX_RUN_SECONDS:-2400}

# Path to Primus repo root and the CLI entry point. The repo root comes from
# git rather than this script's location, because the script lives in
# tools/perf/ rather than at the repo root.
PRIMUS_PATH=${PRIMUS_PATH:-"$(git -C "$(dirname "$(realpath "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel 2>/dev/null || (cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/../.." && pwd))"}
PRIMUS_CLI=${PRIMUS_CLI:-"./primus-cli"}

# Name of the docker container Primus spawns on each node; used by the
# cleanup paths so a hung, failed or interrupted attempt doesn't hold the GPUs.
#
# primus-cli does not use the bare base name: runner/primus-cli-container.sh
# appends "-${SLURM_JOB_ID:-$$}" to whatever runner/.primus.yaml configures.
# Under Slurm the job id is identical on every node of the allocation and
# unique across jobs, so the full name is both predictable and specific to
# this batch -- another job's containers can never match it.
PRIMUS_CONTAINER_PREFIX="${PRIMUS_CONTAINER_PREFIX:-primus-training}"
if [ -n "${PRIMUS_CONTAINER:-}" ]; then
    : # caller pinned an explicit name
elif [ -n "${SLURM_JOB_ID:-}" ]; then
    PRIMUS_CONTAINER="${PRIMUS_CONTAINER_PREFIX}-${SLURM_JOB_ID}"
else
    # No allocation: the suffix is a per-node pid we cannot predict, so
    # stop_containers falls back to prefix matching against a snapshot.
    PRIMUS_CONTAINER=""
fi

# Multi-node launcher mode. The CLI runs the container on each of the
# $NNODES allocated nodes via `slurm srun`.
PRIMUS_MODE="${PRIMUS_MODE:-slurm srun}"

# ----------------------------- Derived paths ---------------------------------
SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"
SCRIPT_NAME="$(basename "$SCRIPT_PATH")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"

# Sourced at runtime from the script's own directory; shellcheck cannot
# resolve the path statically.
# shellcheck source=/dev/null
source "$SCRIPT_DIR/lib/common.sh"
# Multi-node layout for the shared banner, footer and system snapshot.
PERF_MULTINODE=1
# shellcheck source=/dev/null
source "$SCRIPT_DIR/lib/run.sh"

# Absolute from here on. The batch later does `cd "$PRIMUS_PATH"`, so a
# relative RESULT_DIR would put the summary and the per-run logs in two
# different directories depending on where the script was invoked from.
mkdir -p "$RESULT_DIR" 2>/dev/null || true
RESULT_DIR="$(realpath "$RESULT_DIR" 2>/dev/null || echo "$RESULT_DIR")"

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

# ----------------------------- Sanity Checks ---------------------------------
if [ -z "$RESULT_DIR" ]; then
    echo "[ERROR] RESULT_DIR is not set. Export RESULT_DIR before running." >&2
    exit 1
fi

if [ ! -d "$RESULT_DIR" ]; then
    echo "[ERROR] RESULT_DIR does not exist and could not be created: $RESULT_DIR" >&2
    exit 1
fi

# Required, never defaulted: a token baked into a shared script leaks the
# moment the script is copied next to the results.
: "${HF_TOKEN:?HF_TOKEN is required; export your Hugging Face token before running}"
export HF_TOKEN

# Required: benchmarking the wrong build because a default went stale is worse
# than being told to name the image.
: "${DOCKER_IMAGE:?DOCKER_IMAGE is required; export the image:tag to benchmark}"

# yq is required. When it is missing (or is the unrelated Python `yq`), a
# private copy is fetched into ~/.cache/primus-perf/bin -- no root needed and
# no change to the user's PATH beyond this run.
if ! perf_ensure_yq; then
    echo "[ERROR] 'yq' is required and could not be installed automatically." >&2
    echo "        See https://github.com/mikefarah/yq for manual installation." >&2
    exit 1
fi

if ! command -v srun >/dev/null 2>&1; then
    echo "[ERROR] 'srun' not found in PATH. Allocate nodes (salloc) first." >&2
    exit 1
fi

if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "[WARN] SLURM_JOB_ID is not set; are you inside a salloc/sbatch allocation?" >&2
fi

if ! [[ "${NNODES:-}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] NNODES is unset or not a positive integer: '${NNODES:-}'" >&2
    echo "        Run inside a Slurm allocation (SLURM_NNODES is set) or export NNODES=<n>." >&2
    exit 1
fi

if [ -n "${SLURM_NNODES:-}" ] && [ "$NNODES" -ne "$SLURM_NNODES" ]; then
    echo "[WARN] NNODES=$NNODES differs from SLURM_NNODES=$SLURM_NNODES (using NNODES)." >&2
fi

shopt -s nullglob

# Select the configs to run: catalog (default, filtered by GPU/BACKEND) or a
# directory of YAMLs (CONFIG_DIR). EXTRA_ENV, missing paths and duplicates are
# all reported here, before anything launches.
if ! check_batch_inputs; then
    exit 1
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

    # With no EXTRA_ENV, keep the shorthand that routes through the default
    # container entry. With EXTRA_ENV, switch to the explicit
    # `-- container <entry args> -- <primus args>` form documented by
    # primus-cli-slurm.sh, which is where entry-level --env flags belong.
    if [ ${#PERF_EXTRA_ENV_ARGS[@]} -gt 0 ]; then
        LAUNCH_ARGV=(
            "$PRIMUS_CLI" slurm srun -N "$NNODES"
            -- container "${PERF_EXTRA_ENV_ARGS[@]}"
            -- train "$suite" --config "$config_file"
        )
    else
        LAUNCH_ARGV=( "$PRIMUS_CLI" slurm srun -N "$NNODES" -- train "$suite" --config "$config_file" )
    fi

    append_train_steps_override "$config_file" "$framework"
}

# Training containers matching the prefix that were already up before this
# batch started, across all allocated nodes. Only consulted in the no-Slurm
# fallback below; with a job id the container name is already unambiguous.
PREEXISTING_CONTAINERS=""

snapshot_preexisting_containers() {
    [ -n "$PRIMUS_CONTAINER" ] && return 0
    PREEXISTING_CONTAINERS=$(
        srun -N "$NNODES" docker ps -q --filter "name=^/${PRIMUS_CONTAINER_PREFIX}" 2>/dev/null |
            tr '\n' ' '
    )
}

# Multi-node cleanup: stop this batch's training container on every node.
# Never blocks the script if the container isn't running.
stop_containers() {
    if [ -n "$PRIMUS_CONTAINER" ]; then
        srun -N "$NNODES" docker stop -t 20 "$PRIMUS_CONTAINER" >/dev/null 2>&1 || true
        return 0
    fi

    # No SLURM_JOB_ID, so container names carry unpredictable per-node pids.
    # Match on the prefix instead, skipping anything that was already running:
    # nodes get shared, and another job's container must survive our cleanup.
    #
    # Single quotes are deliberate: this body runs on the remote nodes, and the
    # exclusion list and prefix are passed as $1 / $2 rather than interpolated.
    # shellcheck disable=SC2016
    srun -N "$NNODES" bash -c '
        pre=" $1 "
        for id in $(docker ps -q --filter "name=^/${2}" 2>/dev/null); do
            case "$pre" in *" $id "*) continue ;; esac
            docker stop -t 20 "$id" >/dev/null 2>&1 || true
        done
    ' _ "$PREEXISTING_CONTAINERS" "$PRIMUS_CONTAINER_PREFIX" >/dev/null 2>&1 || true
}

# Kill the *entire* process group rooted at $1 (the PID returned by `$!`
# after we launch primus-cli via setsid). With setsid, that PID is also
# the process-group ID, so `kill -- -PGID` reaches the bash subshell,
# primus-cli, and the local srun that is holding the SLURM step. Without
# this, killing only the bash subshell leaves srun (and its remote tasks)
# orphaned and spamming the log forever.
kill_process_group() {
    local pid="$1"
    local sig="${2:-TERM}"
    [[ -z "$pid" ]] && return 0
    # First try the group (PGID == PID when we used setsid).
    kill -"$sig" -- "-$pid" 2>/dev/null || true
    # Fall back to the bare PID in case setsid wasn't available.
    kill -"$sig" "$pid" 2>/dev/null || true
}

# Aggressive cleanup of a failed/timed-out attempt. Tries (in order):
#   1) SIGTERM the launcher's whole process group, then SIGKILL after a
#      short grace period. With setsid (see launch site below), this
#      reaches the bash subshell, primus-cli AND the local srun process
#      that is holding the SLURM step open. Killing local srun causes
#      slurmstepd on every remote node to reap the remote tasks.
#   2) docker stop the Primus training container on every node, in case
#      anything inside the container is still alive after step 1.
# We intentionally do NOT `scancel` the SLURM job: that would tear down
# the user's allocation, but we want to keep the allocation so that the
# next retry / next config can reuse it.
kill_attempt() {
    local pid="$1"
    if [[ -n "$pid" ]]; then
        kill_process_group "$pid" TERM
        # Give children a moment to handle SIGTERM and write any final logs.
        sleep 10
        kill_process_group "$pid" KILL
    fi
    stop_containers
}

# ---------------------------- Signal handling --------------------------------
# These are mutated as we go through the loop so the trap can clean up
# the in-flight attempt if the user hits Ctrl+C.
train_pid=""
tail_pid=""
current_log=""

# Invoked indirectly via the `trap` below, so shellcheck cannot see the call
# site and reports the whole body as unreachable.
# shellcheck disable=SC2317,SC2329
cleanup() {
    echo "" >&2
    echo ">>> Signal received, shutting down..." >&2
    if [[ -n "$tail_pid" ]]; then
        kill "$tail_pid" 2>/dev/null
    fi
    if [[ -n "$train_pid" ]]; then
        echo ">>> Killing launcher process group (pgid=$train_pid)..." >&2
        kill_attempt "$train_pid"
    else
        stop_containers
    fi
    if [[ -n "$current_log" ]]; then
        echo ">>> In-flight attempt log: $current_log" >&2
    fi
    echo ">>> Cleanup done. Exiting." >&2
    exit 130
}
trap cleanup SIGINT SIGTERM SIGHUP

# ---------------------------- Batch Execution --------------------------------
# Preserve the exact script that drove this batch before running anything, so
# an interrupted batch still records what it was doing.
cp -f "$SCRIPT_PATH" "$SCRIPT_COPY"

# Initialise summary and env files.
: > "$SUMMARY_FILE"
collect_system_info | tee -a "$SUMMARY_FILE"

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
    echo "# Launcher mode         : multi-node (primus-cli slurm srun -N $NNODES)"
    echo "# Container to stop     : ${PRIMUS_CONTAINER:-<no SLURM_JOB_ID; matching on ${PRIMUS_CONTAINER_PREFIX}* prefix>}"
    echo "# TRAIN_STEPS           : ${TRAIN_STEPS:-<unset; use YAML>}"
    echo "# NUM_REPS (target)     : $NUM_REPS"
    echo "# MAX_ATTEMPTS_PER_CFG  : $MAX_ATTEMPTS_PER_CONFIG"
    echo "# POLL_INTERVAL (sec)   : $POLL_INTERVAL"
    echo "# MAX_RUN_SECONDS       : $MAX_RUN_SECONDS  (wall-clock cap per attempt)"
    echo "# FAIL_PATTERN          : $FAIL_PATTERN"
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

# Batch-level provenance: redacted env, submodule pins, GPU state, and the
# library versions from inside the container.
perf_write_batch_snapshots "$RESULT_DIR" "$BATCH_TS" "${DOCKER_IMAGE:-}"

# Record which training containers are already up, so cleanup only stops the
# ones this batch starts. No-op when the job id makes the name unambiguous.
snapshot_preexisting_containers
if [ -n "$PREEXISTING_CONTAINERS" ]; then
    {
        echo "[INFO] pre-existing ${PRIMUS_CONTAINER_PREFIX}* containers will be left alone:"
        echo "         $PREEXISTING_CONTAINERS"
    } | tee -a "$SUMMARY_FILE"
fi

cd "$PRIMUS_PATH" || { echo "[ERROR] cannot cd into $PRIMUS_PATH" >&2; exit 1; }

TOTAL_ATTEMPTS=0
TOTAL_GOOD=0
TOTAL_FAILED=0

for config_file in "${CONFIG_FILES[@]}"; do
    # `config_name` keeps the path relative to the repo root, so the banner and
    # summary show where the config came from rather than a bare basename that
    # is ambiguous across framework folders. The extractor tolerates a path:
    # `_split_model_precision` takes the last segment before parsing.
    config_rel="${config_file#"$PRIMUS_PATH"/}"
    config_name="${config_rel%.yaml}"
    config_name="${config_name%.yml}"

    # For the on-disk log filename we use only the basename (no subfolder),
    # since the `${fw_safe}-` prefix already disambiguates same-named YAMLs
    # that live under different framework folders.
    config_basename=$(basename "$config_file")
    config_basename="${config_basename%.yaml}"
    config_basename="${config_basename%.yml}"

    extract_config_fields "$config_file"
    YAML_STEPS="$STEPS"
    if [ -n "$TRAIN_STEPS" ]; then
        STEPS="$TRAIN_STEPS (CLI override; yaml=$YAML_STEPS)"
    fi

    if ! build_launch_argv "$config_file" "$FRAMEWORK" "$TRAIN_SUITE"; then
        echo "[SKIP/FAIL] ${FRAMEWORK}/${config_name}: cannot apply TRAIN_STEPS (unknown framework)" \
            | tee -a "$SUMMARY_FILE"
        continue
    fi
    launcher_cmd="${LAUNCH_ARGV[*]}"

    fw_safe=$(sanitize "$FRAMEWORK")
    name_safe=$(sanitize "$config_basename")
    mbs_safe=$(sanitize "$MBS")
    gbs_safe=$(sanitize "$GBS")
    config_hash=$(config_fingerprint "$config_file")

    good_runs=0
    attempts=0

    echo "" | tee -a "$SUMMARY_FILE"
    echo "==============================================================" | tee -a "$SUMMARY_FILE"
    echo "[$(date '+%H:%M:%S')] Model: ${FRAMEWORK}/${config_name} [${config_hash}]" | tee -a "$SUMMARY_FILE"
    echo "  target=${NUM_REPS} good runs   max_attempts=${MAX_ATTEMPTS_PER_CONFIG}   steps=${STEPS}" | tee -a "$SUMMARY_FILE"
    echo "==============================================================" | tee -a "$SUMMARY_FILE"

    while [[ $good_runs -lt $NUM_REPS && $attempts -lt $MAX_ATTEMPTS_PER_CONFIG ]]; do
        attempts=$((attempts + 1))
        TOTAL_ATTEMPTS=$((TOTAL_ATTEMPTS + 1))
        rep=$((good_runs + 1))

        # "Active" log file: written into during the run, renamed afterwards
        # if the attempt failed so we never lose a failed-attempt log either.
        result_file="${RESULT_DIR}/${fw_safe}-${name_safe}-${config_hash}-MBS${mbs_safe}-GBS${gbs_safe}-rep${rep}_${BATCH_TS}.log"

        echo "" | tee -a "$SUMMARY_FILE"
        echo "[$(date '+%H:%M:%S')] [attempt ${attempts}/${MAX_ATTEMPTS_PER_CONFIG}] ${FRAMEWORK}/${config_name} steps=${STEPS} rep ${rep}/${NUM_REPS} (good so far: ${good_runs}/${NUM_REPS})" \
            | tee -a "$SUMMARY_FILE"
        echo "                    -> $result_file" | tee -a "$SUMMARY_FILE"

        print_per_run_banner "$result_file" "$config_file" "$config_name" "$config_hash" \
            "$FRAMEWORK" "$MODEL" "$MBS" "$GBS" "$SEQ_LEN" "$STEPS" \
            "$rep" "$NUM_REPS" "$launcher_cmd" "$MODULE" "$TRAIN_SUITE" \
            "$attempts" "$MAX_ATTEMPTS_PER_CONFIG"

        # Remember how many lines the banner + config dump occupy so we
        # can scan only the training output for FAIL_PATTERN and avoid
        # false positives from our own banner or the dumped YAML config.
        banner_lines=$(wc -l < "$result_file")
        grep_from_line=$((banner_lines + 1))

        # Belt-and-braces: make sure no zombie container from a prior
        # failed attempt is still holding the GPUs on any node.
        stop_containers
        sleep 2

        current_log="$result_file"
        run_start_ts=$(date +%s)

        # Launch the multi-node training in the background **inside its
        # own process group via setsid**. With a setsid'd child, the PID we
        # capture (`$!`) is also the PGID, so a `kill -- -PGID` reaches
        # the bash subshell, primus-cli AND the local srun process. Just
        # `kill $train_pid` would only terminate the bash subshell and
        # leave the SLURM step + remote workers running orphaned -- which
        # is exactly the hang we saw in
        # megatron-llama3.1_8B-BF16-pretrain-...-rep1.log (5.3M lines of
        # NCCL/TCPStore spam after the script could not stop the run).
        # Output goes through perf_redact_stream so secret values never reach
        # the log: primus-cli echoes the container command it runs, which
        # carries `--env HF_TOKEN=<value>`. Process substitution rather than a
        # pipe, because `$!` has to stay the setsid PID for the kill above to
        # reach the whole process group.
        if command -v setsid >/dev/null 2>&1; then
            setsid "${LAUNCH_ARGV[@]}" \
                > >(perf_redact_stream >> "$result_file") 2>&1 &
        else
            # Fallback if setsid is missing: at least we still have the
            # pattern-based detection and wall-clock timeout below.
            "${LAUNCH_ARGV[@]}" \
                > >(perf_redact_stream >> "$result_file") 2>&1 &
        fi
        train_pid=$!

        # Mirror the log to the terminal in real time. `-F` keeps tailing
        # across rename/truncate (we don't rename until after we stop it,
        # but it's a safer default than `-f`).
        tail -F "$result_file" 2>/dev/null &
        tail_pid=$!

        error_detected=false
        timeout_kill=false
        while kill -0 "$train_pid" 2>/dev/null; do
            sleep "$POLL_INTERVAL"

            # (1) Failure pattern in training output? Skip the banner /
            # YAML dump (lines 1..banner_lines) so we only match training
            # output. FAIL_PATTERN is an extended regex (alternation).
            if tail -n +"$grep_from_line" "$result_file" 2>/dev/null \
                | grep -qE -- "$FAIL_PATTERN"; then
                {
                    echo ""
                    echo ">>> [$(date '+%H:%M:%S')] ERROR DETECTED — failure pattern seen in training output."
                    echo ">>> Killing launcher process group (pgid=$train_pid) and stopping containers..."
                } | tee -a "$result_file" "$SUMMARY_FILE"
                kill_attempt "$train_pid"
                error_detected=true
                break
            fi

            # (2) Wall-clock watchdog. Hard cap on the time any single
            # attempt may take. This is the last-resort defense for
            # scenarios where (a) FAIL_PATTERN never appears, or (b) some
            # ranks are stuck in NCCL HeartbeatMonitor / TCPStore loops
            # that spam the log forever without surfacing a recognisable
            # error.
            now_ts=$(date +%s)
            elapsed=$((now_ts - run_start_ts))
            if [[ $elapsed -gt $MAX_RUN_SECONDS ]]; then
                {
                    echo ""
                    echo ">>> [$(date '+%H:%M:%S')] TIMEOUT — attempt exceeded MAX_RUN_SECONDS=${MAX_RUN_SECONDS}s (elapsed=${elapsed}s)."
                    echo ">>> Killing launcher process group (pgid=$train_pid) and stopping containers..."
                } | tee -a "$result_file" "$SUMMARY_FILE"
                kill_attempt "$train_pid"
                error_detected=true
                timeout_kill=true
                break
            fi
        done

        wait "$train_pid" 2>/dev/null
        exit_code=$?

        kill "$tail_pid" 2>/dev/null
        wait "$tail_pid" 2>/dev/null
        train_pid=""
        tail_pid=""

        run_end_ts=$(date +%s)
        run_elapsed=$((run_end_ts - run_start_ts))

        print_per_run_footer "$result_file" "$exit_code" "$run_elapsed" "$error_detected"

        if $error_detected || [ "$exit_code" -ne 0 ]; then
            TOTAL_FAILED=$((TOTAL_FAILED + 1))
            failed_file="${RESULT_DIR}/${fw_safe}-${name_safe}-${config_hash}-MBS${mbs_safe}-GBS${gbs_safe}-rep${rep}-failed-attempt${attempts}_${BATCH_TS}.log"
            mv "$result_file" "$failed_file"
            if $timeout_kill; then
                reason="timeout (>${MAX_RUN_SECONDS}s)"
            elif $error_detected; then
                reason="pattern hit"
            else
                reason="non-zero exit"
            fi
            echo "                    [FAIL] $reason exit=${exit_code} elapsed=${run_elapsed}s" \
                | tee -a "$SUMMARY_FILE"
            echo "                           -> $failed_file" | tee -a "$SUMMARY_FILE"
            stop_containers
            sleep 5
        else
            good_runs=$((good_runs + 1))
            TOTAL_GOOD=$((TOTAL_GOOD + 1))
            echo "                    [ OK ] rep ${rep}/${NUM_REPS} done in ${run_elapsed}s (good ${good_runs}/${NUM_REPS})" \
                | tee -a "$SUMMARY_FILE"
        fi
        current_log=""
    done

    if [[ $good_runs -ge $NUM_REPS ]]; then
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
    echo "# Primus Multi-Node Benchmark Batch Complete"
    echo "################################################################################"
    echo "# End time      : $BATCH_END"
    echo "# Elapsed (sec) : $BATCH_ELAPSED"
    echo "# Configs       : ${#CONFIG_FILES[@]}"
    echo "# Target good   : $EXPECTED_GOOD  ($NUM_REPS x ${#CONFIG_FILES[@]})"
    echo "# Good runs     : $TOTAL_GOOD"
    echo "# Failed runs   : $TOTAL_FAILED"
    echo "# Total attempts: $TOTAL_ATTEMPTS"
    echo "################################################################################"
} | tee -a "$SUMMARY_FILE"

# The batch driver was copied before the first run, so an interrupted batch
# still has it; just report the artefacts here.
echo "Saved batch driver -> $SCRIPT_COPY" | tee -a "$SUMMARY_FILE"
echo "Saved env snapshot -> $ENV_FILE"    | tee -a "$SUMMARY_FILE"
echo "Saved provenance   -> $RESULT_DIR/batch_{submodules,gpu,stack}_${BATCH_TS}.txt" | tee -a "$SUMMARY_FILE"
echo "Saved summary      -> $SUMMARY_FILE"

if [ "$TOTAL_GOOD" -ge "$EXPECTED_GOOD" ]; then
    exit 0
else
    exit 1
fi
