#!/bin/bash
###############################################################################
# Primus Perf Batch Runner (single node)
#
# Purpose:
#   Run a batch of Primus training benchmarks (Megatron, Megatron Bridge,
#   TorchTitan, MaxText, MaxDiffusion), and optionally the diffusion and DLRM
#   benchmarks of diffusion_bench/, and collect the results in a single
#   directory with enough metadata that a reviewer can reproduce or audit each
#   run from the logs alone.
#
# Usage:
#   export HF_TOKEN=hf_...                       # required
#   export DOCKER_IMAGE=<image:tag>              # required
#   export BACKEND=megatron,torchtitan           # required
#   export RESULT_DIR=~/primus-bench/mi325x-v26.7
#   bash tools/perf/run_batch.sh
#
#   Nothing in this file needs editing: every setting below reads from the
#   environment. The script can be invoked from any directory.
#
#   GPU is optional: when unset it is read from the amd-smi / rocm-smi product
#   name ("AMD Instinct MI325X" -> MI325X). Set it to run another device's
#   configs. BACKEND is required because an image only supports some
#   backends -- a torch image has no JAX -- so defaulting to "all" would queue
#   impossible runs.
#
# Choosing configs -- two modes, no paths to type:
#   Catalog mode (default)
#       Reads tools/perf/configs.yaml, the curated release suite, keyed by GPU
#       then backend. GPU picks the top-level key, BACKEND the second-level
#       keys (comma-separated). Comment a line out in the catalog to drop that
#       model from the run. Point CONFIG_FILE at your own copy for an ad-hoc set.
#   Directory mode
#       CONFIG_DIR=<dir> runs every *.yaml under <dir>, recursively and sorted.
#       Renaming a config to *.yaml.done removes it from future runs, which is
#       how you resume a batch that died partway.
#
#   Missing paths and duplicate entries are reported before anything launches.
#
# Diffusion and DLRM (BACKEND=...,diffusion_dlrm):
#   Not a catalog backend. Adding it to BACKEND runs diffusion_bench/
#   run_benchmarks.sh once, after the catalog configs, for every diffusion and
#   DLRM model its config enables (batch sizes, models and step_timeout come
#   from that config; the runner sets no time limit on this run). It uses
#   DOCKER_IMAGE, which must contain /workspace/AMDiffusionBenchmark and
#   /workspace/DLRMBenchmark (rocm/primus images do), and writes its results
#   folder under $RESULT_DIR/diffusion_dlrm. For example, everything the
#   PyTorch image runs:
#     BACKEND=megatron,torchtitan,megatron_bridge,diffusion_dlrm
#   DIFFUSION_DLRM_CONFIG=<file>   diffusion_bench config to use instead of
#                                  diffusion_bench/config.yaml
#   DIFFUSION_DLRM_HF_CACHE=<dir>  Hugging Face cache for the diffusion models
#                                  (default: hf_cache_dir in that config)
#   TRAIN_STEPS sets the diffusion models' num_iterations; DLRM keeps its own.
#
# Optional knobs:
#   DRY_RUN=1        print the selected configs and launch commands, then exit
#                    without touching docker or the GPUs. HF_TOKEN and
#                    DOCKER_IMAGE are not required.
#   TRAIN_STEPS=20   cap training length for the whole batch (smoke / short
#                    bench). The CLI override flag is chosen from the config's
#                    framework (see below). Unset -- the default -- keeps
#                    whatever the YAML specifies.
#   NUM_REPS=3       repetitions per config (default 1).
#   MAX_RUN_SECONDS  stop a run that is still going after this long
#                    (default 3600; 0 = no limit).
#   MAX_IDLE_SECONDS stop a run whose log has not grown for this long, which
#                    is how a hang shows up (default 1200; 0 = no limit).
#   POLL_INTERVAL    how often the log is checked, in seconds (default 15).
#   GPU_BUSY_MIB     a GPU with more memory in use than this counts as busy,
#                    and the run is refused (default 2048).
#   GPU_BUSY_WAIT_SECONDS  how long busy GPUs may take to free up before the
#                    run is refused (default 60).
#   ALLOW_BUSY_GPUS=1  run even when the GPUs are busy.
#   PULL_IMAGE=1     pull DOCKER_IMAGE before the batch even if the tag is
#                    present, replacing a stale or re-tagged local copy.
#   EXTRACT=0        do not refresh results.csv / results.json at the end.
#   EXTRA_ENV=...    space-separated KEY=VALUE pairs forwarded into the
#                    container as explicit `--env` flags, e.g.
#                    EXTRA_ENV="DEBUG_HIP_DYNAMIC_QUEUES=0". A plain export is
#                    NOT enough: primus-cli only forwards names listed in
#                    runner/.primus.yaml's container.options.env plus the
#                    PRIMUS_/NCCL_/RCCL_/GLOO_/IONIC_/HIPBLASLT_ prefixes, and
#                    drops everything else without warning.
#   PRIMUS_MODE      launcher mode, "container" (default) or "direct" when the
#                    runner itself already runs inside a container.
#
# Before the first run (and the GPU part before every run):
#   * the image is pulled if it is not present locally;
#   * containers that earlier batches of this user left running are stopped
#     (only ones this tool labelled; see lib/control.sh);
#   * the GPUs must be idle, or the batch refuses to start or stops;
#   * NCCL/Gloo must not bootstrap over docker0 or another bridge.
#
# Pre-training vs post-training:
#   A Primus experiment YAML declares exactly one training module, and the
#   module decides the CLI verb and where the overrides live:
#     modules.pre_trainer  -> `train pretrain`   (all frameworks)
#     modules.post_trainer -> `train posttrain`  (Megatron Bridge SFT / LoRA,
#                             e.g. examples/megatron_bridge/configs/*/
#                             qwen3_32b_{sft,lora}_posttrain.yaml)
#   The module is detected per config, so a single batch can mix both.
#
# TRAIN_STEPS CLI flags (Primus deep-merges unknown tokens after the train verb):
#   megatron / megatron_bridge : --train_iters N
#   torchtitan                 : --training.steps N   (NOT --steps)
#   maxtext                    : --steps N
#   maxdiffusion               : --max_train_steps N  (NOT --steps; ignored if used)
#
# Per-config behaviour:
#   The script runs each config NUM_REPS times (default 1). For every run it
#   writes a log file in $RESULT_DIR with:
#     * A structured info banner (timestamp, host, framework, MBS/GBS, etc.).
#     * The full YAML config used, dumped verbatim.
#     * The raw stdout/stderr of the primus-cli invocation.
#     * A footer with exit code, elapsed seconds and the failure reason.
#
# Exit codes: 0 every run passed, 1 a run failed or was not started, 2 bad
# input or a failed pre-run check (nothing launched), 130 interrupted.
#
# Batch-level outputs in $RESULT_DIR:
#     batch_summary_<stamp>.txt     - rolling summary, system info, per-run status.
#     batch_env_<stamp>.txt         - shell environment, secrets redacted.
#     batch_submodules_<stamp>.txt  - `git submodule status` (what code ran).
#     batch_gpu_<stamp>.txt         - rocm-smi / amd-smi clocks, power, topology.
#     batch_stack_<stamp>.txt       - image digest and ID, ROCm/HIP and library
#                                     versions from inside the container.
#     batch_host_<stamp>.txt        - CPU governor, NUMA balancing, THP, IOMMU,
#                                     amdgpu driver, network interfaces.
#     batch_manifest_<stamp>/       - the image's /workspace/.manifest (build
#                                     recipe, dpkg and pip lists), if it has one.
#     run_batch_<stamp>.sh          - a copy of THIS script as it was run,
#                                     written before the first run starts.
#     <framework>-<config>-<hash>-MBS<m>-GBS<g>-rep<r>_<stamp>.log - per-run logs.
#     results.csv, results.json     - every log in RESULT_DIR parsed by
#                                     extract_results.py, refreshed at the end.
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
# These three are in runner/.primus.yaml's container.options.env, so exporting
# them here does reach the container. Do not add arbitrary variables alongside
# them: anything outside that list and the PRIMUS_/NCCL_/RCCL_/GLOO_/IONIC_/
# HIPBLASLT_ prefixes is dropped without warning. Use EXTRA_ENV instead, e.g.
#   EXTRA_ENV="DEBUG_HIP_DYNAMIC_QUEUES=0" bash tools/perf/run_batch.sh
export HSA_NO_SCRATCH_RECLAIM="${HSA_NO_SCRATCH_RECLAIM:-1}"
export PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32="${PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32:-1}"
export NVTE_CK_IS_V3_ATOMIC_FP32="${NVTE_CK_IS_V3_ATOMIC_FP32:-1}"

PRIMUS_MODE="${PRIMUS_MODE:-container}"

# Required: which image to benchmark is the whole point of a run, and a stale
# default would silently measure the wrong build. Validated below.
export DOCKER_IMAGE="${DOCKER_IMAGE:-}"

# Where this batch writes its logs. Created if missing.
RESULT_DIR="${RESULT_DIR:-$PWD/primus-perf-$(date +%Y%m%d)}"

# Which configs to run. See the header: GPU/BACKEND filter the catalog,
# CONFIG_DIR switches to directory mode.
GPU="${GPU:-}"
BACKEND="${BACKEND:-}"
CONFIG_FILE="${CONFIG_FILE:-}"
CONFIG_DIR="${CONFIG_DIR:-}"

# Number of repetitions per config.
NUM_REPS=${NUM_REPS:-1}

# Optional training-length override for every config in this batch.
# Empty (the default) means "use whatever the YAML specifies". Must be a
# positive integer when set. Backend flag is selected from the config's
# framework (see header).
TRAIN_STEPS="${TRAIN_STEPS:-}"

DRY_RUN="${DRY_RUN:-0}"

# Run limits (see header). A single-node run that fails exits on its own, so
# unlike the multi-node runner there is no FAIL_PATTERN here.
MAX_RUN_SECONDS="${MAX_RUN_SECONDS:-3600}"
MAX_IDLE_SECONDS="${MAX_IDLE_SECONDS:-1200}"
POLL_INTERVAL="${POLL_INTERVAL:-15}"
FAIL_PATTERN=""

# Cluster topology (forwarded to the Primus launcher via env).
export NNODES=${NNODES:-1}
export NODE_RANK=${NODE_RANK:-0}
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-1234}

# ----------------------------- Derived paths ---------------------------------
SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"
SCRIPT_NAME="$(basename "$SCRIPT_PATH")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"

# Repo root. Derived from git rather than from this script's location, because
# the script lives in tools/perf/ rather than at the repo root.
PRIMUS_PATH=${PRIMUS_PATH:-"$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || (cd "$SCRIPT_DIR/../.." && pwd))"}
PRIMUS_CLI=${PRIMUS_CLI:-"./primus-cli"}

# Sourced at runtime from the script's own directory; shellcheck cannot
# resolve the paths statically.
# shellcheck source=/dev/null
source "$SCRIPT_DIR/lib/common.sh"
# Single-node layout for the shared banner and system snapshot.
PERF_MULTINODE=0
# shellcheck source=/dev/null
source "$SCRIPT_DIR/lib/run.sh"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/lib/control.sh"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/lib/diffusion_dlrm.sh"

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

shopt -s nullglob

# Select the configs to run: catalog (default, filtered by GPU/BACKEND) or a
# directory of YAMLs (CONFIG_DIR). EXTRA_ENV, missing paths and duplicates are
# all reported here, before anything launches.
if ! check_batch_inputs || ! perf_check_run_limits || ! perf_dd_prepare; then
    exit "$PERF_EXIT_REFUSED"
fi

# ---------------------------- Helper functions -------------------------------

# Build the primus-cli argv for one config. Extra tokens after --config are
# deep-merged into the overrides of the module being trained (CLI > YAML).
build_launch_argv() {
    local config_file="$1"
    local framework="$2"
    local suite="$3"
    local label_args=()

    if [ -z "$suite" ]; then
        echo "[ERROR] no modules.pre_trainer or modules.post_trainer found in: $config_file" >&2
        echo "        A Primus experiment YAML must declare one of them." >&2
        return 1
    fi

    # The label lets cleanup find exactly this batch's container.
    if perf_uses_containers; then
        label_args=( --label "${PERF_LABEL_KEY}=${PERF_LABEL_VALUE}" )
    fi

    # $PRIMUS_MODE is intentionally unquoted so a value like "container" or
    # "direct" (or a short extra mode flag string) still splits as intended.
    # EXTRA_ENV becomes explicit `--env KEY=VALUE` flags before the `--`, which
    # is the only way to be sure a variable reaches the container.
    # shellcheck disable=SC2206
    LAUNCH_ARGV=( "$PRIMUS_CLI" $PRIMUS_MODE ${label_args[@]+"${label_args[@]}"} ${PERF_EXTRA_ENV_ARGS[@]+"${PERF_EXTRA_ENV_ARGS[@]}"} -- train "$suite" --config "$config_file" )

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

# Image, leftover containers, idle GPUs, NCCL interface, and what diffusion_dlrm
# needs. Done before the batch configuration is written so the image digest
# below is already known.
if ! perf_preflight || ! perf_dd_preflight; then
    {
        echo ""
        echo "################################################################################"
        echo "# Primus Benchmark Batch REFUSED: a pre-run check failed (see above)"
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
    echo "# Start time   : $BATCH_START"
    echo "# Batch stamp  : $BATCH_TS"
    echo "# RESULT_DIR   : $RESULT_DIR"
    echo "# Config source: $CONFIG_SOURCE"
    echo "# PRIMUS_MODE  : $PRIMUS_MODE"
    echo "# Docker image : ${DOCKER_IMAGE:-<unset; container mode uses runner/.primus.yaml>}"
    echo "# Image digest : $(perf_image_digest "${DOCKER_IMAGE:-}")"
    echo "# Image ID     : $(perf_image_id "${DOCKER_IMAGE:-}")"
    echo "# ROCm (host)  : $(perf_rocm_version)"
    echo "# ROCm (image) : ${PERF_CONTAINER_ROCM:-unknown}  (HIP ${PERF_CONTAINER_HIP:-unknown})"
    echo "# Reps/config  : $NUM_REPS"
    echo "# TRAIN_STEPS  : ${TRAIN_STEPS:-<unset; use YAML>}"
    echo "# Run limits   : $(perf_run_limits_text)"
    echo "# GPU check    : $PERF_GPU_CHECK"
    echo "# NCCL iface   : $PERF_NIC_CHECK"
    if perf_uses_containers; then
        echo "# Label        : ${PERF_LABEL_KEY}=${PERF_LABEL_VALUE}"
    fi
    # CONFIG_FILES is populated by perf_select_configs in lib/common.sh.
    # shellcheck disable=SC2153
    echo "# Total configs: ${#CONFIG_FILES[@]}"
    echo "# Total runs   : $(( ${#CONFIG_FILES[@]} * NUM_REPS + PERF_DD_RUNS ))"
    echo "# Configs to run (resolved path + content hash):"
    for f in ${CONFIG_FILES[@]+"${CONFIG_FILES[@]}"}; do
        echo "#   - $(sha256sum "$f" | cut -c1-8)  $f"
    done
    if [ "$PERF_DD_RUNS" -gt 0 ]; then
        perf_dd_config_lines
    fi
    echo "################################################################################"
    echo ""
} | tee -a "$SUMMARY_FILE"

cd "$PRIMUS_PATH" || { echo "[ERROR] cannot cd into $PRIMUS_PATH" >&2; exit "$PERF_EXIT_REFUSED"; }

RUNS_STARTED=0
RUNS_FAILED=0
TOTAL_PLANNED=$(( ${#CONFIG_FILES[@]} * NUM_REPS + PERF_DD_RUNS ))
BATCH_STOPPED=""

for config_file in ${CONFIG_FILES[@]+"${CONFIG_FILES[@]}"}; do
    [ -z "$BATCH_STOPPED" ] || break

    prepare_config "$config_file"

    if ! build_launch_argv "$config_file" "$FRAMEWORK" "$TRAIN_SUITE"; then
        RUNS_STARTED=$((RUNS_STARTED + NUM_REPS))
        RUNS_FAILED=$((RUNS_FAILED + NUM_REPS))
        echo "[SKIP/FAIL] ${FRAMEWORK}/${config_name}: cannot build launch command" \
            | tee -a "$SUMMARY_FILE"
        continue
    fi
    launcher_cmd="${LAUNCH_ARGV[*]}"

    for rep in $(seq 1 "$NUM_REPS"); do
        # The first run was checked by perf_preflight a moment ago.
        if [ "$RUNS_STARTED" -gt 0 ] && ! perf_prepare_gpus; then
            BATCH_STOPPED="the GPUs were busy before run $((RUNS_STARTED + 1))"
            break
        fi

        result_file=$(run_log_name "$rep")

        echo "[$(date '+%H:%M:%S')] [run $((RUNS_STARTED + 1))/${TOTAL_PLANNED}] ${FRAMEWORK}/${TRAIN_SUITE}/${config_name} [${config_hash}] MBS=${MBS} GBS=${GBS} steps=${STEPS} rep ${rep}/${NUM_REPS}" \
            | tee -a "$SUMMARY_FILE"
        echo "                    -> $result_file" | tee -a "$SUMMARY_FILE"

        print_per_run_banner "$result_file" "$config_file" "$config_name" "$config_hash" \
            "$FRAMEWORK" "$MODEL" "$MBS" "$GBS" "$SEQ_LEN" "$STEPS" "$rep" "$NUM_REPS" \
            "$launcher_cmd" "$MODULE" "$TRAIN_SUITE"

        RUNS_STARTED=$((RUNS_STARTED + 1))
        perf_launch_with_watchdog "$result_file" "$(( $(wc -l < "$result_file") + 1 ))"

        if [ "$RUN_FAILURE_REASON" != "none" ]; then
            RUNS_FAILED=$((RUNS_FAILED + 1))
            echo "                    [FAIL] ${RUN_FAILURE_REASON} exit=${RUN_EXIT_CODE} elapsed=${RUN_ELAPSED}s" \
                | tee -a "$SUMMARY_FILE"
        else
            echo "                    [ OK ] exit=${RUN_EXIT_CODE} elapsed=${RUN_ELAPSED}s" \
                | tee -a "$SUMMARY_FILE"
        fi
    done
done

# diffusion_dlrm: one diffusion_bench run per repetition, after everything else.
if [ "$PERF_DD_RUNS" -gt 0 ]; then
    perf_dd_launch_argv
    launcher_cmd="${LAUNCH_ARGV[*]}"
    dd_config_name=$(perf_dd_config_name)
    dd_config_hash=$(config_fingerprint "$DIFFUSION_DLRM_CONFIG")
    dd_models=$(perf_dd_summary)
    dd_steps="${TRAIN_STEPS:+$TRAIN_STEPS (diffusion num_iterations; DLRM unchanged)}"
    for rep in $(seq 1 "$NUM_REPS"); do
        [ -z "$BATCH_STOPPED" ] || break
        if [ "$RUNS_STARTED" -gt 0 ] && ! perf_prepare_gpus; then
            BATCH_STOPPED="the GPUs were busy before run $((RUNS_STARTED + 1))"
            break
        fi

        result_file=$(perf_dd_log_name "$rep")

        echo "[$(date '+%H:%M:%S')] [run $((RUNS_STARTED + 1))/${TOTAL_PLANNED}] diffusion_dlrm/${dd_config_name} [${dd_config_hash}] ${dd_models} rep ${rep}/${NUM_REPS}" \
            | tee -a "$SUMMARY_FILE"
        echo "                    -> $result_file" | tee -a "$SUMMARY_FILE"

        PERF_RUN_LIMITS_TEXT="$(perf_dd_limits_text)" \
            print_per_run_banner "$result_file" "$DIFFUSION_DLRM_CONFIG" "$dd_config_name" "$dd_config_hash" \
            "$PERF_DIFFUSION_DLRM_BACKEND" "$dd_models" "N/A" "N/A" "N/A" "${dd_steps:-per diffusion_bench config}" \
            "$rep" "$NUM_REPS" "$launcher_cmd" "diffusion_bench" ""

        RUNS_STARTED=$((RUNS_STARTED + 1))
        # Read by the interrupt handler, which then also runs perf_dd_finalize.
        # shellcheck disable=SC2034
        PERF_DD_ACTIVE=1
        # For this run only: no runner time limit, and a failure reason that
        # names the steps that failed.
        MAX_RUN_SECONDS=0 MAX_IDLE_SECONDS=0 PERF_REASON_FN=perf_dd_failure_reason \
            perf_launch_with_watchdog "$result_file" "$(( $(wc -l < "$result_file") + 1 ))"
        perf_dd_finalize
        # shellcheck disable=SC2034
        PERF_DD_ACTIVE=0

        if [ "$RUN_FAILURE_REASON" != "none" ]; then
            RUNS_FAILED=$((RUNS_FAILED + 1))
            echo "                    [FAIL] ${RUN_FAILURE_REASON} exit=${RUN_EXIT_CODE} elapsed=${RUN_ELAPSED}s" \
                | tee -a "$SUMMARY_FILE"
        else
            echo "                    [ OK ] exit=${RUN_EXIT_CODE} elapsed=${RUN_ELAPSED}s" \
                | tee -a "$SUMMARY_FILE"
        fi
        perf_dd_report_steps "$result_file"
    done
fi

BATCH_END=$(date '+%Y-%m-%d %H:%M:%S %Z')
BATCH_END_TS=$(date +%s)
BATCH_ELAPSED=$((BATCH_END_TS - BATCH_START_TS))
RUNS_NOT_STARTED=$((TOTAL_PLANNED - RUNS_STARTED))

{
    echo ""
    echo "################################################################################"
    if [ -n "$BATCH_STOPPED" ]; then
        echo "# Primus Benchmark Batch STOPPED: $BATCH_STOPPED"
    else
        echo "# Primus Benchmark Batch Complete"
    fi
    echo "################################################################################"
    echo "# End time     : $BATCH_END"
    echo "# Elapsed (sec): $BATCH_ELAPSED"
    echo "# Total runs   : $RUNS_STARTED"
    echo "# Failed runs  : $RUNS_FAILED"
    echo "# Not started  : $RUNS_NOT_STARTED"
    echo "################################################################################"
} | tee -a "$SUMMARY_FILE"

# The batch driver was copied before the first run, so an interrupted batch
# still has it; just report the artefacts here.
echo "Saved batch driver -> $SCRIPT_COPY" | tee -a "$SUMMARY_FILE"
echo "Saved env snapshot -> $ENV_FILE"    | tee -a "$SUMMARY_FILE"
echo "Saved provenance   -> $RESULT_DIR/batch_{submodules,gpu,stack,host}_${BATCH_TS}.txt" | tee -a "$SUMMARY_FILE"
perf_extract_results
echo "Saved summary      -> $SUMMARY_FILE"

if [ "$RUNS_FAILED" -eq 0 ] && [ "$RUNS_NOT_STARTED" -eq 0 ]; then
    exit "$PERF_EXIT_OK"
else
    exit "$PERF_EXIT_RUN_FAILED"
fi
