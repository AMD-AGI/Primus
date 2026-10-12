#!/bin/bash
###############################################################################
# Primus Perf Batch Runner (single node)
#
# Purpose:
#   Run a batch of Primus training benchmarks (Megatron, Megatron Bridge,
#   TorchTitan, MaxText, MaxDiffusion) and collect the results in a single
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
#   GPU is optional: when unset it is read from the rocm-smi product name
#   ("AMD Instinct MI325X" -> MI325X). Set it to run another device's configs.
#   BACKEND is required because an image only supports some backends -- a
#   torch image has no JAX -- so defaulting to "all" would queue impossible
#   runs.
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
# Optional knobs:
#   TRAIN_STEPS=20   cap training length for the whole batch (smoke / short
#                    bench). The CLI override flag is chosen from the config's
#                    framework (see below). Unset -- the default -- keeps
#                    whatever the YAML specifies.
#   NUM_REPS=3       repetitions per config (default 1).
#   EXTRA_ENV=...    space-separated KEY=VALUE pairs forwarded into the
#                    container as explicit `--env` flags, e.g.
#                    EXTRA_ENV="DEBUG_HIP_DYNAMIC_QUEUES=0". A plain export is
#                    NOT enough: primus-cli only forwards names listed in
#                    runner/.primus.yaml's container.options.env plus the
#                    PRIMUS_/NCCL_/RCCL_/GLOO_/IONIC_/HIPBLASLT_ prefixes, and
#                    drops everything else without warning.
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
#     * A footer with exit code and elapsed seconds.
#
# Batch-level outputs in $RESULT_DIR:
#     batch_summary_<stamp>.txt     - rolling summary, system info, per-run status.
#     batch_env_<stamp>.txt         - shell environment, secrets redacted.
#     batch_submodules_<stamp>.txt  - `git submodule status` (what code ran).
#     batch_gpu_<stamp>.txt         - rocm-smi clocks / power / topology.
#     batch_stack_<stamp>.txt       - image digest + library versions from
#                                     inside the container.
#     run_batch_<stamp>.sh          - a copy of THIS script as it was run,
#                                     written before the first run starts.
#     <framework>-<config>-<hash>-MBS<m>-GBS<g>-rep<r>_<stamp>.log - per-run logs.
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
# resolve the path statically.
# shellcheck source=/dev/null
source "$SCRIPT_DIR/lib/common.sh"

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

shopt -s nullglob

# Select the configs to run: catalog (default, filtered by GPU/BACKEND) or a
# directory of YAMLs (CONFIG_DIR). Missing paths and duplicates are reported
# here, before anything launches.
# EXTRA_ENV -> explicit --env flags, validated before anything launches.
if ! perf_parse_extra_env; then
    exit 1
fi

if ! perf_select_configs; then
    exit 1
fi

if [ ! -d "$PRIMUS_PATH" ]; then
    echo "[ERROR] PRIMUS_PATH does not exist: $PRIMUS_PATH" >&2
    exit 1
fi

# Resolve absolute path of the CLI (handles default './primus-cli').
PRIMUS_CLI_ABS="$PRIMUS_PATH/${PRIMUS_CLI#./}"
if [ ! -x "$PRIMUS_CLI_ABS" ]; then
    echo "[ERROR] primus-cli not found or not executable at: $PRIMUS_CLI_ABS" >&2
    exit 1
fi

if [ -n "$TRAIN_STEPS" ]; then
    if ! [[ "$TRAIN_STEPS" =~ ^[1-9][0-9]*$ ]]; then
        echo "[ERROR] TRAIN_STEPS must be a positive integer, got: '$TRAIN_STEPS'" >&2
        exit 1
    fi
fi

# ---------------------------- Helper functions -------------------------------

# Read a YAML field with yq. Returns "N/A" if absent/null/empty.
yq_or_na() {
    local expr="$1"
    local file="$2"
    local val
    val=$(yq -r "$expr" "$file" 2>/dev/null)
    if [ -z "$val" ] || [ "$val" = "null" ]; then
        echo "N/A"
    else
        echo "$val"
    fi
}

# Make a string safe to use in a filename (no slashes, no spaces).
sanitize() {
    local v="$1"
    v="${v//\//_}"
    v="${v// /_}"
    echo "$v"
}

# 8-char SHA-256 fingerprint of the config file contents. Lets the filename
# disambiguate two runs of "the same" config when its contents change between
# batches (or when the same config name is reused with edits).
config_fingerprint() {
    local file="$1"
    sha256sum "$file" | cut -c1-8
}

# Which training module a config declares. Primus experiment YAMLs carry
# exactly one: `pre_trainer` for pretraining, `post_trainer` for the Megatron
# Bridge SFT/LoRA recipes. Everything else in this script keys off the answer,
# because the module fixes both the CLI verb and the overrides path.
detect_train_module() {
    local config_file="$1"
    if [ "$(yq -r '.modules | has("pre_trainer")' "$config_file" 2>/dev/null)" = "true" ]; then
        echo "pre_trainer"
    elif [ "$(yq -r '.modules | has("post_trainer")' "$config_file" 2>/dev/null)" = "true" ]; then
        echo "post_trainer"
    else
        echo "N/A"
    fi
}

# `train <suite>` verb that drives a given module (primus/cli/subcommands/train.py).
train_suite_for_module() {
    case "$1" in
        pre_trainer)  echo "pretrain"  ;;
        post_trainer) echo "posttrain" ;;
        *)            echo ""          ;;
    esac
}

# Extract per-framework key fields from a config file. Sets globals
# MODULE, TRAIN_SUITE, FRAMEWORK, MODEL, MBS, GBS, SEQ_LEN, STEPS for the caller.
extract_config_fields() {
    local config_file="$1"

    MODULE=$(detect_train_module "$config_file")
    TRAIN_SUITE=$(train_suite_for_module "$MODULE")

    FRAMEWORK=$(yq_or_na ".modules.$MODULE.framework" "$config_file")
    MODEL=$(yq_or_na ".modules.$MODULE.model" "$config_file")

    case "$FRAMEWORK" in
        megatron)
            MBS=$(yq_or_na ".modules.$MODULE.overrides.micro_batch_size" "$config_file")
            GBS=$(yq_or_na ".modules.$MODULE.overrides.global_batch_size" "$config_file")
            SEQ_LEN=$(yq_or_na ".modules.$MODULE.overrides.seq_length" "$config_file")
            STEPS=$(yq_or_na ".modules.$MODULE.overrides.train_iters" "$config_file")
            ;;
        torchtitan)
            MBS=$(yq_or_na ".modules.$MODULE.overrides.training.local_batch_size" "$config_file")
            GBS=$(yq_or_na ".modules.$MODULE.overrides.training.global_batch_size" "$config_file")
            SEQ_LEN=$(yq_or_na ".modules.$MODULE.overrides.training.seq_len" "$config_file")
            STEPS=$(yq_or_na ".modules.$MODULE.overrides.training.steps" "$config_file")
            ;;
        maxtext)
            # MaxText (JAX) uses per_device_batch_size; the effective global
            # batch is per_device_batch_size * total_devices. The "MBS"/"GBS"
            # columns are reused here so logs/summary stay aligned with the
            # other frameworks; MBS == per-device batch, GBS == global batch.
            MBS=$(yq_or_na ".modules.$MODULE.overrides.per_device_batch_size" "$config_file")
            SEQ_LEN=$(yq_or_na ".modules.$MODULE.overrides.max_target_length" "$config_file")
            STEPS=$(yq_or_na ".modules.$MODULE.overrides.steps" "$config_file")
            if [ "$MBS" != "N/A" ]; then
                # Float-safe multiply via awk; per_device_batch_size can be
                # fractional in MaxText (e.g. 0.5). Print as int when whole.
                GBS=$(awk -v m="$MBS" -v n="$NNODES" -v g="$GPUS_PER_NODE" 'BEGIN {
                    v = m * n * g
                    if (v == int(v)) printf "%d", v; else printf "%g", v
                }')
                [ -z "$GBS" ] && GBS="N/A"
            else
                GBS="N/A"
            fi
            ;;
        maxdiffusion)
            MBS=$(yq_or_na ".modules.$MODULE.overrides.per_device_batch_size" "$config_file")
            GBS=$(yq_or_na ".modules.$MODULE.overrides.global_batch_size" "$config_file")
            SEQ_LEN="N/A"
            STEPS=$(yq_or_na ".modules.$MODULE.overrides.max_train_steps" "$config_file")
            ;;
        megatron_bridge)
            # Same override names for pretrain and for the SFT/LoRA posttrain
            # recipes; only the module they live under differs.
            MBS=$(yq_or_na ".modules.$MODULE.overrides.micro_batch_size" "$config_file")
            GBS=$(yq_or_na ".modules.$MODULE.overrides.global_batch_size" "$config_file")
            SEQ_LEN=$(yq_or_na ".modules.$MODULE.overrides.seq_length" "$config_file")
            STEPS=$(yq_or_na ".modules.$MODULE.overrides.train_iters" "$config_file")
            ;;
        *)
            MBS=$(yq_or_na ".modules.$MODULE.overrides.micro_batch_size" "$config_file")
            GBS=$(yq_or_na ".modules.$MODULE.overrides.global_batch_size" "$config_file")
            SEQ_LEN="N/A"
            STEPS="N/A"
            ;;
    esac
}

# Primus CLI override flag that sets training length for a given framework.
# These names come from trainer tests / user-guide examples, not from the
# backend's native argparse (e.g. TorchTitan is --training.steps, not --steps).
train_step_override_flag() {
    local framework="$1"
    case "$framework" in
        megatron|megatron_bridge) echo "--train_iters" ;;
        torchtitan)               echo "--training.steps" ;;
        maxtext)                  echo "--steps" ;;
        maxdiffusion)             echo "--max_train_steps" ;;
        *)                        echo "" ;;
    esac
}

# Build the primus-cli argv for one config. Extra tokens after --config are
# deep-merged into the overrides of the module being trained (CLI > YAML).
build_launch_argv() {
    local config_file="$1"
    local framework="$2"
    local suite="$3"
    local flag

    if [ -z "$suite" ]; then
        echo "[ERROR] no modules.pre_trainer or modules.post_trainer found in: $config_file" >&2
        echo "        A Primus experiment YAML must declare one of them." >&2
        return 1
    fi

    # $PRIMUS_MODE is intentionally unquoted so a value like "container" or
    # "direct" (or a short extra mode flag string) still splits as intended.
    # EXTRA_ENV becomes explicit `--env KEY=VALUE` flags before the `--`, which
    # is the only way to be sure a variable reaches the container.
    # shellcheck disable=SC2206
    LAUNCH_ARGV=( "$PRIMUS_CLI" $PRIMUS_MODE ${PERF_EXTRA_ENV_ARGS[@]+"${PERF_EXTRA_ENV_ARGS[@]}"} -- train "$suite" --config "$config_file" )

    if [ -z "$TRAIN_STEPS" ]; then
        return 0
    fi

    flag=$(train_step_override_flag "$framework")
    if [ -z "$flag" ]; then
        echo "[ERROR] TRAIN_STEPS=$TRAIN_STEPS but no step-override flag for framework '$framework' (config: $config_file)" >&2
        echo "        Known: megatron, megatron_bridge, torchtitan, maxtext, maxdiffusion." >&2
        return 1
    fi
    LAUNCH_ARGV+=( "$flag" "$TRAIN_STEPS" )
}

# Collect host / git / GPU info that helps reviewers understand the run.
collect_system_info() {
    {
        echo "################################################################################"
        echo "# System / Environment Snapshot"
        echo "################################################################################"
        echo "# Date           : $(date '+%Y-%m-%d %H:%M:%S %Z')"
        echo "# Hostname       : $(hostname)"
        echo "# Kernel         : $(uname -srm)"
        echo "# OS             : $(grep ^PRETTY_NAME /etc/os-release 2>/dev/null | cut -d= -f2 | tr -d '\"' || echo 'unknown')"
        echo "# User           : $(whoami)"
        echo "# CWD            : $(pwd)"
        echo "# PRIMUS_PATH    : $PRIMUS_PATH"
        echo "# PRIMUS_MODE    : $PRIMUS_MODE"
        echo "# Docker image   : ${DOCKER_IMAGE:-<unset; container mode uses runner/.primus.yaml>}"
        echo "# Script         : $SCRIPT_PATH"
        echo "# Bash           : $BASH_VERSION"
        echo "# Python         : $(command -v python3 >/dev/null && python3 --version 2>&1 || echo 'not found')"
        echo "# yq             : $(yq --version 2>&1 | head -1)"
        if command -v git >/dev/null 2>&1 && git -C "$PRIMUS_PATH" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
            echo "# Git branch     : $(git -C "$PRIMUS_PATH" rev-parse --abbrev-ref HEAD 2>/dev/null)"
            echo "# Git commit    : $(git -C "$PRIMUS_PATH" rev-parse HEAD 2>/dev/null)"
            local dirty
            dirty=$(git -C "$PRIMUS_PATH" status --porcelain 2>/dev/null | wc -l)
            echo "# Git dirty?    : $([ "$dirty" -eq 0 ] && echo no || echo "yes ($dirty files modified)")"
        else
            echo "# Git           : not a git checkout"
        fi
        if command -v rocminfo >/dev/null 2>&1; then
            local rocm_ver
            rocm_ver=$(rocminfo 2>/dev/null | awk '/ROCm Version/ {print $3; exit}')
            echo "# ROCm version  : ${rocm_ver:-unknown}"
        fi
        if command -v rocm-smi >/dev/null 2>&1; then
            local gpu_count
            gpu_count=$(rocm-smi --showid 2>/dev/null | grep -c '^GPU\[' || echo "?")
            echo "# ROCm GPUs     : $gpu_count"
        fi
        echo "################################################################################"
    }
}

print_per_run_banner() {
    local result_file="$1"
    local config_file="$2"
    local config_name="$3"
    local config_hash="$4"
    local framework="$5"
    local model="$6"
    local mbs="$7"
    local gbs="$8"
    local seq_len="$9"
    local steps="${10}"
    local rep="${11}"
    local total_reps="${12}"
    local launcher_cmd="${13}"
    local module="${14}"
    local suite="${15}"

    {
        echo "################################################################################"
        echo "# Primus Benchmark Run"
        echo "################################################################################"
        echo "# Timestamp        : $(date '+%Y-%m-%d %H:%M:%S %Z')"
        echo "# Hostname         : $(hostname)"
        echo "# Result Dir       : $RESULT_DIR"
        echo "# Result File      : $result_file"
        echo "# Batch Stamp      : $BATCH_TS"
        echo "# --------------------------------------------------------------------------------"
        echo "# Config File      : $config_file"
        echo "# Config Name      : $config_name"
        echo "# Config Hash      : $config_hash  (sha256, first 8 hex chars)"
        echo "# Framework        : $framework"
        echo "# Train Module     : $module  (CLI verb: train $suite)"
        echo "# Model            : $model"
        echo "# Micro Batch Size : $mbs"
        echo "# Global Batch Size: $gbs"
        echo "# Sequence Length  : $seq_len"
        echo "# Train Steps/Iters: $steps"
        echo "# Repetition       : $rep / $total_reps"
        echo "# --------------------------------------------------------------------------------"
        echo "# Cluster          : NNODES=$NNODES GPUS_PER_NODE=$GPUS_PER_NODE NODE_RANK=$NODE_RANK"
        echo "# Master           : $MASTER_ADDR:$MASTER_PORT"
        echo "# PRIMUS_MODE      : $PRIMUS_MODE"
        echo "# Launcher command : $launcher_cmd"
        echo "# --------------------------------------------------------------------------------"
        echo "# HF_TOKEN         : $([ -n "$HF_TOKEN" ] && echo '<set>' || echo '<unset>')"
        perf_print_provenance_banner "${DOCKER_IMAGE:-}"
        echo "################################################################################"
        echo ""
        echo "########################## Begin Config File Dump ##############################"
        cat "$config_file"
        # Some YAML files don't end with a newline; force one so the End marker
        # below lands on its own line instead of being appended to the last
        # config line.
        [ -n "$(tail -c1 "$config_file")" ] && echo
        echo "########################### End Config File Dump ###############################"
        echo ""
        echo "############################## Begin Run Output ################################"
    } > "$result_file"
}

print_per_run_footer() {
    local result_file="$1"
    local exit_code="$2"
    local run_elapsed="$3"
    {
        echo "############################### End Run Output #################################"
        echo ""
        echo "################################################################################"
        echo "# Run finished at : $(date '+%Y-%m-%d %H:%M:%S %Z')"
        echo "# Exit code       : $exit_code"
        echo "# Elapsed (sec)   : $run_elapsed"
        echo "################################################################################"
    } >> "$result_file"
}

# --------------------------- Interrupt handling ------------------------------
# Ctrl+C should stop the run that is in flight, tear down its container, and
# abandon the rest of the batch rather than rolling on to the next config.
#
# The terminal already delivers SIGINT to the foreground process group, so the
# launcher itself goes down on its own. What does not go down is the training
# container: `docker run` losing its client does not stop the container, so we
# have to stop it explicitly.

# primus-cli names containers "<base>-<SLURM_JOB_ID or pid>" (the base comes
# from runner/.primus.yaml, the suffix is appended by
# runner/primus-cli-container.sh), so there is no fixed name to stop -- match
# on the prefix instead.
PRIMUS_CONTAINER_PREFIX="${PRIMUS_CONTAINER_PREFIX:-primus-training}"

# Containers matching that prefix that were already up before this batch
# started. Nodes get shared, and a previous or concurrent job's container must
# survive our Ctrl+C, so only containers absent from this list are ours to stop.
PREEXISTING_CONTAINERS=""

CURRENT_RESULT_FILE=""
CURRENT_RUN_START=0

primus_container_ids() {
    command -v docker >/dev/null 2>&1 || return 0
    docker ps -q --filter "name=^/${PRIMUS_CONTAINER_PREFIX}" 2>/dev/null
}

snapshot_preexisting_containers() {
    PREEXISTING_CONTAINERS="$(primus_container_ids)"
    if [ -n "$PREEXISTING_CONTAINERS" ]; then
        echo "[INFO] pre-existing ${PRIMUS_CONTAINER_PREFIX}* containers will be left alone on interrupt:" \
            | tee -a "$SUMMARY_FILE"
        local id
        for id in $PREEXISTING_CONTAINERS; do
            echo "         $id  $(docker inspect --format '{{.Name}}' "$id" 2>/dev/null)" \
                | tee -a "$SUMMARY_FILE"
        done
    fi
}

# Invoked from on_interrupt, which itself runs via trap, so shellcheck sees no
# call site and reports the whole body as unreachable.
# shellcheck disable=SC2317,SC2329
stop_our_containers() {
    local id name
    for id in $(primus_container_ids); do
        case " $PREEXISTING_CONTAINERS " in
            *" $id "*) continue ;;
        esac
        name=$(docker inspect --format '{{.Name}}' "$id" 2>/dev/null || echo "$id")
        echo ">>> Stopping container ${name#/} ..." >&2
        docker stop -t 20 "$id" >/dev/null 2>&1 || true
    done
}

# shellcheck disable=SC2317,SC2329  # invoked indirectly via trap
on_interrupt() {
    # A second Ctrl+C should not re-enter this handler mid-cleanup.
    trap '' INT TERM

    echo "" >&2
    echo ">>> Interrupted. Stopping the current run; the rest of the batch is abandoned." >&2

    # The launcher has already been signalled by the terminal; give it a moment
    # to unwind before we force the container down.
    sleep 3
    stop_our_containers

    if [ -n "$CURRENT_RESULT_FILE" ]; then
        print_per_run_footer "$CURRENT_RESULT_FILE" 130 "$(( $(date +%s) - CURRENT_RUN_START ))"
        echo ">>> Interrupted run log: $CURRENT_RESULT_FILE" >&2
    fi

    {
        echo ""
        echo "################################################################################"
        echo "# Primus Benchmark Batch INTERRUPTED"
        echo "################################################################################"
        echo "# Interrupted at: $(date '+%Y-%m-%d %H:%M:%S %Z')"
        # Defaulted because an interrupt can land before the run loop starts,
        # e.g. during the container probe, and `set -u` is on.
        echo "# Runs started  : ${TOTAL_RUNS:-0}"
        echo "# Runs failed   : ${FAILED_RUNS:-0}"
        echo "# In-flight run : ${CURRENT_RESULT_FILE:-<none>}"
        echo "################################################################################"
    } | tee -a "$SUMMARY_FILE"

    exit 130
}
trap on_interrupt INT TERM

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
    echo "# Start time   : $BATCH_START"
    echo "# Batch stamp  : $BATCH_TS"
    echo "# RESULT_DIR   : $RESULT_DIR"
    echo "# Config source: $CONFIG_SOURCE"
    echo "# PRIMUS_MODE  : $PRIMUS_MODE"
    echo "# Docker image : ${DOCKER_IMAGE:-<unset; container mode uses runner/.primus.yaml>}"
    echo "# Image digest : $(perf_image_digest "${DOCKER_IMAGE:-}")"
    echo "# Reps/config  : $NUM_REPS"
    echo "# TRAIN_STEPS  : ${TRAIN_STEPS:-<unset; use YAML>}"
    # CONFIG_FILES is populated by perf_select_configs in lib/common.sh.
    # shellcheck disable=SC2153
    echo "# Total configs: ${#CONFIG_FILES[@]}"
    echo "# Total runs   : $(( ${#CONFIG_FILES[@]} * NUM_REPS ))"
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

# Record which training containers are already up, so an interrupt only stops
# the ones this batch starts.
snapshot_preexisting_containers

cd "$PRIMUS_PATH" || { echo "[ERROR] cannot cd into $PRIMUS_PATH" >&2; exit 1; }

TOTAL_RUNS=0
FAILED_RUNS=0

for config_file in "${CONFIG_FILES[@]}"; do
    # `config_name` keeps the path relative to the repo root, so the banner and
    # summary show where the config came from (examples/megatron/configs/... )
    # rather than a bare basename that is ambiguous across framework folders.
    # The extractor tolerates a path here: `_split_model_precision` takes the
    # last segment before parsing the model name.
    config_rel="${config_file#"$PRIMUS_PATH"/}"
    config_name="${config_rel%.yaml}"
    config_name="${config_name%.yml}"

    # For the on-disk log filename we use only the basename (no subfolder),
    # since the `${fw_safe}-` prefix already disambiguates same-named YAMLs
    # that live under different framework folders. Including the subfolder
    # here produced redundant names like `megatron-megatron_<config>-...log`.
    config_basename=$(basename "$config_file")
    config_basename="${config_basename%.yaml}"
    config_basename="${config_basename%.yml}"

    extract_config_fields "$config_file"
    YAML_STEPS="$STEPS"
    if [ -n "$TRAIN_STEPS" ]; then
        STEPS="$TRAIN_STEPS (CLI override; yaml=$YAML_STEPS)"
    fi

    if ! build_launch_argv "$config_file" "$FRAMEWORK" "$TRAIN_SUITE"; then
        TOTAL_RUNS=$((TOTAL_RUNS + NUM_REPS))
        FAILED_RUNS=$((FAILED_RUNS + NUM_REPS))
        echo "[SKIP/FAIL] ${FRAMEWORK}/${config_name}: cannot build launch command" \
            | tee -a "$SUMMARY_FILE"
        continue
    fi
    launcher_cmd="${LAUNCH_ARGV[*]}"

    fw_safe=$(sanitize "$FRAMEWORK")
    name_safe=$(sanitize "$config_basename")
    mbs_safe=$(sanitize "$MBS")
    gbs_safe=$(sanitize "$GBS")
    config_hash=$(config_fingerprint "$config_file")

    for rep in $(seq 1 "$NUM_REPS"); do
        TOTAL_RUNS=$((TOTAL_RUNS + 1))
        result_file="${RESULT_DIR}/${fw_safe}-${name_safe}-${config_hash}-MBS${mbs_safe}-GBS${gbs_safe}-rep${rep}_${BATCH_TS}.log"

        echo "[$(date '+%H:%M:%S')] [run ${TOTAL_RUNS}/$(( ${#CONFIG_FILES[@]} * NUM_REPS ))] ${FRAMEWORK}/${TRAIN_SUITE}/${config_name} [${config_hash}] MBS=${MBS} GBS=${GBS} steps=${STEPS} rep ${rep}/${NUM_REPS}" \
            | tee -a "$SUMMARY_FILE"
        echo "                    -> $result_file" | tee -a "$SUMMARY_FILE"

        print_per_run_banner "$result_file" "$config_file" "$config_name" "$config_hash" \
            "$FRAMEWORK" "$MODEL" "$MBS" "$GBS" "$SEQ_LEN" "$STEPS" "$rep" "$NUM_REPS" \
            "$launcher_cmd" "$MODULE" "$TRAIN_SUITE"

        run_start_ts=$(date +%s)

        # Published for on_interrupt, so a Ctrl+C can close out the log of
        # whichever run was in flight.
        CURRENT_RESULT_FILE="$result_file"
        CURRENT_RUN_START="$run_start_ts"

        # The actual benchmark invocation. Stdout+stderr are appended to the
        # per-run log so we capture everything Primus prints -- with secret
        # values masked on the way, because primus-cli logs the container
        # command it runs, `--env HF_TOKEN=<value>` included.
        # PIPESTATUS[0] is still the launcher's exit code.
        "${LAUNCH_ARGV[@]}" 2>&1 | perf_redact_stream | tee -a "$result_file"
        exit_code=${PIPESTATUS[0]}

        run_end_ts=$(date +%s)
        run_elapsed=$((run_end_ts - run_start_ts))

        CURRENT_RESULT_FILE=""

        print_per_run_footer "$result_file" "$exit_code" "$run_elapsed"

        if [ "$exit_code" -ne 0 ]; then
            FAILED_RUNS=$((FAILED_RUNS + 1))
            echo "                    [FAIL] exit=$exit_code elapsed=${run_elapsed}s" \
                | tee -a "$SUMMARY_FILE"
        else
            echo "                    [ OK ] exit=$exit_code elapsed=${run_elapsed}s" \
                | tee -a "$SUMMARY_FILE"
        fi
    done
done

BATCH_END=$(date '+%Y-%m-%d %H:%M:%S %Z')
BATCH_END_TS=$(date +%s)
BATCH_ELAPSED=$((BATCH_END_TS - BATCH_START_TS))

{
    echo ""
    echo "################################################################################"
    echo "# Primus Benchmark Batch Complete"
    echo "################################################################################"
    echo "# End time     : $BATCH_END"
    echo "# Elapsed (sec): $BATCH_ELAPSED"
    echo "# Total runs   : $TOTAL_RUNS"
    echo "# Failed runs  : $FAILED_RUNS"
    echo "################################################################################"
} | tee -a "$SUMMARY_FILE"

# The batch driver was copied before the first run, so an interrupted batch
# still has it; just report the artefacts here.
echo "Saved batch driver -> $SCRIPT_COPY" | tee -a "$SUMMARY_FILE"
echo "Saved env snapshot -> $ENV_FILE"    | tee -a "$SUMMARY_FILE"
echo "Saved provenance   -> $RESULT_DIR/batch_{submodules,gpu,stack}_${BATCH_TS}.txt" | tee -a "$SUMMARY_FILE"
echo "Saved summary      -> $SUMMARY_FILE"

if [ "$FAILED_RUNS" -eq 0 ]; then
    exit 0
else
    exit 1
fi
