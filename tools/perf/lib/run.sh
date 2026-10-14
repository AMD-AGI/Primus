#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Per-run helpers shared by run_batch.sh and run_batch_multinode.sh: input
# validation, reading a config's key fields, the TRAIN_STEPS override, and the
# system snapshot, per-run banner and footer.
#
# Callers source lib/common.sh first, and set PERF_MULTINODE=1 to get the
# multi-node layout of the snapshot, banner and footer.
###############################################################################

# Input validation run before anything launches: EXTRA_ENV, the selected
# configs, the repo and CLI paths, and TRAIN_STEPS.
check_batch_inputs() {
    local cli_abs
    perf_parse_extra_env || return 1
    perf_select_configs || return 1

    if [ ! -d "$PRIMUS_PATH" ]; then
        echo "[ERROR] PRIMUS_PATH does not exist: $PRIMUS_PATH" >&2
        return 1
    fi

    # Resolve absolute path of the CLI (handles default './primus-cli').
    cli_abs="$PRIMUS_PATH/${PRIMUS_CLI#./}"
    if [ ! -x "$cli_abs" ]; then
        echo "[ERROR] primus-cli not found or not executable at: $cli_abs" >&2
        return 1
    fi

    if [ -n "$TRAIN_STEPS" ]; then
        if ! [[ "$TRAIN_STEPS" =~ ^[1-9][0-9]*$ ]]; then
            echo "[ERROR] TRAIN_STEPS must be a positive integer, got: '$TRAIN_STEPS'" >&2
            return 1
        fi
    fi
}

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
# Bridge SFT/LoRA recipes. Everything else keys off the answer, because the
# module fixes both the CLI verb and the overrides path.
#
# Two separate yq calls on purpose: `.modules | has("a") or has("b")` binds
# the second has() to the document root, so a post_trainer-only config would
# test false.
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
            if [ "$GBS" = "N/A" ] && [[ "$MBS" =~ ^[0-9]+$ ]]; then
                GBS=$(torchtitan_default_gbs "$config_file" "$MBS")
            fi
            ;;
        maxtext)
            # MaxText (JAX) uses per_device_batch_size; the effective global
            # batch is per_device_batch_size * total_devices. The "MBS"/"GBS"
            # columns are reused here so logs/summary stay aligned with the
            # other frameworks; MBS == per-device batch, GBS == global batch.
            # GPUS_PER_NODE isn't exported in the multi-node flow (primus-cli
            # sets it inside the container), so it falls back to 8 for this
            # display-only calculation.
            MBS=$(yq_or_na ".modules.$MODULE.overrides.per_device_batch_size" "$config_file")
            SEQ_LEN=$(yq_or_na ".modules.$MODULE.overrides.max_target_length" "$config_file")
            STEPS=$(yq_or_na ".modules.$MODULE.overrides.steps" "$config_file")
            if [ "$MBS" != "N/A" ]; then
                # Float-safe multiply via awk; per_device_batch_size can be
                # fractional in MaxText (e.g. 0.5). Print as int when whole.
                GBS=$(awk -v m="$MBS" -v n="$NNODES" -v g="${GPUS_PER_NODE:-8}" 'BEGIN {
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

# TorchTitan's global batch when the config leaves it unset: one local batch
# per data-parallel rank, i.e. MBS x GPUs divided by the tensor, pipeline and
# context parallel degrees (expert parallelism does not change it).
torchtitan_default_gbs() {
    local config_file="$1" mbs="$2" degree divisor=1 key
    for key in tensor_parallel_degree pipeline_parallel_degree context_parallel_degree; do
        degree=$(yq_or_na ".modules.$MODULE.overrides.parallelism.$key" "$config_file")
        if [[ "$degree" =~ ^[0-9]+$ ]] && [ "$degree" -gt 1 ]; then
            divisor=$((divisor * degree))
        fi
    done
    echo $(( mbs * NNODES * ${GPUS_PER_NODE:-8} / divisor ))
}

# Everything the loop needs to know about one config. Sets config_rel,
# config_name (path relative to the repo, shown in banners), the
# extract_config_fields globals, YAML_STEPS, STEPS (with the TRAIN_STEPS
# override noted), config_hash and the filename-safe *_safe parts.
prepare_config() {
    local config_file="$1" config_basename
    # The extractor tolerates a path in Config Name: `_split_model_precision`
    # takes the last segment before parsing the model name.
    config_rel="${config_file#"$PRIMUS_PATH"/}"
    config_name="${config_rel%.yaml}"
    config_name="${config_name%.yml}"

    # The log filename uses only the basename: the `${fw_safe}-` prefix already
    # tells same-named YAMLs from different framework folders apart.
    config_basename=$(basename "$config_file")
    config_basename="${config_basename%.yaml}"
    config_basename="${config_basename%.yml}"

    extract_config_fields "$config_file"
    YAML_STEPS="$STEPS"
    if [ -n "$TRAIN_STEPS" ]; then
        STEPS="$TRAIN_STEPS (CLI override; yaml=$YAML_STEPS)"
    fi

    fw_safe=$(sanitize "$FRAMEWORK")
    name_safe=$(sanitize "$config_basename")
    mbs_safe=$(sanitize "$MBS")
    gbs_safe=$(sanitize "$GBS")
    config_hash=$(config_fingerprint "$config_file")
}

# Log path for repetition <rep> of the config last passed to prepare_config.
# <suffix> marks a failed multi-node attempt, e.g. "-failed-attempt2".
run_log_name() {
    local rep="$1" suffix="${2:-}"
    echo "${RESULT_DIR}/${fw_safe}-${name_safe}-${config_hash}-MBS${mbs_safe}-GBS${gbs_safe}-rep${rep}${suffix}_${BATCH_TS}.log"
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

# Append the TRAIN_STEPS override to LAUNCH_ARGV with the flag the config's
# framework understands. Extra tokens after --config are deep-merged into the
# overrides of the module being trained (CLI > YAML). No-op when unset.
append_train_steps_override() {
    local config_file="$1"
    local framework="$2"
    local flag

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
        if [ "${PERF_MULTINODE:-0}" = 1 ]; then
            echo "# --------------------------------------------------------------------------------"
            echo "# SLURM job      : ${SLURM_JOB_ID:-<none>}"
            echo "# SLURM nodes    : ${SLURM_JOB_NODELIST:-<unknown>}"
            echo "# SLURM nnodes   : ${SLURM_NNODES:-<unknown>}"
            echo "# NNODES (script): $NNODES"
        fi
        # The container's ROCm is in the batch configuration below.
        echo "# ROCm (host)   : $(perf_rocm_version)"
        echo "# GPU model     : $(perf_gpu_model)"
        echo "# GPUs (local)  : $(perf_gpu_count)"
        echo "################################################################################"
    }
}

# Per-run banner, written to a fresh log before the run starts. The extractor
# reads every `# key : value` line in it.
#
# Arguments: result_file config_file config_name config_hash framework model
# mbs gbs seq_len steps rep total_reps launcher_cmd module suite, then attempt
# and max_attempts for the multi-node retry loop.
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
    local attempt="${16:-}"
    local max_attempts="${17:-}"
    local multinode="${PERF_MULTINODE:-0}"

    {
        echo "################################################################################"
        if [ "$multinode" = 1 ]; then
            echo "# Primus Multi-Node Benchmark Run"
        else
            echo "# Primus Benchmark Run"
        fi
        echo "################################################################################"
        echo "# Timestamp        : $(date '+%Y-%m-%d %H:%M:%S %Z')"
        echo "# Hostname         : $(hostname)"
        if [ "$multinode" = 1 ]; then
            # Split across two keys so both reach the extractor's CSV; a slow run
            # can then be pinned on a specific node.
            echo "# SLURM job        : ${SLURM_JOB_ID:-<none>}"
            echo "# SLURM nodes      : ${SLURM_JOB_NODELIST:-<unknown>}"
        fi
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
        if [ "$multinode" = 1 ]; then
            echo "# Repetition       : $rep / $total_reps  (successful runs target)"
            echo "# Attempt          : $attempt / $max_attempts (incl. failed)"
        else
            echo "# Repetition       : $rep / $total_reps"
        fi
        echo "# --------------------------------------------------------------------------------"
        if [ "$multinode" = 1 ]; then
            echo "# Cluster          : NNODES=$NNODES GPUS_PER_NODE=${GPUS_PER_NODE:-8}  (rank/master set by primus-cli)"
        else
            echo "# Cluster          : NNODES=$NNODES GPUS_PER_NODE=$GPUS_PER_NODE NODE_RANK=$NODE_RANK"
            echo "# Master           : $MASTER_ADDR:$MASTER_PORT"
        fi
        echo "# PRIMUS_MODE      : $PRIMUS_MODE"
        echo "# Launcher command : $launcher_cmd"
        echo "# Run limits       : $(perf_run_limits_text)"
        if [ -n "$FAIL_PATTERN" ]; then
            # Not printed verbatim: the watchdog greps the run output for it,
            # and keeping it out of the log avoids any chance of a self-match.
            echo "# Fail pattern     : <recorded in batch_summary.txt>"
        fi
        if perf_uses_containers; then
            echo "# Container label  : ${PERF_LABEL_KEY}=${PERF_LABEL_VALUE}"
        fi
        echo "# GPU check        : $PERF_GPU_CHECK"
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

# Footer appended once the run has ended (or been interrupted). The extractor
# reads `Exit code` and `Elapsed (sec)` from the last lines of the log;
# `Failure reason` is "none" or "<category>: <detail>" (see
# perf_launch_with_watchdog).
print_per_run_footer() {
    local result_file="$1"
    local exit_code="$2"
    local run_elapsed="$3"
    local failure_reason="$4"
    local rule="################################################################################"
    local footer
    # Built first and written with one printf: a signal can then not land
    # between its lines, nor run the interrupt handler with stdout still
    # pointing into the log.
    printf -v footer '\n%s\n\n%s\n# Run finished at  : %s\n# Exit code        : %s\n# Elapsed (sec)    : %s\n# Failure reason   : %s\n%s\n' \
        "############################### End Run Output #################################" "$rule" \
        "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$exit_code" "$run_elapsed" "$failure_reason" "$rule"
    printf '%s' "$footer" >> "$result_file"
}
