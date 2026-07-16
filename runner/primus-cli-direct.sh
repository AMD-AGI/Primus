#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

set -euo pipefail

print_usage() {
cat << EOF
Primus Direct Launcher

Usage:
    primus-cli direct [--env KEY=VALUE ...] [--single] [--script <file.py>] [--patch <file>] [-- primus-args]

Description:
    Launch Primus training, benchmarking, or preflight directly on the host (or inside a container).
    Distributed settings can be controlled by either exporting environment variables in advance,
    or by specifying them inline using --env KEY=VALUE.

    This script automatically detects your GPU model (e.g., MI300X, MI250X) and loads GPU-specific
    environment variable optimizations from runner/helpers/envs/<GPU_MODEL>.sh for best performance.

Options:
    --config <file>      Specify configuration file (default: .primus.yaml or system default)
    --debug              Enable debug mode with verbose logging
    --dry-run            Show configuration and command without executing
    --single             Run with python3 instead of torchrun (single process only)
    --script <file.py>   Python script to execute (default: primus/cli/main.py)
    --env KEY=VALUE      Set environment variable before execution
    --patch <file>       Apply a patch script before execution (can specify multiple times)
    --log_file PATH      Save log to a specific file (default: logs/log_TIMESTAMP.txt)
    --numa               Force enable NUMA binding for processes
    --no-numa            Force disable NUMA binding for processes
    --silent             Suppress all launcher and python tool stdout (back-pocket option).
                         Must be placed BEFORE the '--' separator. Launcher errors
                         (LOG_ERROR / LOG_WARN, written to stderr) and the log file
                         are preserved. Exit code is propagated. NOT recommended for
                         normal use -- you lose live progress; prefer the log file.

Distributed Environment Variables:
    NNODES        Number of nodes participating in distributed run        [default: 1]
    NODE_RANK     Rank of the current node (unique integer per node)      [default: 0]
    GPUS_PER_NODE Number of GPUs to use per node                          [default: 8]
    MASTER_ADDR   Hostname or IP of master node                           [default: localhost]
    MASTER_PORT   Port of master node                                     [default: 1234]

    When running inside a SLURM allocation (SLURM_JOB_ID set), the above are
    auto-derived from SLURM_NNODES / SLURM_NODEID / SLURM_NODELIST when not
    pre-exported. Pre-exported values always win, so the standard
    slurm-entry -> container -> direct chain is unaffected.

Optional Environment Variables:
    VENV_ACTIVATE Path to a Python virtualenv activate script. If set, sourced
                  before the primus run. Unset = no-op (use system python or
                  the container's bundled python). Set + missing file =
                  fail-fast (avoid silent torch-version mismatches).

You can set these variables in either of the following ways:
    # (1) Export variables before launch (recommended for scripts or single-node runs)
      export NNODES=2 GPUS_PER_NODE=8 NODE_RANK=0 MASTER_ADDR=host1
      primus-cli direct -- train pretrain --config exp.yaml

    # (2) Inject via CLI with --env (useful for launchers and multi-node jobs)
      primus-cli direct --env NNODES=2 --env GPUS_PER_NODE=8 --env NODE_RANK=1 --env MASTER_ADDR=host1 -- \\
        benchmark gemm -M 4096 -N 4096 -K 4096

    # (3) Let SLURM provide them (inside an existing allocation)
      export VENV_ACTIVATE=~/envs/preflight/.venv/bin/activate
      srun -N 4 --ntasks-per-node=1 ./runner/primus-cli direct -- preflight --quick

Examples:
    # Pretrain with a config file (single node)
      primus-cli direct -- train pretrain --config examples/megatron/exp_pretrain.yaml

    # Benchmark GEMM (single node)
      primus-cli direct -- benchmark gemm

    # Distributed GEMM benchmark, 2 nodes, 8 GPUs per node (rank 0)
      primus-cli direct --env NNODES=2 --env GPUS_PER_NODE=8 --env NODE_RANK=0 --env MASTER_ADDR=host1 -- \\
        benchmark gemm -M 4096 -N 4096 -K 4096

    # Launch as rank 1 (2-node distributed)
      primus-cli direct --env NNODES=2 --env GPUS_PER_NODE=8 --env NODE_RANK=1 --env MASTER_ADDR=host1 -- \\
        benchmark gemm -M 4096 -N 4096 -K 4096

    # Run a custom script directly (no torchrun)
      primus-cli direct --single --script examples/debug_run.py -- --arg1 val1

    # Run a custom script with torchrun
      primus-cli direct --script examples/run_distributed.py -- --config conf.yaml

    # Apply patch scripts before execution
      primus-cli direct --patch patches/fix_env.sh --patch patches/setup.py -- train pretrain --config exp.yaml

    # Force enable NUMA binding for better performance
      primus-cli direct --numa -- benchmark gemm -M 8192 -N 8192 -K 8192

    # Run silently (back-pocket option; launcher errors and log file preserved)
      primus-cli direct --silent -- preflight --quick

Notes:
    - If --single is specified, Primus skips torchrun and uses python3 directly.
    - run_mode auto-detection: when the primus subcommand is 'node_smoke', run_mode
      defaults to 'single' (node_smoke runs one process per node by design).
      Explicit --single or config direct.run_mode still wins.
    - If --script is not specified, defaults to primus/cli/main.py.
    - Always separate Primus arguments from launcher options using '--'.
    - Environment variables can be mixed: 'export' takes precedence unless overridden by '--env'.
    - Multi-node jobs require MASTER_ADDR set to the master node's hostname/IP.
      Inside a SLURM allocation it is auto-resolved from SLURM_NODELIST.
    - Patch scripts are executed in order before running the main script (useful for env setup, hot fixes, etc.).
    - NUMA binding is auto-disabled by default; use --numa to enable for better memory locality.
    - GPU-specific optimizations: The script automatically sources primus-env.sh, which detects your
      GPU model and loads optimized environment variables from runner/helpers/envs/<GPU_MODEL>.sh
      (e.g., MI300X.sh, MI250X.sh). If no GPU-specific config is found, it uses base_env.sh only.

EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    print_usage
    exit 0
fi

###############################################################################
# Runner-level --silent (back-pocket option for clean stdout)
#
# Contract:
#   - launcher LOG_INFO / LOG_INFO_RANK0 / LOG_DEBUG_* (stdout)  -> /dev/null
#   - launcher LOG_WARN / LOG_ERROR  (stderr, written via >&2)   -> terminal
#   - python tool's stdout + stderr                              -> /dev/null
#       (the launched CMD ends with `2>&1 | tee <log_file>`, so the python
#        child's fd2 is merged into the pipe and tee writes to the log file
#        + this script's fd1 = /dev/null)
#   - log file                                                   -> captures all
#   - exit code                                                  -> propagated
#
# --silent is only honored BEFORE the `--` separator. Anywhere after `--` it
# is forwarded to the python tool whose argparse will reject it (intentional;
# python tools have no --silent flag and never will -- silencing is a bash-side
# concern).
#
# Not recommended for normal use: read the log file or omit --silent if you
# want to see live tool output and progress.
###############################################################################
SILENT=0
for _arg in "$@"; do
    if [[ "$_arg" == "--" ]]; then
        break
    fi
    if [[ "$_arg" == "--silent" ]]; then
        SILENT=1
        break
    fi
done
if [[ "$SILENT" == "1" ]]; then
    # fd1 -> /dev/null. fd2 is left attached to the terminal, so LOG_ERROR /
    # LOG_WARN (which write via >&2) still reach the operator.
    exec >/dev/null
fi

# Resolve runner directory
# Use RUNNER_DIR instead of SCRIPT_DIR to avoid conflicts with sourced scripts
RUNNER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load common library (required)
# shellcheck disable=SC1091
source "$RUNNER_DIR/lib/common.sh" || {
    echo "[ERROR] Failed to load common library: $RUNNER_DIR/lib/common.sh" >&2
    exit 1
}

# Load config library (required)
# shellcheck disable=SC1091
source "$RUNNER_DIR/lib/config.sh" || {
    LOG_ERROR "[direct] Failed to load config library: $RUNNER_DIR/lib/config.sh"
    exit 1
}

LOG_INFO_RANK0 "-----------------------------------------------"
LOG_INFO_RANK0 "primus-cli-direct.sh"
LOG_INFO_RANK0 "-----------------------------------------------"



###############################################################################
# STEP 1: Pre-parse global options (--config, --debug, --dry-run, --help)
###############################################################################
CONFIG_FILE=""
DEBUG_MODE=0
DRY_RUN_MODE=0
PRE_PARSE_ARGS=()
PRIMUS_ARGS=()

# If the first argument is the subcommand "direct", skip it
if [[ "${1:-}" == "direct" ]]; then
    shift
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --debug)
            DEBUG_MODE=1
            shift
            ;;
        --dry-run)
            DRY_RUN_MODE=1
            shift
            ;;
        --help|-h)
            print_usage
            exit 0
            ;;
        --env)
            if [[ "${2:-}" == *=* ]]; then
                # Inline KEY=VALUE form: export immediately and keep for later so that
                # direct.env / --env can be re-applied with highest priority.
                _env_val="${2:-}"
                export "${_env_val%%=*}"="${_env_val#*=}"
                PRE_PARSE_ARGS+=("$1" "${2:-}")
                shift 2
            else
                # Non KEY=VALUE form: treat as env file path and defer sourcing until
                # after hooks and patches (STEP 9). Use a synthetic --env_file option
                # so that it is tracked via direct_config[env_file].
                PRE_PARSE_ARGS+=("--env_file" "${2:-}")
                LOG_INFO_RANK0 "[direct] CLI: --env_file ${2:-}"
                shift 2
            fi
            ;;
        --script|--log_file|--patch)
            # Runner options that take a value
            PRE_PARSE_ARGS+=("$1" "${2:-}")
            shift 2
            ;;
        --numa|--no-numa|--single)
            # Runner boolean flags (no value)
            PRE_PARSE_ARGS+=("$1")
            shift
            ;;
        --silent)
            # Already consumed by the pre-scan above (which has applied
            # `exec >/dev/null`). Swallow it here so it never reaches the
            # python parser, which has no --silent flag.
            shift
            ;;
        --)
            # Explicit separator: remaining args are for primus Python module
            shift  # skip the '--'
            PRIMUS_ARGS+=("$@")
            break
            ;;
        *)
            # First unrecognized argument (typically a subcommand like 'train'):
            # Stop parsing runner options, pass everything from here to primus Python module.
            # This prevents runner from intercepting options meant for Python (e.g., --config).
            PRIMUS_ARGS+=("$@")
            break
            ;;
    esac
done
# Restore arguments for second pass. Use `set --` so that options parsing stops
# and all following tokens (including those starting with '-') become positional
# parameters for the next parsing stage.
set -- "${PRE_PARSE_ARGS[@]}" -- "${PRIMUS_ARGS[@]}"
LOG_INFO_RANK0 "[direct] PRE_PARSE_ARGS (runner options): ${PRE_PARSE_ARGS[*]}"
LOG_INFO_RANK0 "[direct] PRIMUS_ARGS (python module args): ${PRIMUS_ARGS[*]}"
LOG_INFO_RANK0 "[direct] Combined args for second pass: $*"

# Enable debug mode early if set via CLI
if [[ "$DEBUG_MODE" == "1" ]]; then
    export PRIMUS_LOG_LEVEL="DEBUG"
    LOG_INFO_RANK0 "[direct] Debug mode enabled via CLI (PRIMUS_LOG_LEVEL=DEBUG)"
fi

###############################################################################
# STEP 2: Load configuration files
###############################################################################
load_config_auto "$CONFIG_FILE" "direct" || {
    LOG_ERROR "[direct] Configuration loading failed"
    exit 1
}

###############################################################################
# STEP 3: Extract direct.* config and apply defaults
###############################################################################
declare -A direct_config
extract_config_section "direct" direct_config || true

# Check debug/dry-run from config (CLI takes precedence, already set in STEP 1)
if [[ "$DEBUG_MODE" == "0" ]]; then
    debug_value="${direct_config[debug]:-false}"
    if [[ "$debug_value" == "true" || "$debug_value" == "1" ]]; then
        DEBUG_MODE=1
        export PRIMUS_LOG_LEVEL="DEBUG"
        LOG_INFO_RANK0 "[direct] Debug mode enabled via config (PRIMUS_LOG_LEVEL=DEBUG)"
    fi
fi

if [[ "$DRY_RUN_MODE" == "0" ]]; then
    dry_run_value="${direct_config[dry_run]:-false}"
    if [[ "$dry_run_value" == "true" || "$dry_run_value" == "1" ]]; then
        DRY_RUN_MODE=1
        LOG_INFO_RANK0 "[direct] Dry-run mode enabled via config"
    fi
fi

# Set default values for parameters not in config.
# NOTE: run_mode default is deliberately NOT applied here. The auto-detect step
# after STEP 4 needs to distinguish "user did not specify run_mode" (so we can
# auto-select `single` for node_smoke) from "default got applied in STEP 3".
# Defaulting to `torchrun` therefore happens after STEP 4 instead.
direct_config[script]="${direct_config[script]:-primus/cli/main.py}"
direct_config[numa]="${direct_config[numa]:-auto}"
direct_config[log_file]="${direct_config[log_file]:-}"

# Clean up empty array markers ("[]")
for key in "${!direct_config[@]}"; do
    if [[ "${direct_config[$key]}" == "[]" ]]; then
        unset "direct_config[$key]"
    fi
done

LOG_DEBUG_RANK0 "[direct] Configuration loaded, ready for CLI argument override"

###############################################################################
# STEP 4: Parse CLI arguments (all stored as newline-separated strings)
# Priority: CLI args > Config file > Defaults
###############################################################################
primus_args=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --)
            shift
            primus_args+=("$@")
            break
            ;;
        --single)
            # Special case: --single maps to run_mode=single
            direct_config[run_mode]="single"
            LOG_DEBUG_RANK0 "[direct] CLI: run_mode=single"
            shift
            ;;
        --no-numa)
            # Special case: --no-numa maps to numa=false
            direct_config[numa]="false"
            LOG_DEBUG_RANK0 "[direct] CLI: numa=false"
            shift
            ;;
        --*)
            # Generic option handler
            opt_name="${1#--}"
            opt_value="${2:-}"

            # Check if next argument is a value or another flag
            if [[ -z "$opt_value" ]] || [[ "$opt_value" == --* ]]; then
                # Boolean flag (no value or next is a flag)
                direct_config[$opt_name]="true"
                LOG_INFO_RANK0 "[direct] CLI: $opt_name=true"
                shift
            else
                # Key-value option: append with newline
                if [[ -z "${direct_config[$opt_name]:-}" ]]; then
                    direct_config[$opt_name]="$opt_value"
                else
                    direct_config[$opt_name]+=$'\n'"$opt_value"
                fi
                LOG_INFO_RANK0 "[direct] CLI: $opt_name += $opt_value"
                shift 2
            fi
            ;;
        *)
            primus_args+=("$1")
            shift
            ;;
    esac
done
set -- "${primus_args[@]}"

# Prepend --debug to Python CLI args when debug mode is enabled (CLI or config).
# Done here so it flows through hooks/patches and matches container/slurm pattern.
[[ "$DEBUG_MODE" == "1" ]] && set -- --debug "$@"

###############################################################################
# STEP 4.4: Auto-select run_mode based on primus subcommand
#
# Some primus subcommands MUST run as a single process per srun task (no
# torchrun fan-out, no inter-node rendezvous). node_smoke is the canonical
# example: it runs one process per node by design and the per-GPU phase is
# launched as subprocesses internally. Listed here so users don't need to
# remember --single. Explicit --single / config direct.run_mode still wins
# (those paths set direct_config[run_mode] before this block runs).
###############################################################################
SINGLE_MODE_SUBCOMMANDS=(node_smoke)
if [[ -z "${direct_config[run_mode]:-}" ]]; then
    _detected_subcmd=""
    for _arg in "${primus_args[@]}"; do
        case "$_arg" in
            --*|-*) continue ;;
            *) _detected_subcmd="$_arg"; break ;;
        esac
    done
    _default_run_mode="torchrun"
    for _sc in "${SINGLE_MODE_SUBCOMMANDS[@]}"; do
        if [[ "$_detected_subcmd" == "$_sc" ]]; then
            _default_run_mode="single"
            LOG_INFO_RANK0 "[direct] Auto-selected run_mode=single for subcommand '$_detected_subcmd'"
            break
        fi
    done
    direct_config[run_mode]="$_default_run_mode"
fi

###############################################################################
# STEP 4.5: Process non-cumulative parameters (use last value only)
###############################################################################

# For non-cumulative parameters with multiple values (config + CLI), use only the last value
for param in "script" "log_file"; do
    if [[ -n "${direct_config[$param]:-}" && "${direct_config[$param]}" == *$'\n'* ]]; then
        # Has newlines, take last value (CLI overrides config)
        last_value=$(echo "${direct_config[$param]}" | tail -1)
        direct_config[$param]="$last_value"
        LOG_DEBUG_RANK0 "[direct] Using last value for $param: $last_value"
    fi
done

###############################################################################
# STEP 4.6: Setup log file path
###############################################################################
if [[ -z "${direct_config[log_file]:-}" ]]; then
    direct_config[log_file]="logs/log_$(date +%Y%m%d_%H%M%S).txt"
fi
mkdir -p "$(dirname "${direct_config[log_file]:-}")"

###############################################################################
# STEP 4.7: Activate virtualenv (R1) and derive distributed env from SLURM (R2)
#
# Previously these lived in runner/run_preflight_direct.sh and
# runner/run_node_smoke_direct.sh (now deleted). Hoisting them here makes every
# `primus-cli direct -- ...` call site (host srun, slurm-entry -> direct,
# slurm-entry -> container -> direct) inherit identical behavior.
###############################################################################

# R1 -- Python virtualenv. VENV_ACTIVATE unset = no-op (this is the right
# default for the container path: primus-cli-container.sh does not auto-forward
# VENV_ACTIVATE through its env passthrough whitelist, so inside the container
# we use the container's bundled python). Set + missing = fail-fast: better a
# loud error than a silent torch-version mismatch.
if [[ -n "${VENV_ACTIVATE:-}" ]]; then
    if [[ ! -f "$VENV_ACTIVATE" ]]; then
        LOG_ERROR "[direct] VENV_ACTIVATE is set but file does not exist: $VENV_ACTIVATE"
        exit 1
    fi
    # shellcheck disable=SC1090
    source "$VENV_ACTIVATE"
    LOG_INFO_RANK0 "[direct] Activated virtualenv: $VENV_ACTIVATE"
fi

# R2 -- distributed env. Pre-set values always win (the existing
# slurm-entry -> container -> direct chain already passes them via --env, so
# this block is a no-op there). When called directly under srun on the host,
# this block derives them from SLURM_*.
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
export MASTER_PORT="${MASTER_PORT:-1234}"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    # Pre-exported values always win (per plan). The standard
    # slurm-entry -> container -> direct chain already exports NNODES /
    # NODE_RANK with the SLURM-derived values before direct.sh runs, so this
    # block is a no-op there. When called directly under srun on bare metal
    # without pre-set values, derive from SLURM_*.
    export NNODES="${NNODES:-${SLURM_NNODES:-${SLURM_JOB_NUM_NODES:-1}}}"
    export NODE_RANK="${NODE_RANK:-${SLURM_NODEID:-${SLURM_PROCID:-0}}}"
    if [[ -z "${MASTER_ADDR:-}" || "${MASTER_ADDR}" == "localhost" ]]; then
        if command -v scontrol >/dev/null 2>&1 && [[ -n "${SLURM_NODELIST:-}" ]]; then
            if ! MASTER_ADDR="$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)" \
               || [[ -z "$MASTER_ADDR" ]]; then
                LOG_ERROR "[direct] Failed to resolve MASTER_ADDR from SLURM_NODELIST=${SLURM_NODELIST:-<unset>}"
                exit 1
            fi
            export MASTER_ADDR
        else
            # In a SLURM context but we cannot resolve a real address from
            # scontrol -- either scontrol is not installed (CI / dev VM) or
            # SLURM_NODELIST was not propagated (rare; can happen in stubbed
            # tests or single-node allocations). Fall back to localhost so
            # the downstream sanity check at line 503 has a valid value;
            # NODE_RANK=0 + MASTER_ADDR=localhost is correct for single-node
            # SLURM runs and for dry-run smoke tests. Real multi-node
            # bare-srun on a SLURM head node always has both, so this
            # branch never fires in production.
            export MASTER_ADDR="localhost"
        fi
    fi
    LOG_INFO_RANK0 "[direct] SLURM detected: JOB_ID=$SLURM_JOB_ID NNODES=$NNODES NODE_RANK=$NODE_RANK MASTER_ADDR=${MASTER_ADDR:-<unset>}"
else
    export NNODES="${NNODES:-1}"
    export NODE_RANK="${NODE_RANK:-0}"
    export MASTER_ADDR="${MASTER_ADDR:-localhost}"
fi

# Sanity-check the resolved distributed env (lifted from the deleted wrappers).
[[ "$NNODES"      =~ ^[1-9][0-9]*$ ]] || { LOG_ERROR "[direct] NNODES must be a positive integer (got '$NNODES')"; exit 1; }
[[ "$NODE_RANK"   =~ ^[0-9]+$       ]] || { LOG_ERROR "[direct] NODE_RANK must be a non-negative integer (got '$NODE_RANK')"; exit 1; }
[[ "$MASTER_PORT" =~ ^[0-9]+$       ]] || { LOG_ERROR "[direct] MASTER_PORT must be a non-negative integer (got '$MASTER_PORT')"; exit 1; }
[[ -n "$MASTER_ADDR" ]]                || { LOG_ERROR "[direct] MASTER_ADDR is empty"; exit 1; }
(( NODE_RANK < NNODES ))               || { LOG_ERROR "[direct] NODE_RANK ($NODE_RANK) must be < NNODES ($NNODES)"; exit 1; }
if [[ "$MASTER_ADDR" == "localhost" && "${NNODES:-1}" -gt 1 ]]; then
  LOG_WARN "[direct] MASTER_ADDR=localhost with NNODES=$NNODES — multi-node will likely fail"
fi

###############################################################################
# STEP 5: Source GPU environment and helper modules
###############################################################################

# Source primus-env.sh (it will set its own SCRIPT_DIR, which is fine)
# shellcheck disable=SC1091
source "${RUNNER_DIR}/helpers/envs/primus-env.sh"

###############################################################################
# STEP 5.5: Auto-sync third_party sources for installed wheels
###############################################################################
# When running from an installed wheel (the packaged _thirdparty.lock exists),
# clone the pinned backend sources on first use and prepend them to PYTHONPATH,
# so backends like Megatron run from full source (Makefile/helpers can compile).
# Set PRIMUS_AUTO_DEPS_SYNC=0 to skip.
_PRIMUS_LOCK="${RUNNER_DIR}/../_thirdparty.lock"
if [[ "${PRIMUS_AUTO_DEPS_SYNC:-1}" != "0" && -f "$_PRIMUS_LOCK" ]]; then
    _PRIMUS_TP_DIR="${PRIMUS_THIRDPARTY_DIR:-$HOME/.cache/Primus/third_party}"
    if [[ ! -d "${_PRIMUS_TP_DIR}/Megatron-LM" || ! -d "${_PRIMUS_TP_DIR}/torchtitan" ]]; then
        LOG_INFO_RANK0 "[direct] third_party sources not found; running 'primus-cli deps sync' (set PRIMUS_AUTO_DEPS_SYNC=0 to skip)"
        bash "${RUNNER_DIR}/primus-cli-deps.sh" sync --dir "${_PRIMUS_TP_DIR}" || LOG_WARN "[direct] deps sync failed; continuing"
    fi
    if [[ -d "${_PRIMUS_TP_DIR}" ]]; then
        for _primus_tp in "${_PRIMUS_TP_DIR}"/*/; do
            [[ -d "$_primus_tp" ]] && export PYTHONPATH="${_primus_tp%/}${PYTHONPATH:+:$PYTHONPATH}"
        done
    fi
fi

###############################################################################
# STEP 6: Execute hooks and capture extra arguments / env
###############################################################################
# Hooks can return:
#   - Extra Primus CLI arguments by printing lines:  extra.<name>=<value>
#   - Environment variables by printing lines:       env.VAR_NAME=VALUE
# The detailed parsing logic lives in execute_hooks.sh, which fills the
# global HOOK_EXTRA_PRIMUS_ARGS array and exports env.* entries.
# shellcheck disable=SC1091
source "${RUNNER_DIR}/helpers/execute_hooks.sh"
HOOK_EXTRA_PRIMUS_ARGS=()
if ! execute_hooks "${1:-}" "${2:-}" "$@"; then
    LOG_ERROR "[direct] Hooks execution failed"
    exit 1
fi

# Prepend collected extra args to the current Primus arguments ($@) so that
# they are automatically picked up when building the final CMD below.
if [[ ${#HOOK_EXTRA_PRIMUS_ARGS[@]} -gt 0 ]]; then
    set -- "$@" "${HOOK_EXTRA_PRIMUS_ARGS[@]}"
fi

###############################################################################
# STEP 7: Execute patch scripts
###############################################################################
# Execute patch scripts from config + CLI.
# Note: direct_config[patch] is stored as a newline-separated list.
# Initialize so it is defined when no patches run (set -u safe).
PATCH_EXTRA_PRIMUS_ARGS=()
if [[ -n "${direct_config[patch]:-}" ]]; then
    # shellcheck disable=SC1091
    source "${RUNNER_DIR}/helpers/execute_patches.sh"
    # Reset collected extra args from patches for this run
    PATCH_EXTRA_PRIMUS_ARGS=()
    if ! execute_patches "${direct_config[patch]}"; then
        LOG_ERROR "[direct] Patch execution failed"
        exit 1
    fi
fi

###############################################################################
# STEP 7.5: Apply extra Primus args from patches (extra.* protocol)
###############################################################################
if [[ ${#PATCH_EXTRA_PRIMUS_ARGS[@]} -gt 0 ]]; then
    set -- "$@" "${PATCH_EXTRA_PRIMUS_ARGS[@]}"
    LOG_INFO_RANK0 "[direct] Applied extra args from patches: ${PATCH_EXTRA_PRIMUS_ARGS[*]}"
fi

###############################################################################
# STEP 8: Build and export environment variables (highest priority)
###############################################################################

# First, source any env files specified via --env <file> (tracked as env_file).
if [[ -n "${direct_config[env_file]:-}" ]]; then
    while IFS= read -r env_file; do
        [[ -n "$env_file" ]] || continue
        if [[ ! -f "$env_file" || ! -r "$env_file" ]]; then
            LOG_ERROR "[direct] Env file not found or not readable: $env_file"
            exit 1
        fi
        # shellcheck disable=SC1090
        source "$env_file"
        LOG_INFO_RANK0 "[direct] Sourced env file (final): $env_file"
    done <<< "${direct_config[env_file]:-}"
fi

# Then, build primus_env_kv array and export inline KEY=VALUE envs. This happens
# after hooks, patches, and env files so that CLI/config-provided envs
# (direct.env / --env KEY=VALUE) take final precedence for the actual Primus run.
primus_env_kv=()
if [[ -n "${direct_config[env]:-}" ]]; then
    while IFS= read -r env_entry; do
        [[ -n "$env_entry" ]] || continue
        # Validate env format (KEY=VALUE)
        if ! [[ "$env_entry" == *=* ]]; then
            LOG_ERROR "[direct] Invalid env format: $env_entry"
            LOG_ERROR "  Expected format: KEY=VALUE (e.g., NCCL_DEBUG=INFO)"
            exit 1
        fi
        primus_env_kv+=("$env_entry")
        export "${env_entry%%=*}"="${env_entry#*=}"
        LOG_INFO_RANK0 "[direct] Exported env (final): ${env_entry%%=*}=${env_entry#*=}"
    done <<< "${direct_config[env]:-}"
fi

# HipBLASLt stage-2 offline tuning may request skipping main training launch.
if [[ "${PRIMUS_SKIP_MAIN_LAUNCH:-0}" == "1" ]]; then
    LOG_INFO_RANK0 "[direct] PRIMUS_SKIP_MAIN_LAUNCH=1, skip main training launch."
    exit 0
fi

###############################################################################
# STEP 9: Build launch command
###############################################################################

# Final run-mode resolution. RUN_MODE-from-env wins over direct_config[run_mode]
# because framework prepare-hooks (e.g. runner/helpers/hooks/train/pretrain/
# maxtext/prepare.py) emit `env.RUN_MODE=single` from STEP 6 -- the hook layer
# knows things the launcher can't (e.g. "this framework is JAX, not torch, so
# torchrun would be wrong"). $RUN_MODE is the authoritative value for the rest
# of the script -- the display block and the torchrun-only Distributed Settings
# gate at STEP 10 both read $RUN_MODE, NOT direct_config[run_mode], so a hook
# that flips the mode is faithfully reflected in the printed configuration.
RUN_MODE="${RUN_MODE:-${direct_config[run_mode]:-torchrun}}"

# Resolve the launch target. Normally this is the script path
# (direct_config[script], default: primus/cli/main.py) relative to the repo
# checkout. When Primus is run from an installed wheel outside the repo, that
# file is absent; in that case fall back to the installed module form
# (-m primus.cli.main) so `primus-cli direct ...` works from any directory.
LAUNCH_TARGET=("${direct_config[script]:-}")
if [[ "${direct_config[script]:-}" == "primus/cli/main.py" && ! -f "${direct_config[script]:-}" ]]; then
    LAUNCH_TARGET=(-m primus.cli.main)
    LOG_INFO_RANK0 "[direct] Default script '${direct_config[script]}' not found in CWD; using installed module 'python -m primus.cli.main'"
fi

# Build the launch command as an ARRAY and execute it directly (no eval), so
# Primus arg values containing shell metacharacters are passed verbatim.
CMD=("${LAUNCH_TARGET[@]}" "$@")

if [[ "$RUN_MODE" == "single" ]]; then
    CMD=(python3 "${CMD[@]}")
    LOG_INFO_RANK0 "[direct] Using python launcher (single mode)"
elif [[ "$RUN_MODE" == "torchrun" ]]; then
    # Step 2: Add NUMA binding prefix if enabled
    if [[ "${direct_config[numa]:-}" == "true" ]]; then
        CMD=(--no-python "${RUNNER_DIR}/helpers/numa_bind.sh" "${CMD[@]}")
        LOG_INFO_RANK0 "[direct] NUMA binding: ENABLED (forced by CLI)"
    else
        LOG_INFO_RANK0 "[direct] NUMA binding: AUTO (default OFF)"
    fi

    DISTRIBUTED_ARGS=(
        --nproc_per_node "${GPUS_PER_NODE:-8}"
        --nnodes "${NNODES:-1}"
        --node_rank "${NODE_RANK:-0}"
        --master_addr "${MASTER_ADDR:-localhost}"
        --master_port "${MASTER_PORT:-1234}"
    )

    LAST_NODE=$((${NNODES:-1} - 1))
    FILTERS=()
    # Add local rank 0 on the first node
    if [ "${NODE_RANK:-0}" -eq 0 ]; then
        FILTERS+=(0)
    fi

    # Add the last local rank on the last node
    if [ "${NODE_RANK:-0}" -eq "$LAST_NODE" ] && [ "${NNODES:-1}" -ne 1 ]; then
        FILTERS+=($((${GPUS_PER_NODE:-8} - 1)))
    fi

    # Build filter argument (only if FILTERS is non-empty)
    if [ "${#FILTERS[@]}" -gt 0 ]; then
        LOCAL_FILTER=$(IFS=,; echo "${FILTERS[*]}")
        FILTER_ARG=(--local-ranks-filter "$LOCAL_FILTER")
    else
        FILTER_ARG=()
    fi

    # LOCAL_RANKS stays unquoted to allow word-splitting into multiple tokens.
    # shellcheck disable=SC2206
    CMD=(torchrun "${DISTRIBUTED_ARGS[@]}" "${FILTER_ARG[@]}" ${LOCAL_RANKS:-} "${CMD[@]}")
fi

###############################################################################
# STEP 10: Display configuration (always)
###############################################################################
if [[ "$DRY_RUN_MODE" == "1" ]]; then
    print_section "[DRY RUN] Direct Launch Configuration"
else
    print_section "Primus Direct Launch Configuration"
fi

PRINT_INFO_RANK0 "  Run Mode        : ${RUN_MODE}"
PRINT_INFO_RANK0 "  Script Path     : ${LAUNCH_TARGET[*]:-${direct_config[script]:-}}"
PRINT_INFO_RANK0 "  Config File     : ${CONFIG_FILE:-<none>}"
PRINT_INFO_RANK0 "  Log File        : ${direct_config[log_file]:-}"
PRINT_INFO_RANK0 "  NUMA Binding    : ${direct_config[numa]:-}"
if [[ -n "${direct_config[patch]:-}" ]]; then
    PRINT_INFO_RANK0 "  Patch Scripts   : $(echo "${direct_config[patch]:-}" | tr '\n' ' ')"
else
    PRINT_INFO_RANK0 "  Patch Scripts   : <none>"
fi
PRINT_INFO_RANK0 "  Primus Args     : $*"
PRINT_INFO_RANK0 ""

if [[ ${#primus_env_kv[@]} -gt 0 ]]; then
    PRINT_INFO_RANK0 "  Environment Variables:"
    for kv in "${primus_env_kv[@]}"; do
        PRINT_INFO_RANK0 "    $kv"
    done
    PRINT_INFO_RANK0 ""
fi

if [[ "${RUN_MODE}" == "torchrun" ]]; then
    PRINT_INFO_RANK0 "  Distributed Settings:"
    PRINT_INFO_RANK0 "    NNODES          : ${NNODES:-1}"
    PRINT_INFO_RANK0 "    NODE_RANK       : ${NODE_RANK:-0}"
    PRINT_INFO_RANK0 "    GPUS_PER_NODE   : ${GPUS_PER_NODE:-8}"
    PRINT_INFO_RANK0 "    MASTER_ADDR     : ${MASTER_ADDR:-localhost}"
    PRINT_INFO_RANK0 "    MASTER_PORT     : ${MASTER_PORT:-1234}"
    PRINT_INFO_RANK0 ""
fi

print_system_info

PRINT_INFO_RANK0 "  Full Command:"
if [[ "$DRY_RUN_MODE" == "1" ]]; then
    PRINT_INFO_RANK0 "    Would Execute: ${CMD[*]} 2>&1 | tee ${direct_config[log_file]}"
else
    PRINT_INFO_RANK0 "    Executing: ${CMD[*]} 2>&1 | tee ${direct_config[log_file]}"
fi

# In dry-run mode, exit after displaying the command
if [[ "$DRY_RUN_MODE" == "1" ]]; then
    print_section "End of Dry Run"
    LOG_INFO_RANK0 "[direct] Dry-run mode: command not executed"
    exit 0
fi

print_section ""

###############################################################################
# STEP 11: Execute command
###############################################################################
# PIPESTATUS[0] is the launcher's exit code.
set +e
"${CMD[@]}" 2>&1 | tee "${direct_config[log_file]}"
exit_code=${PIPESTATUS[0]}
set -e
# Print result based on exit code
if [[ $exit_code -ge 128 ]]; then
    LOG_ERROR "[direct] torchrun crashed due to signal $((exit_code - 128))"
elif [[ $exit_code -ne 0 ]]; then
    LOG_ERROR "[direct] torchrun exited with code $exit_code"
else
    LOG_INFO "[direct] torchrun finished successfully (code 0)"
fi

exit "$exit_code"
