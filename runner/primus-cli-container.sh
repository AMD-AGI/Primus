#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Primus Container Mode Launcher
#
# This script launches Primus workflows in a Docker/Podman container.
#
# Execution Flow:
#   1. Parse global options (--config, --debug, --dry-run)
#   2. Load configuration from YAML files
#   3. Extract and apply container.* configuration parameters
#   4. Parse CLI arguments (--image, --volume, generic docker options)
#   5. Build volume mounts and container options
#   6. Detect docker/podman CLI
#   7. Launch container with primus-cli-direct.sh inside
#
###############################################################################

set -euo pipefail

print_usage() {
cat <<EOF
Usage: bash primus-run-container.sh [OPTIONS] -- [SCRIPT_ARGS...]

Launch a Primus task (train / benchmark / preflight / etc.) in a Docker/Podman container.

Global Options:
    --config <FILE>             Load configuration from specified YAML file
    --debug                     Enable debug mode (verbose logging)
    --dry-run                   Show what would be executed without running
    --clean                     Remove all containers before launch
    --help, -h                  Show this message and exit

Docker/Podman Options:
    All docker/podman run options are supported. Some key options have special handling:

    Cumulative Options (can be specified multiple times):
        --volume <HOST[:CONTAINER]>  Mount volumes. If only HOST given, mounts to same path.
        --env KEY=VALUE              Set environment variables
        --device <DEVICE_PATH>       Add host device access (e.g., /dev/kfd, /dev/dri)
        --cap-add <CAPABILITY>       Add Linux capabilities (e.g., SYS_PTRACE)

    Container Configuration:
        --image <DOCKER_IMAGE>       Docker image [default: rocm/primus:v25.10]
        --name <NAME>                Container name
        --user <UID:GID>             Run as specific user (e.g., 1000:1000)
        --network <NET>              Network mode (e.g., host, bridge)
        --ipc <MODE>                 IPC mode (e.g., host, private)

    Resource Limits:
        --cpus <N>                   Limit CPU cores (e.g., 8, 16.5)
        --memory <SIZE>              Limit memory (e.g., 64G, 128G)
        --shm-size <SIZE>            Shared memory size (e.g., 16G)
        --gpus <N>                   GPU limit (for nvidia-docker)

    Note: Any other docker/podman run option (e.g., --privileged, --rm) is also supported.

Examples:
    # Basic training with mounted data
    primus-cli container --volume /mnt/data -- train --config /mnt/data/exp.yaml

    # Run with resource limits
    primus-cli container --cpus 16 --memory 128G --gpus 8 -- train pretrain

    # Run as specific user
    primus-cli container --user 1000:1000 -- benchmark gemm

    # Use configuration file
    primus-cli --config .primus.yaml container -- train
EOF
}

if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    print_usage
    exit 0
fi

###############################################################################
# STEP 0: Initialization
###############################################################################

# Resolve runner directory
RUNNER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load common library (required)
# shellcheck disable=SC1091
source "$RUNNER_DIR/lib/common.sh" || {
    echo "[ERROR] Failed to load common library: $RUNNER_DIR/lib/common.sh" >&2
    exit 1
}

# Now we can use common.sh functions
PRIMUS_PATH="$(get_absolute_path "$RUNNER_DIR/..")"

# Load config library (required)
# shellcheck disable=SC1091
source "$RUNNER_DIR/lib/config.sh" || {
    LOG_ERROR "[container] Failed to load config library: $RUNNER_DIR/lib/config.sh"
    exit 1
}

# Load validation library (required)
# shellcheck disable=SC1091
source "$RUNNER_DIR/lib/validation.sh" || {
    LOG_ERROR "[container] Failed to load validation library: $RUNNER_DIR/lib/validation.sh"
    exit 1
}

HOSTNAME=$(hostname)

LOG_INFO_RANK0 "-----------------------------------------------"
LOG_INFO_RANK0 "primus-cli-container.sh"
LOG_INFO_RANK0 "-----------------------------------------------"


###############################################################################
# STEP 1: Pre-parse global options (--config, --debug, --dry-run, --clean, --help)
###############################################################################
CONFIG_FILE=""
DEBUG_MODE=false
DRY_RUN_MODE=false
CLEAN_DOCKER_CONTAINER=false
PRE_PARSE_ARGS=()
POST_PARSE_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --)
            shift
            POST_PARSE_ARGS+=("$@")  # Append all arguments after '--'
            break
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --debug)
            DEBUG_MODE=true
            export PRIMUS_LOG_LEVEL="DEBUG"
            shift
            ;;
        --dry-run)
            DRY_RUN_MODE=true
            shift
            ;;
        --clean)
            CLEAN_DOCKER_CONTAINER=true
            shift
            ;;
        --help|-h)
            print_usage
            exit 0
            ;;
        --*)
            # Unrecognized container-level option: keep in PRE_PARSE_ARGS.
            # If it has a non-option value (e.g., '--shm-size 8g'), grab both.
            PRE_PARSE_ARGS+=("$1")
            if [[ "$#" -ge 2 && "$2" != --* ]]; then
                PRE_PARSE_ARGS+=("$2")
                shift 2
            else
                shift
            fi
            ;;
        *)
            # Collect this and all remaining arguments as post-parse args so
            # they are forwarded after '--' (to Primus CLI) without being
            # treated as container-global options.
            POST_PARSE_ARGS+=("$@")
            break
    esac
done
# Restore arguments
set -- "${PRE_PARSE_ARGS[@]}" -- "${POST_PARSE_ARGS[@]}"


###############################################################################
# STEP 2: Load configuration files
###############################################################################

load_config_auto "$CONFIG_FILE" "container" || {
    LOG_ERROR "[container] Configuration loading failed"
    exit 1
}

# Extract container.* config parameters
declare -A container_config
extract_config_section "container" container_config || {
    LOG_ERROR "[container] Failed to extract container config section"
    exit 1
}

###############################################################################
# STEP 3: Process configuration from file
# Note: container_config already loaded in STEP 2, we just check debug/dry-run here
###############################################################################

# Check debug/dry-run from config first (so subsequent processing shows DEBUG logs)
if [[ "$DEBUG_MODE" == "false" ]]; then
    debug_value="${container_config[debug]:-false}"
    if [[ "$debug_value" == "true" ]]; then
        DEBUG_MODE=true
        export PRIMUS_LOG_LEVEL="DEBUG"
        LOG_INFO_RANK0 "[container] Debug mode enabled via config (PRIMUS_LOG_LEVEL=DEBUG)"
    fi
fi

if [[ "$DRY_RUN_MODE" == "false" ]]; then
    dry_run_value="${container_config[dry_run]:-false}"
    if [[ "$dry_run_value" == "true" ]]; then
        DRY_RUN_MODE=true
        LOG_INFO_RANK0 "[container] Dry-run mode enabled via config"
    fi
fi

# Validate container runtime (docker/podman)
# In dry-run mode, allow mock runtime if docker/podman not available
if [[ "$DRY_RUN_MODE" == "true" ]]; then
    if command -v podman >/dev/null 2>&1; then
        export CONTAINER_RUNTIME="podman"
    elif command -v docker >/dev/null 2>&1; then
        export CONTAINER_RUNTIME="docker"
    else
        # Mock runtime for dry-run testing
        export CONTAINER_RUNTIME="docker"
        LOG_INFO_RANK0 "[container] Using mock container runtime for dry-run (no docker/podman found)"
    fi
else
    validate_container_runtime
fi

###############################################################################
# STEP 4: Parse container-specific CLI arguments
# Process Docker/Podman runtime options (--image, --volume, --memory, etc.)
# and override corresponding values in container_config
# Priority: CLI args > Config file
###############################################################################

POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --)
            shift
            POSITIONAL_ARGS+=("${@}")
            break
            ;;
        --*)
            # Generic docker option (--key value or --boolean-flag)
            opt_name="${1#--}"
            opt_value="${2:-}"
            config_key="options.$opt_name"

            if [[ -z "$opt_value" ]] || [[ "$opt_value" == --* ]]; then
                # Boolean flag (next arg is empty or starts with --)
                container_config[$config_key]="true"
                LOG_INFO_RANK0 "[container] CLI: $config_key = true"
                shift
            else
                # Key-value option: append with newline (all stored as multi-value)
                if [[ -z "${container_config[$config_key]:-}" ]] || \
                   [[ "${container_config[$config_key]}" == "[]" ]]; then
                    container_config[$config_key]="$opt_value"
                else
                    container_config[$config_key]+=$'\n'"$opt_value"
                fi
                LOG_INFO_RANK0 "[container] CLI: $config_key += $opt_value"
                shift 2
            fi
            ;;
        *)
            POSITIONAL_ARGS+=("${1}")
            shift
            ;;
    esac
done

###############################################################################
# STEP 4.5: Validate required parameters
###############################################################################
set -- "${POSITIONAL_ARGS[@]}"
LOG_INFO_RANK0 "[container] Validating configuration..."

# Validate required parameters
validate_config_param \
    "${container_config[options.image]:-}" \
    "container.options.image" \
    "[container] Missing required parameter: --image. Specify via CLI (--image <IMAGE>) or config file (container.options.image: <IMAGE>)"

validate_positional_args \
    POSITIONAL_ARGS \
    "[container] Missing Primus commands after '--'. Usage: primus-cli container [options] -- <primus-commands>"

validate_device_paths \
    "${container_config[options.device]:-}" \
    "[container]" \
    "[container] No GPU devices configured. Specify via CLI (--device /dev/kfd --device /dev/dri) or config file:
  container:
    options:
      device:
        - \"/dev/kfd\"
        - \"/dev/dri\"
        - \"/dev/infiniband\"" \
    "[container] Device validation failed. Ensure ROCm drivers are installed on host. Check: ls -la /dev/kfd /dev/dri /dev/infiniband"

# Validate parameter formats (if specified)
validate_memory_format \
    "${container_config[options.memory]:-}" \
    "container.options.memory" \
    "[container] Invalid memory format: ${container_config[options.memory]:-}. Use format <number>[b|k|m|g] (e.g., --memory 256G or config: memory: 1024M)"

validate_cpus_format \
    "${container_config[options.cpus]:-}" \
    "container.options.cpus" \
    "[container] Invalid cpus format: ${container_config[options.cpus]:-}. Use format <number>[.<decimal>] (e.g., --cpus 32 or config: cpus: 16.5)"

# Convert container.options.env into inner Primus --env arguments instead of
# treating them as container-level KEY=VALUE pairs. This allows config-driven
# env propagation to work uniformly with primus-cli-direct semantics:
#   - "KEY=VALUE"  → --env KEY=VALUE
#   - "KEY"        → if KEY is set in current env, expand to KEY=$VALUE;
#                    otherwise pass through as bare KEY.
env_positional_args=()
declare -A env_keys_seen=()

if [[ -n "${container_config[options.env]:-}" ]]; then
    while IFS= read -r env_entry; do
        [[ -n "$env_entry" ]] || continue

        env_kv="$env_entry"
        # Only expand KEY→KEY=VALUE when:
        #   - there is no '=' present, and
        #   - the entry looks like a shell identifier (avoids paths like ./foo.sh)
        if [[ "$env_entry" != *"="* && "$env_entry" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
            env_key="$env_entry"
            env_val="${!env_key-}"
            if [[ -n "$env_val" ]]; then
                env_kv="${env_key}=${env_val}"
            else
                # If KEY is not set in the current environment, ignore this entry
                # instead of passing a bare KEY through.
                continue
            fi
        fi

        env_key="${env_kv%%=*}"
        env_keys_seen["$env_key"]=1
        env_positional_args+=(--env "$env_kv")
    done <<< "${container_config[options.env]}"
fi

# Auto-pass through common runtime/perf/network env vars into container when present.
# This keeps container runs closer to direct-mode behavior without requiring users
# to manually list every env var in config/CLI.
while IFS= read -r env_key; do
    [[ "$env_key" =~ ^(PRIMUS_|NCCL_|GLOO_) ]] || continue
    [[ -n "${env_keys_seen[$env_key]:-}" ]] && continue

    env_val="${!env_key-}"
    [[ -n "$env_val" ]] || continue

    env_keys_seen["$env_key"]=1
    env_positional_args+=(--env "${env_key}=${env_val}")
done < <(compgen -e)

if [[ ${#env_positional_args[@]} -gt 0 ]]; then
    # Prepend env-derived args so they appear before other POSITIONAL_ARGS
    POSITIONAL_ARGS=( "${env_positional_args[@]}" "${POSITIONAL_ARGS[@]}" )
fi

# Validate volume format
validate_volume_format "${container_config[options.volume]:-}" "[container]"

LOG_INFO_RANK0 "[container] Parameter validation passed"

###############################################################################
# STEP 5: Convert container_config to Docker/Podman options
# Now we have a complete container_config with CLI overrides applied
###############################################################################

LOG_INFO_RANK0 "[container] Converting configuration to container options..."

# 1. Image (required, validated above)
# For single-value options like image, take the last value (CLI overrides config)
DOCKER_IMAGE=$(echo "${container_config[options.image]}" | tail -n1)
LOG_INFO_RANK0 "[container] Final image: $DOCKER_IMAGE"

# 2. Build CONTAINER_OPTS from configuration
CONTAINER_OPTS=()

# Always mount project root directory first
CONTAINER_OPTS+=("-v" "$PRIMUS_PATH:$PRIMUS_PATH")
LOG_INFO_RANK0 "[container] Added project root volume: $PRIMUS_PATH"

# Cumulative options (all values used, config + CLI merge)
# Note: options.env is handled separately above and is NOT treated as a
# container-level --env; it becomes inner primus-cli --env arguments instead.
CUMULATIVE_OPTIONS=("device" "cap-add" "volume")

for key in "${!container_config[@]}"; do
    [[ "$key" =~ ^options\. ]] || continue

    opt_name="${key#options.}"
    opt_value="${container_config[$key]}"

    # Skip image (used separately) and empty array markers
    [[ "$opt_name" == "image" ]] && continue
    [[ "$opt_value" == "[]" ]] && continue

    # Check if this is a cumulative option
    is_cumulative=0
    for cum_opt in "${CUMULATIVE_OPTIONS[@]}"; do
        if [[ "$opt_name" == "$cum_opt" ]]; then
            is_cumulative=1
            break
        fi
    done

    # Check if value contains newlines (multi-value)
    if [[ "$opt_value" == *$'\n'* ]]; then
        if [[ $is_cumulative -eq 1 ]]; then
            # Cumulative: use all values
            while IFS= read -r val; do
                [[ -n "$val" ]] || continue
                CONTAINER_OPTS+=("--${opt_name}" "$val")
                LOG_INFO_RANK0 "[container] Added cumulative: --${opt_name} $val"
            done <<< "$opt_value"
        else
            # Non-cumulative: only use last value (CLI overrides config)
            last_value=$(echo "$opt_value" | tail -1)
            CONTAINER_OPTS+=("--${opt_name}" "$last_value")
            LOG_INFO_RANK0 "[container] Added option (last): --${opt_name} $last_value"
        fi
    elif [[ "$opt_value" == "true" || "$opt_value" == "1" ]]; then
        # Boolean flag: only add flag name (no value)
        CONTAINER_OPTS+=("--${opt_name}")
        LOG_INFO_RANK0 "[container] Added boolean flag: --${opt_name}"
    else
        # Single value option
        CONTAINER_OPTS+=("--${opt_name}" "$opt_value")
        LOG_INFO_RANK0 "[container] Added option: --${opt_name} $opt_value"
    fi
done


###############################################################################
# STEP 6: Optional container cleanup
###############################################################################

if [[ "$CLEAN_DOCKER_CONTAINER" == "true" ]]; then
    LOG_INFO_RANK0 "[container] Cleaning up existing containers..."
    CONTAINERS="$($CONTAINER_RUNTIME ps -aq)"
    if [[ -n "$CONTAINERS" ]]; then
        printf '%s\n' "$CONTAINERS" | xargs -r -n1 "$CONTAINER_RUNTIME" rm -f
        LOG_INFO_RANK0 "[container] Removed containers: $CONTAINERS"
    else
        LOG_INFO_RANK0 "[container] No containers to remove."
    fi
fi

###############################################################################
# STEP 7: Prepare launch arguments
###############################################################################

ARGS=()
# Add global options first
if [[ -n "$CONFIG_FILE" ]]; then
    ARGS+=(--config "$CONFIG_FILE")
fi
if [[ "$DEBUG_MODE" == "true" ]]; then
    ARGS+=(--debug)
fi
# Add positional arguments
ARGS+=( "${POSITIONAL_ARGS[@]}")

OPTION_ARGS=("${CONTAINER_OPTS[@]}")


###############################################################################
# STEP 8: Build and execute container command
###############################################################################

# Build the container entrypoint script
CONTAINER_SCRIPT="\
    echo [container ${NODE_RANK:-0}][INFO]: started at \$(date +%Y.%m.%d) \$(date +%H:%M:%S) && \
    [[ -d $PRIMUS_PATH ]] || { echo '[container ${NODE_RANK:-0}][ERROR]: Primus not found at $PRIMUS_PATH' >&2; exit 42; } && \
    cd $PRIMUS_PATH && bash runner/primus-cli-direct.sh \"\$@\" 2>&1 && \
    echo [container ${NODE_RANK:-0}][INFO]: finished at \$(date +%Y.%m.%d) \$(date +%H:%M:%S)"

# Build complete command array
CMD=(
    "${CONTAINER_RUNTIME}"
    run
    --rm
    "${OPTION_ARGS[@]}"
    "$DOCKER_IMAGE"
    /bin/bash
    -c
    "$CONTAINER_SCRIPT"
    bash
    "${ARGS[@]}"
)

# Display command
LOG_INFO_RANK0 "[container] Launching container with the following configuration:"
LOG_INFO_RANK0 "    Runtime: ${CONTAINER_RUNTIME}"
LOG_INFO_RANK0 "    Image: ${DOCKER_IMAGE}"
LOG_INFO_RANK0 "    Container options:"
# Display container options in pairs
opt_i=0
while [[ $opt_i -lt ${#OPTION_ARGS[@]} ]]; do
    opt="${OPTION_ARGS[opt_i]}"
    opt_i=$((opt_i + 1))
    # Check if next element exists and is not a flag
    if [[ $opt_i -lt ${#OPTION_ARGS[@]} ]] && [[ "${OPTION_ARGS[opt_i]}" != -* ]]; then
        # Option with value
        LOG_INFO_RANK0 "        ${opt} ${OPTION_ARGS[opt_i]}"
        opt_i=$((opt_i + 1))
    else
        # Boolean flag
        LOG_INFO_RANK0 "        ${opt}"
    fi
done
LOG_INFO_RANK0 "    Args: ${ARGS[*]}"
LOG_INFO_RANK0 "[container] Would execute: ${CMD[*]}"

if [[ "$DRY_RUN_MODE" == "true" ]]; then
    LOG_INFO "[container] Dry-run mode: command not executed"
    exit 0
fi

LOG_INFO "[container] Executing command..."
"${CMD[@]}"
