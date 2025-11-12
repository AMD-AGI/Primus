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
        --image <DOCKER_IMAGE>       Docker image [default: rocm/primus:v25.9_gfx942]
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

HOSTNAME=$(hostname)

###############################################################################
# Helper Functions
###############################################################################

# Print container options with proper formatting
# Usage: print_container_options "prefix" "suffix" ARRAY_NAME
print_container_options() {
    local prefix="$1"
    local suffix="$2"
    local -n arr=$3  # nameref to array

    local i=0
    while [[ $i -lt ${#arr[@]} ]]; do
        local flag="${arr[i]}"
        i=$((i + 1))
        # Check if next argument exists and is not a flag
        if [[ $i -lt ${#arr[@]} ]] && [[ "${arr[i]}" != --* ]]; then
            # Key-value pair
            echo "${prefix}${flag} ${arr[i]}${suffix}"
            i=$((i + 1))
        else
            # Boolean flag
            echo "${prefix}${flag}${suffix}"
        fi
    done
}

###############################################################################
# STEP 1: Pre-parse global options (--config, --debug, --dry-run, --clean, --help)
###############################################################################
CONFIG_FILE=""
DEBUG_MODE=0
DRY_RUN_MODE=0
CLEAN_DOCKER_CONTAINER=0
PRE_PARSE_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --debug)
            export DEBUG_MODE=1
            shift
            ;;
        --dry-run)
            DRY_RUN_MODE=1
            shift
            ;;
        --clean)
            CLEAN_DOCKER_CONTAINER=1
            shift
            ;;
        --help|-h)
            print_usage
            exit 0
            ;;
        *)
            PRE_PARSE_ARGS+=("$1")
            shift
            ;;
    esac
done
# Restore arguments
set -- "${PRE_PARSE_ARGS[@]}"

# Enable debug mode early if set via CLI (so config loading shows DEBUG logs)
if [[ "$DEBUG_MODE" == "1" ]]; then
    export PRIMUS_LOG_LEVEL="DEBUG"
    LOG_INFO_RANK0 "[container] Debug mode enabled via CLI (PRIMUS_LOG_LEVEL=DEBUG)"
fi

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
if [[ "$DEBUG_MODE" == "0" ]]; then
    debug_value="${container_config[debug]:-false}"
    if [[ "$debug_value" == "true" || "$debug_value" == "1" ]]; then
        DEBUG_MODE=1
        export PRIMUS_LOG_LEVEL="DEBUG"
        LOG_INFO_RANK0 "[container] Debug mode enabled via config (PRIMUS_LOG_LEVEL=DEBUG)"
    fi
fi

if [[ "$DRY_RUN_MODE" == "0" ]]; then
    dry_run_value="${container_config[dry_run]:-false}"
    if [[ "$dry_run_value" == "true" || "$dry_run_value" == "1" ]]; then
        DRY_RUN_MODE=1
        LOG_INFO_RANK0 "[container] Dry-run mode enabled via config"
    fi
fi

LOG_DEBUG_RANK0 "[container] Configuration loaded from file, ready for CLI argument override"

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
            POSITIONAL_ARGS+=("$@")
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
                LOG_DEBUG_RANK0 "[container] CLI: $config_key = true"
                shift
            else
                # Key-value option: append with newline (all stored as multi-value)
                if [[ -z "${container_config[$config_key]:-}" ]] || \
                   [[ "${container_config[$config_key]}" == "[]" ]]; then
                    container_config[$config_key]="$opt_value"
                else
                    container_config[$config_key]+=$'\n'"$opt_value"
                fi
                LOG_DEBUG_RANK0 "[container] CLI: $config_key += $opt_value"
                shift 2
            fi
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

###############################################################################
# STEP 4.5: Validate required parameters
###############################################################################

LOG_DEBUG_RANK0 "[container] Validating configuration..."

# 1. Validate that required parameters are set
if [[ -z "${container_config[options.image]:-}" ]]; then
    LOG_ERROR "[container] Missing required parameter: --image"
    LOG_ERROR "  Please specify a container image via:"
    LOG_ERROR "    1. Command line: --image <DOCKER_IMAGE>"
    LOG_ERROR "    2. Config file: container.options.image: <DOCKER_IMAGE>"
    exit 1
fi

# 2. Validate that Primus commands are provided
if [[ ${#POSITIONAL_ARGS[@]} -eq 0 ]]; then
    LOG_ERROR "[container] Missing Primus commands after '--'"
    LOG_ERROR "  Usage: primus-cli container [options] -- <primus-commands>"
    LOG_ERROR "  Example: primus-cli container --image rocm/primus:latest -- train pretrain"
    exit 1
fi

# 3. Validate that GPU devices are configured and exist on host
devices="${container_config[options.device]:-}"
if [[ -z "$devices" || "$devices" == "[]" ]]; then
    LOG_ERROR "[container] No GPU devices configured"
    LOG_ERROR "  Primus requires GPU devices for training. Please configure:"
    LOG_ERROR "    container.options.device:"
    LOG_ERROR "      - \"/dev/kfd\"  # Kernel Fusion Driver (ROCm core)"
    LOG_ERROR "      - \"/dev/dri\"  # Direct Rendering Infrastructure (GPU access)"
    LOG_ERROR "  Or use --device CLI argument to add devices"
    exit 1
fi

LOG_DEBUG_RANK0 "[container] Validating device paths on host..."
validation_failed=0
while IFS= read -r device; do
    [[ -n "$device" ]] || continue
    if [[ ! -e "$device" ]]; then
        LOG_ERROR "[container] Device does not exist on host: $device"
        validation_failed=1
    else
        LOG_DEBUG_RANK0 "[container] Device validated: $device"
    fi
done <<< "$devices"

if [[ $validation_failed -eq 1 ]]; then
    LOG_ERROR "[container] Device validation failed"
    LOG_ERROR "  Please ensure ROCm drivers are properly installed on the host"
    LOG_ERROR "  Check device availability: ls -la /dev/kfd /dev/dri"
    exit 1
fi

# 4. Validate common parameter formats (optional but helpful)
LOG_DEBUG_RANK0 "[container] Validating parameter formats..."

# Validate memory format (if specified)
memory="${container_config[options.memory]:-}"
if [[ -n "$memory" && "$memory" != *$'\n'* ]]; then
    if ! [[ "$memory" =~ ^[0-9]+[bkmgBKMG]?$ ]]; then
        LOG_ERROR "[container] Invalid memory format: $memory"
        LOG_ERROR "  Expected format: <number>[b|k|m|g] (e.g., 256G, 1024M)"
        exit 1
    fi
fi

# Validate cpus format (if specified)
cpus="${container_config[options.cpus]:-}"
if [[ -n "$cpus" && "$cpus" != *$'\n'* ]]; then
    if ! [[ "$cpus" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        LOG_ERROR "[container] Invalid cpus format: $cpus"
        LOG_ERROR "  Expected format: <number> or <number>.<decimal> (e.g., 32, 16.5)"
        exit 1
    fi
fi

# Validate env format (KEY=VALUE)
env_vars="${container_config[options.env]:-}"
if [[ -n "$env_vars" && "$env_vars" != "[]" ]]; then
    while IFS= read -r env_entry; do
        [[ -n "$env_entry" ]] || continue
        if ! [[ "$env_entry" == *=* ]]; then
            LOG_ERROR "[container] Invalid env format: $env_entry"
            LOG_ERROR "  Expected format: KEY=VALUE (e.g., NCCL_DEBUG=INFO)"
            exit 1
        fi
    done <<< "$env_vars"
fi

# Validate volume format
volumes="${container_config[options.volume]:-}"
if [[ -n "$volumes" && "$volumes" != "[]" ]]; then
    while IFS= read -r volume_entry; do
        [[ -n "$volume_entry" ]] || continue

        # Volume format: /host:/container[:options] or /path or named_volume:/container[:options]
        # Check if volume has colons
        if [[ "$volume_entry" == *:* ]]; then
            IFS=':' read -r src dst opts <<< "$volume_entry"

            # Check that source is not empty
            if [[ -z "$src" ]]; then
                LOG_ERROR "[container] Invalid volume format: $volume_entry"
                LOG_ERROR "  Source path cannot be empty"
                exit 1
            fi

            # If colon is present, destination must not be empty (unless it's just src:dst format)
            # Count colons to distinguish /host: from /host:/dst
            colon_count=$(echo "$volume_entry" | grep -o ":" | wc -l)
            if [[ $colon_count -ge 1 && -z "$dst" ]]; then
                LOG_ERROR "[container] Invalid volume format: $volume_entry"
                LOG_ERROR "  Destination path cannot be empty when colon is present"
                exit 1
            fi
        else
            # Single path format (e.g., /workspace)
            src="$volume_entry"
            dst=""
            opts=""
        fi

        # If options specified, validate they are valid (ro, rw, z, Z, etc.)
        if [[ -n "$opts" ]]; then
            # Split multiple options (e.g., "ro,z")
            IFS=',' read -ra opt_array <<< "$opts"
            for opt in "${opt_array[@]}"; do
                if ! [[ "$opt" =~ ^(ro|rw|z|Z|shared|slave|private|delegated|cached|consistent)$ ]]; then
                    LOG_ERROR "[container] Invalid volume option: $opt in $volume_entry"
                    LOG_ERROR "  Valid options: ro, rw, z, Z, shared, slave, private, delegated, cached, consistent"
                    exit 1
                fi
            done
        fi

        LOG_DEBUG_RANK0 "[container] Volume validated: $volume_entry"
    done <<< "$volumes"
fi

LOG_DEBUG_RANK0 "[container] Parameter validation passed"

###############################################################################
# STEP 5: Convert container_config to Docker/Podman options
# Now we have a complete container_config with CLI overrides applied
###############################################################################

LOG_DEBUG_RANK0 "[container] Converting configuration to container options..."

# 1. Image (required, validated above)
# If image has multiple values (config + CLI), use the last one (CLI overrides)
image_value="${container_config[options.image]}"
if [[ "$image_value" == *$'\n'* ]]; then
    DOCKER_IMAGE=$(echo "$image_value" | tail -1)
else
    DOCKER_IMAGE="$image_value"
fi
LOG_DEBUG_RANK0 "[container] Final image: $DOCKER_IMAGE"

# 2. Build CONTAINER_OPTS from configuration
CONTAINER_OPTS=()

# Cumulative options (all values used, config + CLI merge)
CUMULATIVE_OPTIONS=("device" "cap-add" "volume" "env")

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
                LOG_DEBUG_RANK0 "[container] Added cumulative: --${opt_name} $val"
            done <<< "$opt_value"
        else
            # Non-cumulative: only use last value (CLI overrides config)
            last_value=$(echo "$opt_value" | tail -1)
            CONTAINER_OPTS+=("--${opt_name}" "$last_value")
            LOG_DEBUG_RANK0 "[container] Added option (last): --${opt_name} $last_value"
        fi
    elif [[ "$opt_value" == "true" || "$opt_value" == "1" ]]; then
        # Boolean flag: only add flag name (no value)
        CONTAINER_OPTS+=("--${opt_name}")
        LOG_DEBUG_RANK0 "[container] Added boolean flag: --${opt_name}"
    else
        # Single value option
        CONTAINER_OPTS+=("--${opt_name}" "$opt_value")
        LOG_DEBUG_RANK0 "[container] Added option: --${opt_name} $opt_value"
    fi
done

###############################################################################
# STEP 6: Build final volume arguments (project root is always mounted)
###############################################################################

VOLUME_ARGS=(-v "$PRIMUS_PATH:$PRIMUS_PATH")
LOG_DEBUG_RANK0 "[container] Added project root volume: $PRIMUS_PATH"

###############################################################################
# STEP 7: Detect container runtime (docker/podman)
###############################################################################

if command -v podman >/dev/null 2>&1; then
    DOCKER_CLI="podman"
elif command -v docker >/dev/null 2>&1; then
    DOCKER_CLI="docker"
else
    # In dry-run mode, we don't need actual docker/podman
    if [[ "$DRY_RUN_MODE" == "1" ]]; then
        DOCKER_CLI="docker"  # Use docker for dry-run output
    else
        LOG_ERROR "[container] Neither Docker nor Podman found!"
        exit 1
    fi
fi

###############################################################################
# STEP 8: Optional container cleanup
###############################################################################

if [[ "$CLEAN_DOCKER_CONTAINER" == "1" ]]; then
    LOG_INFO_RANK0 "[container] Cleaning up existing containers..."
    CONTAINERS="$($DOCKER_CLI ps -aq)"
    if [[ -n "$CONTAINERS" ]]; then
        printf '%s\n' "$CONTAINERS" | xargs -r -n1 "$DOCKER_CLI" rm -f
        LOG_INFO_RANK0 "[container] Removed containers: $CONTAINERS"
    else
        LOG_INFO_RANK0 "[container] No containers to remove."
    fi
fi

###############################################################################
# STEP 9: Prepare launch arguments
###############################################################################

ARGS=("${POSITIONAL_ARGS[@]}")
OPTION_ARGS=("${CONTAINER_OPTS[@]}")

###############################################################################
# STEP 10: Display launch information
###############################################################################

PRINT_INFO_RANK0 ""
if [[ "$DRY_RUN_MODE" == "1" ]]; then
    PRINT_INFO_RANK0 "========== [DRY RUN] Container Launch Info ($DOCKER_CLI) =========="
else
    PRINT_INFO_RANK0 "========== Container Launch Info ($DOCKER_CLI) =========="
fi
PRINT_INFO_RANK0 "  IMAGE: $DOCKER_IMAGE"
PRINT_INFO_RANK0 "  HOSTNAME: $HOSTNAME"
PRINT_INFO_RANK0 "  VOLUME_ARGS:"
for ((i = 0; i < ${#VOLUME_ARGS[@]}; i+=2)); do
    PRINT_INFO_RANK0 "      ${VOLUME_ARGS[i]} ${VOLUME_ARGS[i+1]}"
done
if [[ ${#OPTION_ARGS[@]} -gt 0 ]]; then
    PRINT_INFO_RANK0 "  CONTAINER_OPTIONS:"
    opt_i=0
    while [[ $opt_i -lt ${#OPTION_ARGS[@]} ]]; do
        opt_flag="${OPTION_ARGS[opt_i]}"
        opt_i=$((opt_i + 1))
        # Check if next argument exists and is not a flag
        if [[ $opt_i -lt ${#OPTION_ARGS[@]} ]] && [[ "${OPTION_ARGS[opt_i]}" != --* ]]; then
            # Key-value pair
            PRINT_INFO_RANK0 "      ${opt_flag} ${OPTION_ARGS[opt_i]}"
            opt_i=$((opt_i + 1))
        else
            # Boolean flag
            PRINT_INFO_RANK0 "      ${opt_flag}"
        fi
    done
fi
PRINT_INFO_RANK0 "  LAUNCH ARGS:"
PRINT_INFO_RANK0 "      ${ARGS[*]}"
PRINT_INFO_RANK0 "================================================"

###############################################################################
# STEP 11: Build and execute container command
###############################################################################

# Build the container entrypoint script
CONTAINER_SCRIPT="\
    echo [container][INFO]: container started at \$(date +%Y.%m.%d) \$(date +%H:%M:%S) && \
    [[ -d $PRIMUS_PATH ]] || { echo '[container][ERROR]: Primus not found at $PRIMUS_PATH' >&2; exit 42; } && \
    cd $PRIMUS_PATH && bash runner/primus-cli-direct.sh \"\$@\" 2>&1 && \
    echo [container][INFO]: container finished at \$(date +%Y.%m.%d) \$(date +%H:%M:%S)"

# Build complete command array
CMD=(
    "${DOCKER_CLI}"
    run
    --rm
    "${VOLUME_ARGS[@]}"
    "${OPTION_ARGS[@]}"
    "$DOCKER_IMAGE"
    /bin/bash
    -c
    "$CONTAINER_SCRIPT"
    bash
    "${ARGS[@]}"
)

# Print full command in dry-run mode
if [[ "$DRY_RUN_MODE" == "1" ]]; then
    PRINT_INFO_RANK0 ""
    PRINT_INFO_RANK0 "  Full Command: ${CMD[*]}"
    PRINT_INFO_RANK0 "================================================"
    PRINT_INFO_RANK0 ""
fi

# Exit if dry-run mode
if [[ "$DRY_RUN_MODE" == "1" ]]; then
    exit 0
fi

# Execute the command
"${CMD[@]}"
