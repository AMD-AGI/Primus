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
#   4. Parse CLI arguments (--image, --mount, generic docker options)
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

Built-in Options:
    --image <DOCKER_IMAGE>      Docker image to use [default: rocm/primus:v25.9_gfx942]
    --mount <HOST[:CONTAINER]>  Mount a host directory into the container.
                                 - If only HOST is given, mounts to same path inside container.
                                 - If HOST:CONTAINER is given, mounts host directory to container path.
                                 (repeatable; for data, output, cache, etc.)
    --primus-path <HOST_PATH>   Use this Primus repo instead of the image default. The path will be mounted
                                into the container and installed in editable mode.
    --clean                     Remove all containers before launch
    --dry-run                   Show what would be executed without running
    --help                      Show this message and exit

Generic Docker/Podman Options:
    Any --<option> <value> pairs are passed directly to docker/podman run.
    This provides full flexibility for all docker runtime options.

    Examples:
        --cpus <N>              Limit CPU cores (e.g., 8, 16.5)
        --memory <SIZE>         Limit memory (e.g., 64G, 128G)
        --shm-size <SIZE>       Set shared memory size (e.g., 16G)
        --gpus <N>              GPU limit (for nvidia-docker)
        --user <UID:GID>        Run as specific user (e.g., 1000:1000)
        --name <NAME>           Container name
        --network <NET>         Network mode (e.g., host, bridge)
        --env KEY=VALUE         Environment variable
        ... any other docker run options

Examples:
    # Basic training with mounted data
    primus-cli container --mount /mnt/data -- train --config /mnt/data/exp.yaml

    # Use local Primus repo
    primus-cli container --primus-path ~/workspace/Primus -- train pretrain

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
PRIMUS_PATH="$(realpath -m "$RUNNER_DIR/..")"

# Load common library (required)
# shellcheck disable=SC1091
source "$RUNNER_DIR/lib/common.sh" || {
    echo "[ERROR] Failed to load common library: $RUNNER_DIR/lib/common.sh" >&2
    exit 1
}

# Load config library (required)
# shellcheck disable=SC1091
source "$RUNNER_DIR/lib/config.sh" || {
    LOG_ERROR "[container] Failed to load config library: $RUNNER_DIR/lib/config.sh"
    exit 1
}

HOSTNAME=$(hostname)

###############################################################################
# STEP 1: Pre-parse global options (--config, --debug, --dry-run)
###############################################################################
CONFIG_FILE=""
DEBUG_MODE=0
DRY_RUN_MODE=0
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
        *)
            PRE_PARSE_ARGS+=("$1")
            shift
            ;;
    esac
done
# Restore arguments
set -- "${PRE_PARSE_ARGS[@]}"

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
# STEP 3: Apply configuration values
# Priority: CLI args > Config file > Script defaults
###############################################################################
# 1. Image
if [[ -z "${DOCKER_IMAGE:-}" && -n "${container_config[image]:-}" ]]; then
    DOCKER_IMAGE="${container_config[image]}"
    LOG_INFO "[container] Using image from config: $DOCKER_IMAGE"
fi

# 2. Env vars (will be added to CONTAINER_OPTS)
CONTAINER_OPTS=()  # Container options (generic --key value pairs, stored as array to support repeatable options)
MOUNTS=()  # Mounts from CLI arguments (will be processed later)
for key in "${!container_config[@]}"; do
    if [[ "$key" =~ ^env\.[A-Za-z0-9_]+$ ]]; then
        env_name="${key#env.}"
        env_value="${container_config[$key]}"
        CONTAINER_OPTS+=(--env "${env_name}=${env_value}")
        LOG_DEBUG "[container] Added env var from config: ${env_name}=${env_value}"
    fi
done

# 4. Generic options (options.*)
for key in "${!container_config[@]}"; do
    if [[ "$key" =~ ^options\. ]]; then
        opt_name="${key#options.}"
        opt_value="${container_config[$key]}"
        if [[ "$opt_name" =~ ^devices\.[0-9]+$ ]]; then
            CONTAINER_OPTS+=( --device "${opt_value}" )
            LOG_DEBUG "[container] Added device from config: ${opt_value}"
        elif [[ "$opt_name" =~ ^capabilities\.[0-9]+$ ]]; then
            CONTAINER_OPTS+=( --cap-add "${opt_value}" )
            LOG_DEBUG "[container] Added capability from config: ${opt_value}"
        elif [[ "$opt_name" =~ ^mounts\.[0-9]+$ ]]; then
            CONTAINER_OPTS+=( --volume "${opt_value}" )
            LOG_DEBUG "[container] Added mount from config: ${opt_value}"
        else
            CONTAINER_OPTS+=( "--${opt_name}" "${opt_value}" )
            LOG_DEBUG "[container] Added option from config: --${opt_name} ${opt_value}"
        fi
    fi
done


# 5. Debug / dry-run
[[ "$DEBUG_MODE" == "0" && ("${container_config[debug]:-false}" == "true" || "${container_config[debug]:-false}" == "1") ]] && DEBUG_MODE=1
[[ "$DRY_RUN_MODE" == "0" && ("${container_config[dry_run]:-false}" == "true" || "${container_config[dry_run]:-false}" == "1") ]] && DRY_RUN_MODE=1

# Enable debug mode if set
if [[ "$DEBUG_MODE" == "1" ]]; then
    export PRIMUS_LOG_LEVEL="DEBUG"
    LOG_INFO "[container] Debug mode enabled (PRIMUS_LOG_LEVEL=DEBUG)"
fi

###############################################################################
# STEP 4: Parse CLI arguments
# Priority: CLI args override configuration values
###############################################################################

CLEAN_DOCKER_CONTAINER=0
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --image)
            DOCKER_IMAGE="$2"
            shift 2
            ;;
        --mount)
            MOUNTS+=("$2")
            shift 2
            ;;
        --clean)
            CLEAN_DOCKER_CONTAINER=1
            shift
            ;;
        --help|-h)
            print_usage
            exit 0
            ;;
        --)
            shift
            POSITIONAL_ARGS+=("$@")
            break
            ;;
        --*)
            # Generic docker option (--key value)
            opt_value="${2:-}"
            if [[ -n "$opt_value" ]] && [[ "$opt_value" != --* ]]; then
                CONTAINER_OPTS+=("$1" "$opt_value")
                shift 2
            else
                LOG_ERROR_RANK0 "[container] Unknown option: $1"
                exit 1
            fi
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

###############################################################################
# STEP 5: Build volume mounts
###############################################################################

# Defaults (fallback)
DOCKER_IMAGE=${DOCKER_IMAGE:-"docker.io/rocm/primus:v25.9_gfx942"}

# Mount the project root and additional directories into the container
VOLUME_ARGS=(-v "$PRIMUS_PATH:$PRIMUS_PATH")
for mnt in "${MOUNTS[@]}"; do
    # Parse --mount argument (HOST[:CONTAINER])
    if [[ "$mnt" == *:* ]]; then
        host_path="${mnt%%:*}"
        container_path="${mnt#*:}"
        # Check that the host path exists and is a directory
        if [[ ! -d "$host_path" ]]; then
            LOG_ERROR_RANK0 "[container] Invalid directory for --mount $mnt"
            exit 1
        fi
        VOLUME_ARGS+=(-v "$(realpath "$host_path")":"$container_path")
    else
        # Mount to same path inside container
        if [[ ! -d "$mnt" ]]; then
            LOG_ERROR_RANK0 "[container] Invalid directory for --mount $mnt"
            exit 1
        fi
        abs_path="$(realpath "$mnt")"
        VOLUME_ARGS+=(-v "$abs_path":"$abs_path")
    fi
done

###############################################################################
# STEP 6: Detect container runtime (docker/podman)
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
        LOG_ERROR_RANK0 "[container] Neither Docker nor Podman found!"
        exit 1
    fi
fi

###############################################################################
# STEP 7: Optional container cleanup
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
# STEP 8: Prepare launch arguments
###############################################################################

ARGS=("${POSITIONAL_ARGS[@]}")
OPTION_ARGS=("${CONTAINER_OPTS[@]}")

###############################################################################
# STEP 9: Display launch information
###############################################################################
LOG_INFO_RANK0 "[container] ========== Launch Info($DOCKER_CLI) =========="
LOG_INFO_RANK0 "[container]   IMAGE: $DOCKER_IMAGE"
LOG_INFO "[container]   HOSTNAME: $HOSTNAME"
LOG_INFO_RANK0 "[container]   VOLUME_ARGS:"
for ((i = 0; i < ${#VOLUME_ARGS[@]}; i+=2)); do
    LOG_INFO_RANK0 "[container]       ${VOLUME_ARGS[i]} ${VOLUME_ARGS[i+1]}"
done
if [[ ${#OPTION_ARGS[@]} -gt 0 ]]; then
    LOG_INFO_RANK0 "[container]   CONTAINER_OPTIONS:"
    i=0
    while [[ $i -lt ${#OPTION_ARGS[@]} ]]; do
        LOG_INFO_RANK0 "[container]       ${OPTION_ARGS[i]} ${OPTION_ARGS[i+1]}"
        ((i+=2))
    done
fi
LOG_INFO_RANK0 "[container]   LAUNCH ARGS:"
LOG_INFO_RANK0 "[container]     ${ARGS[*]}"

###############################################################################
# STEP 10: Launch container (or show dry-run)
###############################################################################
if [[ "$DRY_RUN_MODE" == "1" ]]; then
    LOG_INFO_RANK0 "[container] [DRY-RUN] Would execute:"
    LOG_INFO_RANK0 "[container]   ${DOCKER_CLI} run --rm \\"
    for ((i = 0; i < ${#VOLUME_ARGS[@]}; i+=2)); do
        LOG_INFO_RANK0 "[container]     ${VOLUME_ARGS[i]} ${VOLUME_ARGS[i+1]} \\"
    done
    i=0
    while [[ $i -lt ${#OPTION_ARGS[@]} ]]; do
        LOG_INFO_RANK0 "[container]     ${OPTION_ARGS[i]} ${OPTION_ARGS[i+1]} \\"
        ((i+=2))
    done
    LOG_INFO_RANK0 "[container]     $DOCKER_IMAGE /bin/bash -c \"...\""
    LOG_INFO_RANK0 "[container]   With args: ${ARGS[*]}"
    exit 0
fi

"${DOCKER_CLI}" run --rm \
    "${VOLUME_ARGS[@]}" \
    "${OPTION_ARGS[@]}" \
    "$DOCKER_IMAGE" /bin/bash -c "\
        echo '[container][INFO]: container started at $(date +"%Y.%m.%d %H:%M:%S")' && \
        [[ -d $PRIMUS_PATH ]] || { echo '[container][ERROR]: Primus not found at $PRIMUS_PATH' >&2; exit 42; } && \
        cd $PRIMUS_PATH && bash runner/primus-cli-direct.sh \"\$@\" 2>&1 && \
        echo '[container][INFO]: container finished at $(date +"%Y.%m.%d %H:%M:%S")'
    " bash "${ARGS[@]}"
