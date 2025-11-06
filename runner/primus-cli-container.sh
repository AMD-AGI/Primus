#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

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

HOSTNAME=$(hostname)

# Derive PRIMUS_PATH as the parent directory of this script by default. This
# makes the container script usable when invoked from the repo's runner/ folder.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRIMUS_PATH="$(realpath -m "$SCRIPT_DIR/..")"

# Parse CLI options first to get --config, --debug, --dry-run if present
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

# Load common library first (required by config.sh)
if [[ -f "$SCRIPT_DIR/lib/common.sh" ]]; then
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/lib/common.sh" 2>/dev/null || true
fi

# Load config library and mode-specific config
if [[ -f "$SCRIPT_DIR/lib/config.sh" ]]; then
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/lib/config.sh" 2>/dev/null || true
    # If config file is provided via --config, load container-specific config
    if [[ -n "$CONFIG_FILE" ]] && [[ -f "$CONFIG_FILE" ]]; then
        load_yaml_config "$CONFIG_FILE" 2>/dev/null || true
        load_mode_config "container" 2>/dev/null || true
    fi
fi

# Parse CLI options
# NOTE: These variables use environment variables from mode-specific config
# Priority: CLI args > Mode config > Script defaults
DOCKER_IMAGE="${DOCKER_IMAGE:-}"
CLEAN_DOCKER_CONTAINER=0
MOUNTS=()
POSITIONAL_ARGS=()

# Container options (generic key-value pairs)
# Format: key1=value1|key2=value2|...
declare -A CONTAINER_OPTS

# Load options from config (CONTAINER_OPTIONS env var)
if [[ -n "${CONTAINER_OPTIONS:-}" ]]; then
    IFS='|' read -ra OPTS <<< "$CONTAINER_OPTIONS"
    for opt in "${OPTS[@]}"; do
        if [[ "$opt" == *=* ]]; then
            key="${opt%%=*}"
            value="${opt#*=}"
            CONTAINER_OPTS[$key]="$value"
        fi
    done
fi

# Load mounts from config (if set via CONTAINER_MOUNTS env var)
# Format: mount1|mount2|mount3
if [[ -n "${CONTAINER_MOUNTS:-}" ]]; then
    IFS='|' read -ra CONFIG_MOUNTS <<< "$CONTAINER_MOUNTS"
    MOUNTS+=("${CONFIG_MOUNTS[@]}")
fi

VERBOSE=1

LOG_ERROR="[primus-cli-container][${HOSTNAME}][ERROR]"
LOG_INFO="[primus-cli-container][${HOSTNAME}][INFO]"

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
        --primus-path)
            raw_path="$2"
            full_path="$(realpath -m "$raw_path")"
            PRIMUS_PATH="$full_path"
            MOUNTS+=("$full_path")
            shift 2
            ;;
        --clean)
            CLEAN_DOCKER_CONTAINER=1
            shift
            ;;
        --no-verbose)
            VERBOSE=0
            shift
            ;;
        --verbose)
            VERBOSE=1
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
            opt_key="${1#--}"
            opt_value="${2:-}"
            if [[ -n "$opt_value" ]] && [[ "$opt_value" != --* ]]; then
                CONTAINER_OPTS[$opt_key]="$opt_value"
                shift 2
            else
                echo "$LOG_ERROR Unknown option: $1" >&2
                exit 1
            fi
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

# Defaults (fallback)
DOCKER_IMAGE=${DOCKER_IMAGE:-"docker.io/rocm/primus:v25.9_gfx942"}

# ----------------- Volume Mounts -----------------
# Mount the project root and dataset directory into the container
VOLUME_ARGS=(-v "$PRIMUS_PATH:$PRIMUS_PATH")
for mnt in "${MOUNTS[@]}"; do
    # Parse --mount argument (HOST[:CONTAINER])
    if [[ "$mnt" == *:* ]]; then
        host_path="${mnt%%:*}"
        container_path="${mnt#*:}"
        # Check that the host path exists and is a directory
        if [[ ! -d "$host_path" ]]; then
            echo "$LOG_ERROR  invalid directory for --mount $mnt" >&2
            exit 1
        fi
        VOLUME_ARGS+=(-v "$(realpath "$host_path")":"$container_path")
    else
        # Mount to same path inside container
        if [[ ! -d "$mnt" ]]; then
            echo "$LOG_ERROR  invalid directory for --mount $mnt" >&2
            exit 1
        fi
        abs_path="$(realpath "$mnt")"
        VOLUME_ARGS+=(-v "$abs_path":"$abs_path")
    fi
done

# ------------------ Optional Container Cleanup ------------------
if command -v podman >/dev/null 2>&1; then
    DOCKER_CLI="podman"
elif command -v docker >/dev/null 2>&1; then
    DOCKER_CLI="docker"
else
    # In dry-run mode, we don't need actual docker/podman
    if [[ "$DRY_RUN_MODE" == "1" ]]; then
        DOCKER_CLI="docker"  # Use docker for dry-run output
    else
        echo "$LOG_ERROR Neither Docker nor Podman found!" >&2
        exit 1
    fi
fi

if [[ "$CLEAN_DOCKER_CONTAINER" == "1" ]]; then
    echo "$LOG_INFO Cleaning up existing containers..."
    CONTAINERS="$($DOCKER_CLI ps -aq)"
    if [[ -n "$CONTAINERS" ]]; then
        printf '%s\n' "$CONTAINERS" | xargs -r -n1 "$DOCKER_CLI" rm -f
        echo "$LOG_INFO Removed containers: $CONTAINERS"
    else
        echo "$LOG_INFO No containers to remove."
    fi
fi

ARGS=("${POSITIONAL_ARGS[@]}")

# ------------------ Build Container Option Arguments ------------------
# Convert CONTAINER_OPTS associative array to docker arguments
OPTION_ARGS=()
for opt_key in "${!CONTAINER_OPTS[@]}"; do
    opt_value="${CONTAINER_OPTS[$opt_key]}"
    OPTION_ARGS+=(--"$opt_key" "$opt_value")
done

# ------------------ Print Info ------------------
if [[ "$VERBOSE" == "1" ]]; then
    echo "$LOG_INFO ========== Launch Info($DOCKER_CLI) =========="
    echo "$LOG_INFO  IMAGE: $DOCKER_IMAGE"
    echo "$LOG_INFO  HOSTNAME: $HOSTNAME"
    echo "$LOG_INFO  VOLUME_ARGS:"
    for ((i = 0; i < ${#VOLUME_ARGS[@]}; i+=2)); do
        echo "$LOG_INFO      ${VOLUME_ARGS[i]} ${VOLUME_ARGS[i+1]}"
    done
    if [[ ${#OPTION_ARGS[@]} -gt 0 ]]; then
        echo "$LOG_INFO  CONTAINER_OPTIONS:"
        for ((i = 0; i < ${#OPTION_ARGS[@]}; i+=2)); do
            echo "$LOG_INFO      ${OPTION_ARGS[i]} ${OPTION_ARGS[i+1]}"
        done
    fi
    echo "$LOG_INFO  LAUNCH ARGS:"
    echo "$LOG_INFO    ${ARGS[*]}"
fi

# ------------------ Launch Training Container ------------------
# Handle dry-run mode
if [[ "$DRY_RUN_MODE" == "1" ]]; then
    echo "$LOG_INFO [DRY-RUN] Would execute:"
    echo "$LOG_INFO   ${DOCKER_CLI} run --rm \\"
    echo "$LOG_INFO     --ipc=host \\"
    echo "$LOG_INFO     --network=host \\"
    echo "$LOG_INFO     --device=/dev/kfd \\"
    echo "$LOG_INFO     --device=/dev/dri \\"
    echo "$LOG_INFO     --cap-add=SYS_PTRACE \\"
    echo "$LOG_INFO     --cap-add=CAP_SYS_ADMIN \\"
    echo "$LOG_INFO     --security-opt seccomp=unconfined \\"
    echo "$LOG_INFO     --group-add video \\"
    echo "$LOG_INFO     --privileged \\"
    echo "$LOG_INFO     --device=/dev/infiniband \\"
    for ((i = 0; i < ${#VOLUME_ARGS[@]}; i+=2)); do
        echo "$LOG_INFO     ${VOLUME_ARGS[i]} ${VOLUME_ARGS[i+1]} \\"
    done
    for ((i = 0; i < ${#OPTION_ARGS[@]}; i+=2)); do
        echo "$LOG_INFO     ${OPTION_ARGS[i]} ${OPTION_ARGS[i+1]} \\"
    done
    echo "$LOG_INFO     $DOCKER_IMAGE /bin/bash -c \"...\""
    echo "$LOG_INFO   With args: ${ARGS[*]}"
    exit 0
fi

"${DOCKER_CLI}" run --rm \
    --ipc=host \
    --network=host \
    --device=/dev/kfd \
    --device=/dev/dri \
    --cap-add=SYS_PTRACE \
    --cap-add=CAP_SYS_ADMIN \
    --security-opt seccomp=unconfined \
    --group-add video \
    --privileged \
    --device=/dev/infiniband \
    "${VOLUME_ARGS[@]}" \
    "${OPTION_ARGS[@]}" \
    "$DOCKER_IMAGE" /bin/bash -c "\
        echo '[primus-cli-container][${HOSTNAME}][INFO]: container started at $(date +"%Y.%m.%d %H:%M:%S")' && \
        [[ -d $PRIMUS_PATH ]] || { echo '$LOG_ERROR Primus not found at $PRIMUS_PATH'; exit 42; } && \
        cd $PRIMUS_PATH && bash runner/primus-cli-entrypoint.sh \"\$@\" 2>&1 && \
        echo '[primus-cli-container][${HOSTNAME}][INFO]: container finished at $(date +"%Y.%m.%d %H:%M:%S")'
    " bash "${ARGS[@]}"
