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

Options:
    --image <DOCKER_IMAGE>      Docker image to use [default: \$DOCKER_IMAGE or rocm/primus:v25.9_gfx942]
    --mount <HOST[:CONTAINER]>  Mount a host directory into the container.
                                 - If only HOST is given, mounts to same path inside container.
                                 - If HOST:CONTAINER is given, mounts host directory to container path.
                                 (repeatable; for data, output, cache, etc.)
    --primus-path <HOST_PATH>   Use this Primus repo instead of the image default. The path will be mounted
                                into the container and installed in editable mode.
    --clean                     Remove all containers before launch

Resource Limits:
    --cpus <N>                  Limit CPU cores (e.g., 8, 16.5)
    --memory <SIZE>             Limit memory (e.g., 64G, 128G, 512M)
    --shm-size <SIZE>           Set shared memory size (e.g., 16G) [default: host IPC]
    --gpus <N>                  Limit GPU count (e.g., 4, 8) [default: all]

Other Options:
    --user <UID:GID>            Run as specific user (e.g., 1000:1000)
    --name <CONTAINER_NAME>     Set container name
    --help                      Show this message and exit

Examples:
    primus-cli container --mount /mnt/data -- train --config /mnt/data/exp.yaml --data-path /mnt/data
    primus-cli container --mount /mnt/profile_out -- benchmark gemm --output /mnt/profile_out/result.txt

    # Mounts and installs your local Primus repo into the container.
    primus-cli container --primus-path ~/workspace/Primus -- train pretrain --config /data/exp.yaml

    # Run with resource limits
    primus-cli container --cpus 16 --memory 128G --gpus 8 -- train pretrain --config exp.yaml

    # Run as specific user
    primus-cli container --user 1000:1000 -- benchmark gemm
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

# Parse CLI options first to get --config, --debug if present
CONFIG_FILE=""
DEBUG_MODE=0
PRE_PARSE_ARGS=()
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
        *)
            PRE_PARSE_ARGS+=("$1")
            shift
            ;;
    esac
done
# Restore arguments
set -- "${PRE_PARSE_ARGS[@]}"

# Load config library and mode-specific config
if [[ -f "$SCRIPT_DIR/lib/config.sh" ]]; then
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

# Resource limits (loaded from container section of config)
CONTAINER_CPUS="${CONTAINER_CPUS:-}"
CONTAINER_MEMORY="${CONTAINER_MEMORY:-}"
CONTAINER_SHM_SIZE="${CONTAINER_SHM_SIZE:-}"
CONTAINER_GPUS="${CONTAINER_GPUS:-}"
CONTAINER_USER="${CONTAINER_USER:-}"
CONTAINER_NAME="${CONTAINER_NAME:-}"

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
        --cpus)
            CONTAINER_CPUS="$2"
            shift 2
            ;;
        --memory)
            CONTAINER_MEMORY="$2"
            shift 2
            ;;
        --shm-size)
            CONTAINER_SHM_SIZE="$2"
            shift 2
            ;;
        --gpus)
            CONTAINER_GPUS="$2"
            shift 2
            ;;
        --user)
            CONTAINER_USER="$2"
            shift 2
            ;;
        --name)
            CONTAINER_NAME="$2"
            shift 2
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
    echo "$LOG_ERROR Neither Docker nor Podman found!" >&2
    exit 1
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

# ------------------ Build Resource Limit Arguments ------------------
RESOURCE_ARGS=()

# CPU limit
if [[ -n "$CONTAINER_CPUS" ]]; then
    RESOURCE_ARGS+=(--cpus "$CONTAINER_CPUS")
fi

# Memory limit
if [[ -n "$CONTAINER_MEMORY" ]]; then
    RESOURCE_ARGS+=(--memory "$CONTAINER_MEMORY")
fi

# Shared memory size
if [[ -n "$CONTAINER_SHM_SIZE" ]]; then
    RESOURCE_ARGS+=(--shm-size "$CONTAINER_SHM_SIZE")
fi

# GPU limit (for Docker with nvidia-docker or similar)
if [[ -n "$CONTAINER_GPUS" ]]; then
    # Note: This works with nvidia-docker or similar GPU runtime
    # For AMD GPUs, device visibility is controlled by HIP_VISIBLE_DEVICES
    if [[ "$DOCKER_CLI" == "docker" ]]; then
        RESOURCE_ARGS+=(--gpus "device=0-$((CONTAINER_GPUS-1))")
    fi
    # Export HIP_VISIBLE_DEVICES for AMD GPUs
    export HIP_VISIBLE_DEVICES=$(seq -s, 0 $((CONTAINER_GPUS-1)))
fi

# User specification
if [[ -n "$CONTAINER_USER" ]]; then
    RESOURCE_ARGS+=(--user "$CONTAINER_USER")
fi

# Container name
if [[ -n "$CONTAINER_NAME" ]]; then
    RESOURCE_ARGS+=(--name "$CONTAINER_NAME")
fi

# ------------------ Print Info ------------------
if [[ "$VERBOSE" == "1" ]]; then
    echo "$LOG_INFO ========== Launch Info($DOCKER_CLI) =========="
    echo "$LOG_INFO  IMAGE: $DOCKER_IMAGE"
    echo "$LOG_INFO  HOSTNAME: $HOSTNAME"
    echo "$LOG_INFO  VOLUME_ARGS:"
    for ((i = 0; i < ${#VOLUME_ARGS[@]}; i+=2)); do
        echo "$LOG_INFO      ${VOLUME_ARGS[i]} ${VOLUME_ARGS[i+1]}"
    done
    if [[ ${#RESOURCE_ARGS[@]} -gt 0 ]]; then
        echo "$LOG_INFO  RESOURCE_LIMITS:"
        [[ -n "$CONTAINER_CPUS" ]] && echo "$LOG_INFO      CPUs: $CONTAINER_CPUS"
        [[ -n "$CONTAINER_MEMORY" ]] && echo "$LOG_INFO      Memory: $CONTAINER_MEMORY"
        [[ -n "$CONTAINER_SHM_SIZE" ]] && echo "$LOG_INFO      SHM Size: $CONTAINER_SHM_SIZE"
        [[ -n "$CONTAINER_GPUS" ]] && echo "$LOG_INFO      GPUs: $CONTAINER_GPUS"
        [[ -n "$CONTAINER_USER" ]] && echo "$LOG_INFO      User: $CONTAINER_USER"
        [[ -n "$CONTAINER_NAME" ]] && echo "$LOG_INFO      Name: $CONTAINER_NAME"
    fi
    echo "$LOG_INFO  LAUNCH ARGS:"
    echo "$LOG_INFO    ${ARGS[*]}"
fi

# ------------------ Launch Training Container ------------------
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
    "${RESOURCE_ARGS[@]}" \
    "$DOCKER_IMAGE" /bin/bash -c "\
        echo '[primus-cli-container][${HOSTNAME}][INFO]: container started at $(date +"%Y.%m.%d %H:%M:%S")' && \
        [[ -d $PRIMUS_PATH ]] || { echo '$LOG_ERROR Primus not found at $PRIMUS_PATH'; exit 42; } && \
        cd $PRIMUS_PATH && bash runner/primus-cli-entrypoint.sh \"\$@\" 2>&1 && \
        echo '[primus-cli-container][${HOSTNAME}][INFO]: container finished at $(date +"%Y.%m.%d %H:%M:%S")'
    " bash "${ARGS[@]}"
