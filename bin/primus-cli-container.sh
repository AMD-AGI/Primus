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
    --image <DOCKER_IMAGE>      Docker image to use [default: \$DOCKER_IMAGE or rocm/megatron-lm:v25.5_py310]
    --mount <HOST[:CONTAINER]>  Mount a host directory into the container.
                                 - If only HOST is given, mounts to same path inside container.
                                 - If HOST:CONTAINER is given, mounts host directory to container path.
                                 (repeatable; for data, output, cache, etc.)
    --primus-path <HOST_PATH>   Use this Primus repo instead of the image default. The path will be mounted
                                into the container and installed in editable mode.
    --clean                     Remove all containers before launch
    --help                      Show this message and exit

Examples:
    primus-cli container --mount /mnt/data -- train --config /mnt/data/exp.yaml --data-path /mnt/data
    primus-cli container --mount /mnt/profile_out -- benchmark gemm --output /mnt/profile_out/result.txt

    # Mounts and installs your local Primus repo into the container.
    primus-cli container --primus-path ~/workspace/Primus -- train pretrain --config /data/exp.yaml
EOF
}

if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    print_usage
    exit 0
fi

HOSTNAME=$(hostname)

# Default Values
PRIMUS_PATH="/workspace/Primus"

# Parse CLI options
DOCKER_IMAGE=""
CLEAN_DOCKER_CONTAINER=0
MOUNTS=()
POSITIONAL_ARGS=()

VERBOSE=1

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
            # PRIMUS_PATH="$2"
            # MOUNTS+=("$2")
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
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

# Defaults (fallback)
DOCKER_IMAGE=${DOCKER_IMAGE:-"docker.io/rocm/megatron-lm:v25.5_py310"}

# ----------------- Volume Mounts -----------------
# Mount the project root and dataset directory into the container
VOLUME_ARGS=()
for mnt in "${MOUNTS[@]}"; do
    # Parse --mount argument (HOST[:CONTAINER])
    if [[ "$mnt" == *:* ]]; then
        host_path="${mnt%%:*}"
        container_path="${mnt#*:}"
        # Check that the host path exists and is a directory
        if [[ ! -d "$host_path" ]]; then
            echo "[primus-cli-container][${HOSTNAME}][ERROR] --mount $host_path does not exist or is not a directory. Please check your path." >&2
            exit 1
        fi
        VOLUME_ARGS+=(-v "$(realpath "$host_path")":"$container_path")
    else
        # Mount to same path inside container
        if [[ ! -d "$mnt" ]]; then
            echo "[primus-cli-container][${HOSTNAME}][ERROR] --mount $mnt does not exist or is not a directory. Please check your path." >&2
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
    echo "[primus-cli-container][${HOSTNAME}][ERROR] Neither Docker nor Podman found!" >&2
    exit 1
fi

if [[ "$CLEAN_DOCKER_CONTAINER" == "1" ]]; then
    echo "[primus-cli-container][${HOSTNAME}][INFO] Cleaning up existing containers..."
    CONTAINERS="$($DOCKER_CLI ps -aq)"
    if [[ -n "$CONTAINERS" ]]; then
        printf '%s\n' "$CONTAINERS" | xargs -r -n1 "$DOCKER_CLI" rm -f
        echo "[primus-cli-container][${HOSTNAME}][INFO] Removed containers: $CONTAINERS"
    else
        echo "[primus-cli-container][${HOSTNAME}][INFO] No containers to remove."
    fi
fi

ARGS=("${POSITIONAL_ARGS[@]}")

# ------------------ Print Info ------------------
if [[ "$VERBOSE" == "1" ]]; then
    echo "[prinus-cli-container][${HOSTNAME}][INFO] ========== Launch Info =========="
    echo "[prinus-cli-container][${HOSTNAME}][INFO]  HOSTNAME: $HOSTNAME"
    echo "[prinus-cli-container][${HOSTNAME}][INFO]  VOLUME_ARGS:"
    for ((i = 0; i < ${#VOLUME_ARGS[@]}; i+=2)); do
        echo "[prinus-cli-container][${HOSTNAME}][INFO]      ${VOLUME_ARGS[i]} ${VOLUME_ARGS[i+1]}"
    done
    echo "[prinus-cli-container][${HOSTNAME}][INFO]  LAUNCH ARGS:"
    echo "[prinus-cli-container][${HOSTNAME}][INFO]      ${ARGS[*]}"
    echo
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
    "$DOCKER_IMAGE" /bin/bash -c "\
        echo '[primus-cli-container][${HOSTNAME}][INFO]: container started at $(date +"%Y.%m.%d %H:%M:%S")' && \
        cd $PRIMUS_PATH && pip install -e . && bin/primus-cli-entrypoint.sh \"\$@\" 2>&1 && \
        echo '[primus-cli-container][${HOSTNAME}][INFO]: container finished at $(date +"%Y.%m.%d %H:%M:%S")'
    " bash "${ARGS[@]}"
