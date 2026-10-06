#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Primus Container Mode Launcher
#
# This script launches Primus workflows in a Docker/Podman or Apptainer/Singularity container.
#
# Execution Flow:
#   1. Parse global options (--config, --debug, --dry-run, --container-runtime)
#   2. Load configuration from YAML files
#   3. Extract and apply container.* configuration parameters
#   4. Parse CLI arguments (--image, --volume, generic docker options)
#   5. Build volume mounts and container options (translated for Apptainer)
#   6. Select docker/podman/apptainer/singularity CLI
#   7. Launch container with primus-cli-direct.sh inside
#
###############################################################################

set -euo pipefail

print_usage() {
cat <<EOF
Usage: bash primus-run-container.sh [OPTIONS] -- [SCRIPT_ARGS...]

Launch a Primus task (train / benchmark / preflight / etc.) in a Docker/Podman or Apptainer container.

Global Options:
    --config <FILE>             Load configuration from specified YAML file
    --debug                     Enable debug mode (verbose logging)
    --dry-run                   Show what would be executed without running
    --clean                     Remove all containers before launch
    --container-runtime <NAME>  docker | podman | apptainer | singularity
                                [default: container.runtime in config, else the first one found in that order]
    --help, -h                  Show this message and exit

Docker/Podman Options:
    All docker/podman run options are supported. Some key options have special handling:

    Cumulative Options (can be specified multiple times):
        --volume <HOST[:CONTAINER]>  Mount volumes. If only HOST given, mounts to same path.
        --env KEY=VALUE              Set environment variables
        --device <DEVICE_PATH>       Add host device access (e.g., /dev/kfd, /dev/dri)
        --cap-add <CAPABILITY>       Add Linux capabilities (e.g., SYS_PTRACE)

    Container Configuration:
        --image <DOCKER_IMAGE>       Docker image [default: container.options.image in runner/.primus.yaml]
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

Apptainer / Singularity:
    Runs 'apptainer exec --cleanenv <image> ...' as the calling user, with the options above translated:
        --volume becomes --bind (only the ro/rw modes are kept).
        --device /dev/... is dropped: Apptainer mounts the host /dev (its --device takes CDI names).
        --name, --ipc, --network/--net, --pid, --uts, --userns, --cgroupns, --privileged,
        --security-opt, --group-add, --cap-add/--cap-drop, --ulimit, --shm-size, --gpus, --user and
        --rm are dropped: Apptainer already shares the host network, IPC and PID namespaces,
        /dev/shm and resource limits.
        Any other option (e.g. --bind, --rocm, --writable-tmpfs, --fakeroot, --overlay) is passed through.
    --image takes a .sif file or sandbox directory; a plain image name is pulled as docker://<image>.
    Each run gets a private /tmp that is removed afterwards, unless a volume already targets /tmp.

Examples:
    # Basic training with mounted data
    primus-cli container --volume /mnt/data -- train --config /mnt/data/exp.yaml

    # Run with resource limits
    primus-cli container --cpus 16 --memory 128G --gpus 8 -- train pretrain

    # Run as specific user
    primus-cli container --user 1000:1000 -- benchmark gemm

    # Use configuration file
    primus-cli --config .primus.yaml container -- train

    # Run with Apptainer from a pre-built SIF (apptainer pull primus.sif docker://rocm/primus:v26.7)
    primus-cli container --container-runtime apptainer --image /shared/primus.sif -- train pretrain
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
CONTAINER_RUNTIME_CLI=""
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
        --container-runtime)
            if [[ -z "${2:-}" || "$2" == --* ]]; then
                LOG_ERROR "[container] --container-runtime requires a value (docker, podman, apptainer or singularity)"
                exit 2
            fi
            CONTAINER_RUNTIME_CLI="$2"
            shift 2
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

# Select container runtime. Priority: --container-runtime > container.runtime > auto-detect
CONTAINER_RUNTIME="${CONTAINER_RUNTIME_CLI:-${container_config[runtime]:-auto}}"
case "$CONTAINER_RUNTIME" in
    auto)
        for candidate in docker podman apptainer singularity; do
            if command -v "$candidate" >/dev/null 2>&1; then
                CONTAINER_RUNTIME="$candidate"
                break
            fi
        done
        if [[ "$CONTAINER_RUNTIME" == "auto" ]]; then
            # Mock runtime for dry-run testing
            CONTAINER_RUNTIME="docker"
            LOG_INFO_RANK0 "[container] Using mock container runtime for dry-run (no docker/podman/apptainer found)"
        fi
        ;;
    docker|podman|apptainer|singularity)
        if ! command -v "$CONTAINER_RUNTIME" >/dev/null 2>&1; then
            if [[ "$DRY_RUN_MODE" != "true" ]]; then
                LOG_ERROR "[container] Container runtime '$CONTAINER_RUNTIME' not found in PATH"
                exit 1
            fi
            LOG_INFO_RANK0 "[container] Container runtime '$CONTAINER_RUNTIME' not found; continuing for dry-run"
        fi
        ;;
    *)
        LOG_ERROR "[container] Unsupported container runtime: '$CONTAINER_RUNTIME' (expected docker, podman, apptainer or singularity)"
        exit 2
        ;;
esac
export CONTAINER_RUNTIME

# Apptainer and Singularity share one CLI dialect; Docker and Podman share the other.
if [[ "$CONTAINER_RUNTIME" == "apptainer" || "$CONTAINER_RUNTIME" == "singularity" ]]; then
    RUNTIME_IS_APPTAINER=true
else
    RUNTIME_IS_APPTAINER=false
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
    [[ "$env_key" =~ ^(PRIMUS_|NCCL_|RCCL_|GLOO_|IONIC_|HIPBLASLT_) ]] || continue
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
# STEP 5: Convert container_config to Docker/Podman or Apptainer options
# Now we have a complete container_config with CLI overrides applied
###############################################################################

LOG_INFO_RANK0 "[container] Converting configuration to container options..."

# Rewrite one container_config option (name and newline-separated values) in
# place for Apptainer. Returns non-zero when the option must not be passed.
apptainer_translate_option() {
    local -n _name="$1"
    local -n _value="$2"
    local entry src dst modes mode kept
    local -a mode_list out=()

    case "$_name" in
        env)
            # Forwarded as inner primus-cli --env arguments instead
            return 1
            ;;
        # Several of these exist in Apptainer as booleans (--ipc, --net, --pid,
        # --uts, --userns), so forwarding Docker's "<flag> host" form would make
        # Apptainer read "host" as the image.
        name|ipc|network|net|pid|uts|userns|cgroupns|privileged|security-opt|group-add|ulimit|shm-size|gpus|user|cap-add|cap-drop|rm)
            APPTAINER_DROPPED_OPTIONS+=("--$_name")
            return 1
            ;;
        volume)
            _name="bind"
            while IFS= read -r entry; do
                [[ -n "$entry" ]] || continue
                IFS=':' read -r src dst modes <<< "$entry"
                # --bind only understands ro/rw; drop Docker-only modes such as z, Z or cached
                kept=""
                IFS=',' read -ra mode_list <<< "$modes"
                for mode in "${mode_list[@]}"; do
                    if [[ "$mode" == "ro" || "$mode" == "rw" ]]; then
                        kept="${kept:+$kept,}$mode"
                    fi
                done
                if [[ -z "$dst" ]]; then
                    out+=("$src")
                elif [[ -z "$kept" ]]; then
                    out+=("$src:$dst")
                else
                    out+=("$src:$dst:$kept")
                fi
            done <<< "$_value"
            ;;
        device)
            # Apptainer mounts the host /dev already; its --device only takes CDI names
            while IFS= read -r entry; do
                [[ -n "$entry" ]] || continue
                if [[ "$entry" == /dev/* ]]; then
                    APPTAINER_DROPPED_OPTIONS+=("--device $entry")
                else
                    out+=("$entry")
                fi
            done <<< "$_value"
            ;;
        *)
            return 0
            ;;
    esac

    [[ ${#out[@]} -gt 0 ]] || return 1
    _value="$(printf '%s\n' "${out[@]}")"
}

# Print the container-side destination of a bind spec (src[:dst[:opts]]).
apptainer_bind_target() {
    local spec="$1" target="$1"
    if [[ "$spec" == *:* ]]; then
        target="${spec#*:}"
        target="${target%%:*}"
    fi
    echo "${target%/}"
}

# True when a configured volume/bind already targets /tmp inside the container.
# Native --bind values may hold a comma-separated list of specs.
apptainer_binds_tmp() {
    local entry spec
    local -a specs
    while IFS= read -r entry; do
        [[ -n "$entry" && "$entry" != "[]" ]] || continue
        IFS=',' read -ra specs <<< "$entry"
        for spec in "${specs[@]}"; do
            if [[ "$(apptainer_bind_target "$spec")" == "/tmp" ]]; then
                return 0
            fi
        done
    done <<< "${container_config[options.volume]:-}"$'\n'"${container_config[options.bind]:-}"
    return 1
}

# 1. Image (required, validated above)
# Allow users to override the image using the environment variable DOCKER_IMAGE.
if [ -z "${DOCKER_IMAGE:-}" ]; then
    # For single-value options like image, take the last value (CLI overrides config)
    DOCKER_IMAGE=$(echo "${container_config[options.image]}" | tail -n1)
fi
# Apptainer treats a bare "repo/image:tag" as a local path, so registry images
# need an explicit transport. Local .sif files, sandbox directories and refs
# that already name a transport are used as-is.
if [[ "$RUNTIME_IS_APPTAINER" == "true" \
      && ! "$DOCKER_IMAGE" =~ ^(docker|docker-daemon|docker-archive|oci|oci-archive|library|shub|oras|https?|instance): \
      && ! -e "$DOCKER_IMAGE" && "$DOCKER_IMAGE" != *.sif \
      && "$DOCKER_IMAGE" != /* && "$DOCKER_IMAGE" != ./* && "$DOCKER_IMAGE" != ../* ]]; then
    DOCKER_IMAGE="docker://$DOCKER_IMAGE"
    LOG_INFO_RANK0 "[container] Pulling $DOCKER_IMAGE at launch; for multi-node runs, build a SIF once with '$CONTAINER_RUNTIME pull primus.sif $DOCKER_IMAGE' on a shared filesystem and pass --image /path/to/primus.sif"
fi
LOG_INFO_RANK0 "[container] Final image: $DOCKER_IMAGE"

# 2. Build CONTAINER_OPTS from configuration
CONTAINER_OPTS=()
CONTAINER_TMPDIR=""

if [[ "$RUNTIME_IS_APPTAINER" == "true" ]]; then
    # Start from the image environment, as Docker does: host variables reach the
    # container only through the inner --env arguments built above.
    CONTAINER_OPTS+=("--cleanenv")

    # Apptainer shares the host /tmp, where hooks write fixed-name files (e.g.
    # /tmp/primus_patch_args.txt) that would collide between users and between
    # concurrent jobs on a node. Give each run its own /tmp, as Docker does.
    if ! apptainer_binds_tmp; then
        CONTAINER_TMPDIR="$(mktemp -d "${TMPDIR:-/tmp}/primus-container.XXXXXX")" || {
            LOG_ERROR "[container] Failed to create a private /tmp under ${TMPDIR:-/tmp}"
            exit 1
        }
        register_cleanup_hook "cleanup_temp $(printf '%q' "$CONTAINER_TMPDIR")"
        # common.sh skips its setup, including this trap, when a parent launcher
        # (primus-cli, slurm-entry) already sourced it and exported the guard.
        trap run_cleanup_hooks EXIT
        CONTAINER_OPTS+=("--bind" "$CONTAINER_TMPDIR:/tmp")
        LOG_INFO_RANK0 "[container] Added private /tmp: $CONTAINER_TMPDIR"
    fi

    CONTAINER_OPTS+=("--bind" "$PRIMUS_PATH:$PRIMUS_PATH")
    BIND_FLAG="--bind"
else
    # Always mount project root directory first
    CONTAINER_OPTS+=("-v" "$PRIMUS_PATH:$PRIMUS_PATH")
    BIND_FLAG="--volume"
fi
LOG_INFO_RANK0 "[container] Added project root volume: $PRIMUS_PATH"

# Cumulative options (all values used, config + CLI merge)
# Note: options.env is handled separately above and is NOT treated as a
# container-level --env; it becomes inner primus-cli --env arguments instead.
CUMULATIVE_OPTIONS=("device" "cap-add" "volume")
if [[ "$RUNTIME_IS_APPTAINER" == "true" ]]; then
    CUMULATIVE_OPTIONS+=("bind" "mount" "overlay")
fi
APPTAINER_DROPPED_OPTIONS=()

for key in "${!container_config[@]}"; do
    [[ "$key" =~ ^options\. ]] || continue

    opt_name="${key#options.}"
    opt_value="${container_config[$key]}"

    # Skip image (used separately) and empty array markers
    [[ "$opt_name" == "image" ]] && continue
    [[ "$opt_value" == "[]" ]] && continue

    if [[ "$RUNTIME_IS_APPTAINER" == "true" ]]; then
        apptainer_translate_option opt_name opt_value || continue
    fi

    # Make the container name unique per job so an orphaned container left by a
    # previous/cancelled job on a reused node doesn't cause a "name already in
    # use" conflict. SLURM_JOB_ID is per-job (same on all its nodes, unique across
    # jobs); fall back to the PID when not under Slurm.
    if [[ "$opt_name" == "name" && -n "$opt_value" && "$opt_value" != *$'\n'* ]]; then
        opt_value="${opt_value}-${SLURM_JOB_ID:-$$}"
    fi

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

if [[ ${#APPTAINER_DROPPED_OPTIONS[@]} -gt 0 ]]; then
    LOG_INFO_RANK0 "[container] Not passed to ${CONTAINER_RUNTIME} (it shares the host network, IPC, /dev and limits, and runs as the calling user): ${APPTAINER_DROPPED_OPTIONS[*]}"
fi

# The bnxt rebuild hook runs inside the container and checks for the tar on
# disk, so forwarding PATH_TO_BNXT_TAR_PACKAGE as an env var is not enough: the
# file has to be visible under the same path. Skip when the configured volumes
# already provide it.
if [[ -n "${PATH_TO_BNXT_TAR_PACKAGE:-}" && -f "${PATH_TO_BNXT_TAR_PACKAGE}" ]]; then
    bnxt_mount="${PATH_TO_BNXT_TAR_PACKAGE}:${PATH_TO_BNXT_TAR_PACKAGE}"
    bnxt_mounted=0
    for opt_value in "${CONTAINER_OPTS[@]}"; do
        if [[ "$opt_value" == "$bnxt_mount" || "$opt_value" == "${PATH_TO_BNXT_TAR_PACKAGE}" ]]; then
            bnxt_mounted=1
            break
        fi
    done
    if [[ $bnxt_mounted -eq 0 ]]; then
        CONTAINER_OPTS+=("$BIND_FLAG" "$bnxt_mount")
        LOG_INFO_RANK0 "[container] Added cumulative: $BIND_FLAG $bnxt_mount"
    fi
fi

# Apptainer cannot create a missing bind destination inside another bind, so
# targets under the private /tmp (e.g. a checkout in /tmp) must exist there.
if [[ -n "${CONTAINER_TMPDIR:-}" ]]; then
    for ((opt_i = 0; opt_i < ${#CONTAINER_OPTS[@]} - 1; opt_i++)); do
        [[ "${CONTAINER_OPTS[opt_i]}" == "--bind" ]] || continue
        IFS=',' read -ra bind_specs <<< "${CONTAINER_OPTS[opt_i + 1]}"
        for bind_spec in "${bind_specs[@]}"; do
            bind_target="$(apptainer_bind_target "$bind_spec")"
            [[ "$bind_target" == /tmp/?* ]] || continue
            bind_point="$CONTAINER_TMPDIR/${bind_target#/tmp/}"
            if [[ -d "${bind_spec%%:*}" ]]; then
                mkdir -p "$bind_point"
            else
                mkdir -p "$(dirname "$bind_point")" && touch "$bind_point"
            fi
            LOG_DEBUG_RANK0 "[container] Created bind point in private /tmp: $bind_target"
        done
    done
fi


###############################################################################
# STEP 6: Optional container cleanup
###############################################################################

if [[ "$CLEAN_DOCKER_CONTAINER" == "true" && "$RUNTIME_IS_APPTAINER" == "true" ]]; then
    LOG_INFO_RANK0 "[container] --clean: ${CONTAINER_RUNTIME} keeps no containers between runs; nothing to remove."
elif [[ "$CLEAN_DOCKER_CONTAINER" == "true" ]]; then
    LOG_INFO_RANK0 "[container] Cleaning up existing containers..."
    CONTAINERS="$($CONTAINER_RUNTIME ps -aq)"
    if [[ -n "$CONTAINERS" ]]; then
        # Tolerate per-container removal failures ("removal already in progress",
        # "No such container") so a stale/concurrently-removing container on a
        # shared node does not abort the whole run via xargs' exit code 123.
        printf '%s\n' "$CONTAINERS" | xargs -r -n1 -I{} sh -c "$CONTAINER_RUNTIME rm -f {} 2>/dev/null || true"
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
if [[ "$RUNTIME_IS_APPTAINER" == "true" ]]; then
    # `exec` runs the given command instead of the image runscript, and an
    # Apptainer container leaves nothing behind, so there is no --rm.
    CMD=(
        "${CONTAINER_RUNTIME}"
        exec
        "${OPTION_ARGS[@]}"
        "$DOCKER_IMAGE"
        /bin/bash
        -c
        "$CONTAINER_SCRIPT"
        bash
        "${ARGS[@]}"
    )
else
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
fi

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

# ---------------------------------------------------------------------------
# Optional: authenticate to the container registry before the implicit pull.
# On clusters whose compute nodes are not pre-logged-in, a private image
# (e.g. docker.io/tasimage/primus:*) cannot be pulled and `docker run` fails
# with "Unable to find image '<image>' locally" followed by a pull-access
# error. Export DOCKER_LOGIN_USER and DOCKER_LOGIN_KEY (e.g. a Docker Hub PAT)
# and the scheduler propagates them to every node (spur sbatch --export=ALL),
# so the login+pull happens automatically at launch. DOCKER_LOGIN_REGISTRY is
# optional and defaults to Docker Hub.
# Apptainer/Singularity read the same credentials from their
# <RUNTIME>_DOCKER_USERNAME/PASSWORD variables when pulling a docker:// image.
# ---------------------------------------------------------------------------
if [[ -n "${DOCKER_LOGIN_KEY:-}" ]]; then
    if [[ -z "${DOCKER_LOGIN_USER:-}" ]]; then
        LOG_ERROR "[container] DOCKER_LOGIN_KEY is set but DOCKER_LOGIN_USER is empty."
        exit 1
    fi
    if [[ "$RUNTIME_IS_APPTAINER" == "true" ]]; then
        # A "singularity" binary is either SingularityCE or Apptainer's compat
        # link, which read different variable prefixes.
        registry_env_prefixes=(APPTAINER)
        [[ "$CONTAINER_RUNTIME" == "singularity" ]] && registry_env_prefixes+=(SINGULARITY)
        for registry_env_prefix in "${registry_env_prefixes[@]}"; do
            export "${registry_env_prefix}_DOCKER_USERNAME=${DOCKER_LOGIN_USER}"
            export "${registry_env_prefix}_DOCKER_PASSWORD=${DOCKER_LOGIN_KEY}"
        done
        LOG_INFO_RANK0 "[container] Using registry credentials of ${DOCKER_LOGIN_USER} for ${CONTAINER_RUNTIME} image pulls"
    else
        LOG_INFO_RANK0 "[container] Logging in to registry ${DOCKER_LOGIN_REGISTRY:-docker.io} as ${DOCKER_LOGIN_USER}..."
        if ! printf '%s' "${DOCKER_LOGIN_KEY}" | \
            "$CONTAINER_RUNTIME" login ${DOCKER_LOGIN_REGISTRY:+"$DOCKER_LOGIN_REGISTRY"} -u "${DOCKER_LOGIN_USER}" --password-stdin; then
            LOG_ERROR "[container] Registry login failed for user ${DOCKER_LOGIN_USER}."
            exit 1
        fi
    fi
fi

LOG_INFO "[container] Executing command..."
"${CMD[@]}"
