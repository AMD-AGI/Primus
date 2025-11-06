#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Configuration file loader for Primus CLI
# Supports: ~/.primusrc (shell format) and .primus.yaml (YAML format)
# Priority: CLI args > Project config > Global config > Defaults
#

# Requires common.sh to be sourced
if [[ -z "${__PRIMUS_COMMON_SOURCED:-}" ]]; then
    echo "[ERROR] config.sh requires common.sh to be sourced first" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Guard: avoid duplicate sourcing
# ---------------------------------------------------------------------------
if [[ -n "${__PRIMUS_CONFIG_SOURCED:-}" ]]; then
  return 0
fi
export __PRIMUS_CONFIG_SOURCED=1

# ---------------------------------------------------------------------------
# Configuration Variables
# ---------------------------------------------------------------------------
declare -A PRIMUS_CONFIG

# Cache directory for parsed configurations
PRIMUS_CONFIG_CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/primus-cli"
PRIMUS_CONFIG_CACHE_TTL=3600  # Cache validity in seconds (1 hour)

# Default configuration values
# Global defaults
PRIMUS_CONFIG[global.dry_run]="${PRIMUS_DRY_RUN:-0}"
PRIMUS_CONFIG[global.debug]="${DEBUG_MODE:-0}"

# Slurm defaults
PRIMUS_CONFIG[slurm.partition]="${SLURM_PARTITION:-}"
PRIMUS_CONFIG[slurm.nodes]="${SLURM_NODES:-}"
PRIMUS_CONFIG[slurm.gpus_per_node]="${GPUS_PER_NODE:-8}"

# Container defaults
PRIMUS_CONFIG[container.image]="${DOCKER_IMAGE:-rocm/primus:v25.9_gfx942}"
PRIMUS_CONFIG[container.cpus]="${CONTAINER_CPUS:-}"
PRIMUS_CONFIG[container.memory]="${CONTAINER_MEMORY:-}"
PRIMUS_CONFIG[container.shm_size]="${CONTAINER_SHM_SIZE:-}"
PRIMUS_CONFIG[container.gpus]="${CONTAINER_GPUS:-}"
PRIMUS_CONFIG[container.user]="${CONTAINER_USER:-}"

# Direct mode defaults
PRIMUS_CONFIG[direct.gpus_per_node]="${GPUS_PER_NODE:-8}"
PRIMUS_CONFIG[direct.master_port]="${MASTER_PORT:-1234}"
PRIMUS_CONFIG[direct.nnodes]="${NNODES:-1}"
PRIMUS_CONFIG[direct.master_addr]="${MASTER_ADDR:-localhost}"

# Path defaults
PRIMUS_CONFIG[paths.log_path]="${LOG_PATH:-logs}"

# ---------------------------------------------------------------------------
# Check if cached config is valid
# ---------------------------------------------------------------------------
is_cache_valid() {
    local config_file="$1"
    local cache_file="$2"

    # Cache doesn't exist
    [[ ! -f "$cache_file" ]] && return 1

    # Source file is newer than cache
    [[ "$config_file" -nt "$cache_file" ]] && return 1

    # Cache is older than TTL
    local cache_age=$(($(date +%s) - $(stat -c %Y "$cache_file" 2>/dev/null || echo 0)))
    [[ $cache_age -gt $PRIMUS_CONFIG_CACHE_TTL ]] && return 1

    return 0
}

# ---------------------------------------------------------------------------
# Load cached config
# ---------------------------------------------------------------------------
load_cache() {
    local cache_file="$1"

    if [[ -f "$cache_file" ]]; then
        # shellcheck disable=SC1090  # Dynamic source path
        source "$cache_file" 2>/dev/null && return 0
    fi
    return 1
}

# ---------------------------------------------------------------------------
# Save config to cache
# ---------------------------------------------------------------------------
save_cache() {
    local cache_file="$1"

    mkdir -p "$(dirname "$cache_file")" 2>/dev/null || return 1

    # Export current config to cache file
    {
        echo "# Primus CLI config cache"
        echo "# Generated: $(date)"
        echo ""
        for key in "${!PRIMUS_CONFIG[@]}"; do
            echo "PRIMUS_CONFIG[$key]=\"${PRIMUS_CONFIG[$key]}\""
        done
    } > "$cache_file" 2>/dev/null

    LOG_DEBUG "Config cached to: $cache_file"
}

# ---------------------------------------------------------------------------
# Load Shell-format Config File (.primusrc)
# ---------------------------------------------------------------------------
load_shell_config() {
    local config_file="$1"

    if [[ ! -f "$config_file" ]]; then
        return 1
    fi

    # Check cache first
    local cache_file
    cache_file="$PRIMUS_CONFIG_CACHE_DIR/$(basename "$config_file").cache"
    if is_cache_valid "$config_file" "$cache_file"; then
        LOG_DEBUG "Loading cached config: $config_file"
        load_cache "$cache_file" && return 0
    fi

    LOG_DEBUG "Loading shell config: $config_file"

    # Source the config file in a subshell to avoid pollution
    while IFS='=' read -r key value || [[ -n "$key" ]]; do
        # Skip comments and empty lines
        [[ "$key" =~ ^[[:space:]]*# ]] && continue
        [[ -z "${key// /}" ]] && continue

        # Remove leading/trailing whitespace
        key=$(echo "$key" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        value=$(echo "$value" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')

        # Remove quotes
        value="${value%\"}"
        value="${value#\"}"
        value="${value%\'}"
        value="${value#\'}"

        # Convert PRIMUS_CONTAINER_IMAGE to container.image
        if [[ "$key" == PRIMUS_* ]]; then
            # Extract the config path
            config_key="${key#PRIMUS_}"
            config_key="${config_key,,}"  # lowercase
            config_key="${config_key//_/.}"  # replace _ with .

            PRIMUS_CONFIG[$config_key]="$value"
            LOG_DEBUG "  $config_key = $value"
        fi
    done < "$config_file"

    # Save to cache
    save_cache "$cache_file"

    return 0
}

# ---------------------------------------------------------------------------
# Load YAML Config File (.primus.yaml)
# ---------------------------------------------------------------------------
load_yaml_config() {
    local config_file="$1"

    if [[ ! -f "$config_file" ]]; then
        return 1
    fi

    LOG_DEBUG "Loading YAML config: $config_file"

    # Simple YAML parser (handles basic key: value format, arrays, and nested sections)
    local current_section=""
    local current_subsection=""
    local current_array_key=""
    local array_index=0

    while IFS= read -r line || [[ -n "$line" ]]; do
        # Skip comments and empty lines
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        [[ -z "${line// /}" ]] && continue

        # Detect top-level section (e.g., "container:")
        if [[ "$line" =~ ^([a-z_]+):[[:space:]]*$ ]]; then
            current_section="${BASH_REMATCH[1]}"
            current_subsection=""
            current_array_key=""
            array_index=0
            continue
        fi

        # Detect subsection with 2 spaces (e.g., "  options:")
        if [[ "$line" =~ ^[[:space:]]{2}([a-z_-]+):[[:space:]]*$ ]]; then
            current_subsection="${BASH_REMATCH[1]}"
            current_array_key=""
            array_index=0
            continue
        fi

        # Parse array item (e.g., "    - /data:/data")
        if [[ "$line" =~ ^[[:space:]]+\-[[:space:]]+(.+)$ ]]; then
            local value="${BASH_REMATCH[1]}"

            # Remove quotes
            value="${value%\"}"
            value="${value#\"}"
            value="${value%\'}"
            value="${value#\'}"

            if [[ -n "$current_section" ]]; then
                if [[ -n "$current_subsection" ]]; then
                    local config_key="${current_section}.${current_subsection}.${array_index}"
                    PRIMUS_CONFIG[$config_key]="$value"
                    LOG_DEBUG "  $config_key = $value"
                    ((array_index++))
                elif [[ -n "$current_array_key" ]]; then
                    local config_key="${current_section}.${current_array_key}.${array_index}"
                    PRIMUS_CONFIG[$config_key]="$value"
                    LOG_DEBUG "  $config_key = $value"
                    ((array_index++))
                fi
            fi
            continue
        fi

        # Parse nested key-value with 4 spaces (e.g., "    cpus: 16")
        if [[ "$line" =~ ^[[:space:]]{4}([a-z_-]+):[[:space:]]*(.+)$ ]]; then
            local key="${BASH_REMATCH[1]}"
            local value="${BASH_REMATCH[2]}"

            # Remove quotes
            value="${value%\"}"
            value="${value#\"}"
            value="${value%\'}"
            value="${value#\'}"

            if [[ -n "$current_section" ]] && [[ -n "$current_subsection" ]]; then
                local config_key="${current_section}.${current_subsection}.${key}"
                PRIMUS_CONFIG[$config_key]="$value"
                LOG_DEBUG "  $config_key = $value"
            fi
            continue
        fi

        # Parse key-value with 2 spaces (e.g., "  image: value")
        if [[ "$line" =~ ^[[:space:]]{2}([a-z_-]+):[[:space:]]*(.+)$ ]]; then
            local key="${BASH_REMATCH[1]}"
            local value="${BASH_REMATCH[2]}"

            # Remove quotes
            value="${value%\"}"
            value="${value#\"}"
            value="${value%\'}"
            value="${value#\'}"

            # Reset subsection when we get a normal key-value pair
            current_subsection=""
            current_array_key=""
            array_index=0

            if [[ -n "$current_section" ]]; then
                local config_key="${current_section}.${key}"
                PRIMUS_CONFIG[$config_key]="$value"
                LOG_DEBUG "  $config_key = $value"
            fi
        fi
    done < "$config_file"

    return 0
}

# ---------------------------------------------------------------------------
# Load All Config Files (with priority)
# ---------------------------------------------------------------------------
load_config() {
    LOG_DEBUG "Loading configuration files..."

    # 1. Load global config (~/.primusrc)
    local global_config="$HOME/.primusrc"
    if [[ -f "$global_config" ]]; then
        LOG_INFO_RANK0 "Loading global config: $global_config"
        load_shell_config "$global_config" || LOG_WARN "Failed to load global config"
    fi

    # 2. Load project config (.primus.yaml in current directory)
    local project_config=".primus.yaml"
    if [[ -f "$project_config" ]]; then
        LOG_INFO_RANK0 "Loading project config: $project_config"
        load_yaml_config "$project_config" || LOG_WARN "Failed to load project config"
    fi

    # 3. Alternative: .primusrc in current directory
    local local_config=".primusrc"
    if [[ -f "$local_config" ]]; then
        LOG_INFO_RANK0 "Loading local config: $local_config"
        load_shell_config "$local_config" || LOG_WARN "Failed to load local config"
    fi

    LOG_DEBUG "Configuration loading complete"
}

# ---------------------------------------------------------------------------
# Get Config Value
# ---------------------------------------------------------------------------
get_config() {
    local key="$1"
    local default="${2:-}"

    if [[ -n "${PRIMUS_CONFIG[$key]:-}" ]]; then
        echo "${PRIMUS_CONFIG[$key]}"
    else
        echo "$default"
    fi
}

# ---------------------------------------------------------------------------
# Set Config Value (override)
# ---------------------------------------------------------------------------
set_config() {
    local key="$1"
    local value="$2"

    PRIMUS_CONFIG[$key]="$value"
    LOG_DEBUG "Config set: $key = $value"
}

# ---------------------------------------------------------------------------
# Apply Global Config to Environment Variables
# ---------------------------------------------------------------------------
apply_global_config() {
    LOG_DEBUG "Applying global configuration..."

    # Only export global settings that affect all modes
    if [[ -n "${PRIMUS_CONFIG[global.debug]:-}" ]]; then
        export PRIMUS_DEBUG="${PRIMUS_CONFIG[global.debug]}"
    fi

    if [[ -n "${PRIMUS_CONFIG[global.dry_run]:-}" ]]; then
        export PRIMUS_DRY_RUN="${PRIMUS_CONFIG[global.dry_run]}"
    fi

    LOG_DEBUG "Global configuration applied"
}

# ---------------------------------------------------------------------------
# Load Mode-Specific Config (called by each mode script)
# ---------------------------------------------------------------------------
load_mode_config() {
    local mode="$1"  # slurm, container, direct

    LOG_DEBUG "Loading config for mode: $mode"

    case "$mode" in
        slurm)
            # Export slurm-specific settings
            [[ -n "${PRIMUS_CONFIG[slurm.partition]:-}" ]] && export SLURM_PARTITION="${PRIMUS_CONFIG[slurm.partition]}"
            [[ -n "${PRIMUS_CONFIG[slurm.nodes]:-}" ]] && export SLURM_NODES="${PRIMUS_CONFIG[slurm.nodes]}"
            [[ -n "${PRIMUS_CONFIG[slurm.gpus_per_node]:-}" ]] && export GPUS_PER_NODE="${PRIMUS_CONFIG[slurm.gpus_per_node]}"
            ;;
        container)
            # Export container image
            [[ -n "${PRIMUS_CONFIG[container.image]:-}" ]] && export DOCKER_IMAGE="${PRIMUS_CONFIG[container.image]}"

            # Export generic container options as CONTAINER_OPTIONS
            # Format: key1=value1|key2=value2|...
            for config_key in "${!PRIMUS_CONFIG[@]}"; do
                if [[ "$config_key" =~ ^container\.options\.(.+)$ ]]; then
                    local opt_key="${BASH_REMATCH[1]}"
                    local opt_value="${PRIMUS_CONFIG[$config_key]}"
                    if [[ -z "${CONTAINER_OPTIONS:-}" ]]; then
                        export CONTAINER_OPTIONS="${opt_key}=${opt_value}"
                    else
                        export CONTAINER_OPTIONS="$CONTAINER_OPTIONS|${opt_key}=${opt_value}"
                    fi
                fi
            done

            # Handle mounts array (stored as container.mounts.0, container.mounts.1, ...)
            local mount_idx=0
            while [[ -n "${PRIMUS_CONFIG[container.mounts.$mount_idx]:-}" ]]; do
                if [[ -z "${CONTAINER_MOUNTS:-}" ]]; then
                    export CONTAINER_MOUNTS="${PRIMUS_CONFIG[container.mounts.$mount_idx]}"
                else
                    export CONTAINER_MOUNTS="$CONTAINER_MOUNTS|${PRIMUS_CONFIG[container.mounts.$mount_idx]}"
                fi
                ((mount_idx++))
            done
            ;;
        direct)
            # Export direct-mode-specific settings
            [[ -n "${PRIMUS_CONFIG[direct.gpus_per_node]:-}" ]] && export GPUS_PER_NODE="${PRIMUS_CONFIG[direct.gpus_per_node]}"
            [[ -n "${PRIMUS_CONFIG[direct.master_port]:-}" ]] && export MASTER_PORT="${PRIMUS_CONFIG[direct.master_port]}"
            [[ -n "${PRIMUS_CONFIG[direct.nnodes]:-}" ]] && export NNODES="${PRIMUS_CONFIG[direct.nnodes]}"
            [[ -n "${PRIMUS_CONFIG[direct.master_addr]:-}" ]] && export MASTER_ADDR="${PRIMUS_CONFIG[direct.master_addr]}"
            ;;
    esac

    LOG_DEBUG "Mode config loaded for: $mode"
}

# ---------------------------------------------------------------------------
# Extract Config Section
# Extract all config keys matching a prefix and remove the prefix
# Usage: extract_config_section "slurm" result_array
# ---------------------------------------------------------------------------
extract_config_section() {
    local prefix="$1"
    # shellcheck disable=SC2034  # result_array is used via nameref
    local -n result_array="$2"  # nameref to associative array

    LOG_DEBUG "Extracting config section: $prefix.*"

    local count=0
    # Extract all config keys matching prefix and remove the prefix
    for key in "${!PRIMUS_CONFIG[@]}"; do
        if [[ "$key" =~ ^${prefix}\. ]]; then
            # Remove prefix to get parameter name (e.g., "slurm.partition" -> "partition")
            local param_name="${key#"${prefix}".}"
            # shellcheck disable=SC2034  # result_array is a nameref, accessed indirectly
            result_array["$param_name"]="${PRIMUS_CONFIG[$key]}"
            LOG_DEBUG "  $param_name = ${PRIMUS_CONFIG[$key]}"
            ((count++))
        fi
    done

    LOG_DEBUG "Extracted $count parameters from $prefix section"
    return 0
}

# ---------------------------------------------------------------------------
# Export all functions
# ---------------------------------------------------------------------------
export -f load_shell_config load_yaml_config load_config
export -f get_config set_config apply_global_config load_mode_config
export -f extract_config_section

# Backward compatibility: apply_config calls apply_global_config
apply_config() {
    apply_global_config
}
export -f apply_config

LOG_DEBUG_RANK0 "Primus config library loaded successfully"
