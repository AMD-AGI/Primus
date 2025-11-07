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
# Configuration Variables
# ---------------------------------------------------------------------------
# Always ensure associative array exists
if ! declare -p PRIMUS_CONFIG &>/dev/null; then
    declare -A PRIMUS_CONFIG
fi

PRIMUS_ROOT_DIR="${PRIMUS_ROOT_DIR:-$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/../../" && pwd)}"
PRIMUS_RUNNER_DIR="${PRIMUS_RUNNER_DIR:-${PRIMUS_ROOT_DIR}/runner}"


# ---------------------------------------------------------------------------
# Guard: avoid duplicate sourcing
# ---------------------------------------------------------------------------
if [[ -n "${__PRIMUS_CONFIG_SOURCED:-}" ]]; then
  return 0
fi
export __PRIMUS_CONFIG_SOURCED=1

# Cache configuration
PRIMUS_CONFIG_CACHE_TTL=3600  # Cache validity in seconds (1 hour)

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
# Load YAML Config File (.primus.yaml)
# ---------------------------------------------------------------------------
load_yaml_config() {
    local config_file="$1"

    LOG_DEBUG_RANK0 "Loading YAML config: $config_file"

    if [[ ! -f "$config_file" ]]; then
        LOG_ERROR_RANK0 "YAML config file not found: $config_file"
        return 1
    fi

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
                    LOG_DEBUG_RANK0 "  $config_key = $value"
                    ((array_index++))
                elif [[ -n "$current_array_key" ]]; then
                    local config_key="${current_section}.${current_array_key}.${array_index}"
                    PRIMUS_CONFIG[$config_key]="$value"
                    LOG_DEBUG_RANK0 "  $config_key = $value"
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
                LOG_DEBUG_RANK0 "  $config_key = $value"
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
                LOG_DEBUG_RANK0 "  $config_key = $value"
            fi
        fi
    done < "$config_file"

    LOG_DEBUG_RANK0 "Loaded YAML config: $config_file"

    return 0
}

# ---------------------------------------------------------------------------
# Load All Config Files (with priority)
# Priority: CLI args > System defaults (runner/.primus.yaml) > User config (~/.primus.yaml)
# Note: Later loads override earlier ones
# ---------------------------------------------------------------------------
load_config() {
    LOG_INFO_RANK0 "  Loading configuration files..."

    # 1. Load user global config (~/.primus.yaml) first - lowest priority
    local global_config="$HOME/.primus.yaml"
    if [[ -f "$global_config" ]]; then
        LOG_INFO_RANK0 "  Loading user config: $global_config"
        load_yaml_config "$global_config" || LOG_ERROR "Failed to load user config"
    fi

    # 2. Load system default config (runner/.primus.yaml) last - highest priority (overrides user config)
    local system_config="${PRIMUS_RUNNER_DIR}/.primus.yaml"
    if [[ -f "$system_config" ]]; then
        LOG_INFO_RANK0 "  Loading system default config: $system_config"
        load_yaml_config "$system_config" ||  {
            LOG_ERROR "Failed to load system default config"
            exit 1
        }
    fi

    LOG_INFO_RANK0 "  Configuration loading complete"
}

# ---------------------------------------------------------------------------
# Load Config Automatically (with CLI override support)
# Usage: load_config_auto [config_file] [log_prefix]
#
# If config_file is provided and non-empty:
#   - Load the specified config file (must succeed)
# Otherwise:
#   - Load default configuration files via load_config()
# ---------------------------------------------------------------------------
load_config_auto() {
    local config_file="${1:-}"
    local log_prefix="${2:-main}"

    if [[ -n "$config_file" ]]; then
        # Load specified config file (must succeed)
        LOG_INFO "[$log_prefix] Loading config: $config_file"
        load_yaml_config "$config_file" || {
            LOG_ERROR "[$log_prefix] Failed to load config: $config_file"
            return 1
        }
    else
        # Load default configuration files (global and project)
        LOG_INFO "[$log_prefix] Loading default configuration files"
        load_config
    fi

    return 0
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
# Extract Config Section
# Extract all config keys matching a prefix and remove the prefix
# Usage: extract_config_section "slurm" result_array
# ---------------------------------------------------------------------------
extract_config_section() {
    local prefix="$1"
    # shellcheck disable=SC2034  # result_array is used via nameref
    local -n result_array="$2"  # nameref to associative array

    # Extract all config keys matching prefix and remove the prefix
    for key in "${!PRIMUS_CONFIG[@]}"; do
        if [[ "$key" =~ ^${prefix}\. ]]; then
            # Remove prefix to get parameter name (e.g., "slurm.partition" -> "partition")
            local param_name="${key#"${prefix}".}"
            # shellcheck disable=SC2034  # result_array is a nameref, accessed indirectly
            result_array["$param_name"]="${PRIMUS_CONFIG[$key]}"
        fi
    done

    return 0
}

# ---------------------------------------------------------------------------
# Export all functions
# ---------------------------------------------------------------------------
export -f load_yaml_config load_config load_config_auto
export -f get_config set_config
export -f extract_config_section

LOG_INFO_RANK0 "Primus config library loaded successfully"
