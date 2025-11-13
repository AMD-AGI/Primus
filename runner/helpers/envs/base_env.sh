#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# =============================================================================
# Base Environment Configuration
# =============================================================================
# This file provides the foundation for all environment configurations:
#   - Logging functions (LOG_INFO, LOG_INFO_RANK0, LOG_ERROR, etc.)
#   - Distributed training cluster information (MASTER_ADDR, NNODES, etc.)
#   - Python path setup
#
# Network, performance tuning, and GPU-specific settings are loaded separately
# =============================================================================

# ---------------------------------------------------------------------------
# Guard: avoid duplicate exports/logging on multiple sourcing
# ---------------------------------------------------------------------------
if [[ -n "${__PRIMUS_BASE_ENV_SOURCED:-}" ]]; then
  return 0
fi
export __PRIMUS_BASE_ENV_SOURCED=1

# ---------------------------------------------------------------------------
# Load common library for consistent logging
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/../../lib/common.sh" ]]; then
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/../../lib/common.sh"
else
    # Fallback logging functions if common.sh not available
    HOSTNAME="$(hostname)"
    export HOSTNAME

    LOG_INFO() {
        if [ "$*" = "" ]; then
            echo ""
        else
            echo "[NODE-${NODE_RANK:-0}($HOSTNAME)] $*"
        fi
    }

    LOG_INFO_RANK0() {
        if [ "${NODE_RANK:-0}" -eq 0 ]; then
            if [ "$*" = "" ]; then
                echo ""
            else
                echo "[NODE-${NODE_RANK:-0}($HOSTNAME)] $*"
            fi
        fi
    }

    LOG_ERROR() {
        echo "[NODE-${NODE_RANK:-0}($HOSTNAME)] [ERROR] $*" >&2
    }

    LOG_WARN() {
        echo "[NODE-${NODE_RANK:-0}($HOSTNAME)] [WARN] $*" >&2
    }

    log_exported_vars() {
        LOG_INFO_RANK0 "========== $1 =========="
        for var in "${@:2}"; do
            LOG_INFO_RANK0 "    $var=${!var-}"
        done
    }
fi

# ---------------------------------------------------------------------------
# Distributed Training Cluster Configuration
# ---------------------------------------------------------------------------
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-1234}
export NNODES=${NNODES:-1}
export NODE_RANK=${NODE_RANK:-0}
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}

log_exported_vars "Training Cluster Info" \
    MASTER_ADDR MASTER_PORT NNODES NODE_RANK GPUS_PER_NODE

# ---------------------------------------------------------------------------
# Python Path Setup
# ---------------------------------------------------------------------------
PRIMUS_PATH=$(cd "$SCRIPT_DIR/../../.." && pwd)
site_packages=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])" 2>/dev/null || echo "")
if [[ -n "$site_packages" ]]; then
    export PYTHONPATH="${PRIMUS_PATH}:${site_packages}:${PYTHONPATH:-}"
else
    export PYTHONPATH="${PRIMUS_PATH}:${PYTHONPATH:-}"
fi

log_exported_vars "Python Path" PYTHONPATH
