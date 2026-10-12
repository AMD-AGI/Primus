#!/bin/bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Hylo / KDA / GDN configs depend on flash-linear-attention Triton kernels that
# hang during autotune (num_stages >= 3) on MI300X/ROCm. examples/run_pretrain.sh
# applied tools/hybrid/patch_fla_triton_autotune_hang.sh for those configs;
# primus-cli does not, so this hook keeps the same workaround on the CLI path.
#
# Named z_* so it runs after prepare_experiment.sh (framework pip install).
# Failure is non-fatal: training may hang during autotune, matching run_pretrain.sh.
###############################################################################
set -euo pipefail

: "${1:?missing hook group}" "${2:?missing hook name}"
shift 2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRIMUS_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../../../hook_common.sh"

CONFIG_FILE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --config=*)
            CONFIG_FILE="${1#--config=}"
            shift
            ;;
        *)
            shift
            ;;
    esac
done

if [[ -z "$CONFIG_FILE" ]]; then
    CONFIG_FILE="${EXP:-}"
fi
if [[ -z "$CONFIG_FILE" ]]; then
    exit 0
fi

EXP_BASENAME="$(basename "$CONFIG_FILE")"
if [[ "${EXP_BASENAME}" != hylo_* && "${EXP_BASENAME}" != kda_* && "${EXP_BASENAME}" != gdn_* ]]; then
    exit 0
fi

PATCH="${PRIMUS_ROOT}/tools/hybrid/patch_fla_triton_autotune_hang.sh"
if [[ ! -f "$PATCH" ]]; then
    LOG_WARN "FLA Triton autotune-hang patch script not found: $PATCH"
    exit 0
fi

LOG_INFO "Detected hybrid model config (${EXP_BASENAME}); applying FLA Triton autotune-hang patch ..."
if ! bash "$PATCH"; then
    LOG_WARN "FLA Triton autotune-hang patch failed; continuing, but training may hang during autotuning."
fi
exit 0
