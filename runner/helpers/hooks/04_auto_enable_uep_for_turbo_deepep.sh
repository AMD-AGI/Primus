#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Global hook: auto-switch turbo deepep to the UEP backend when the installed
# Primus-Turbo binary lacks rocSHMEM internode support.
#
# Behavior:
#   - Only acts when turbo deepep is requested explicitly or via sync-free MoE
#     stages that auto-enable it later.
#   - Only acts for tp_size * ep_size > 8, which is the point where Primus-Turbo
#     DeepEP requires internode RDMA ranks on current MI300X layouts.
#   - Respects explicit caller choices for USING_UEP / REBUILD_UEP /
#     PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND.
#   - Scans installed Primus-Turbo binaries for the known rocSHMEM-disabled
#     assertion string. If uccl/deep_ep are already available, only enables
#     USING_UEP. Otherwise also enables REBUILD_UEP so the existing UEP hooks
#     build them.
###############################################################################

set -euo pipefail

# Pinned defaults for the auto-selected UCCL-EP fallback path.
UCCL_REF_FOR_AUTO_UEP_FALLBACK="15c71c649ce5f2816998b475d0f69bccbbb861a6"
UCCL_EP_ENABLE_AGGRESSIVE_ATOMIC_FOR_AUTO_UEP_FALLBACK="1"
UCCL_EP_FORCE_CURRENT_STREAM_FOR_AUTO_UEP_FALLBACK="1"

log_info_rank0() {
    if declare -F LOG_INFO_RANK0 >/dev/null 2>&1; then
        LOG_INFO_RANK0 "$@"
    else
        echo "$@"
    fi
}

emit_env_default_if_unset() {
    local name="$1"
    local value="$2"
    local current="${!name:-}"

    if [[ -n "${current}" ]]; then
        log_info_rank0 "[hook system] UCCL-EP fallback preserves explicit ${name}=${current}."
    else
        log_info_rank0 "[hook system] UCCL-EP fallback sets ${name}=${value} for the rocSHMEM-disabled Turbo DeepEP path."
        echo "env.${name}=${value}"
    fi
}

# Respect explicit backend or UEP choices from the caller.
if [[ -n "${PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND:-}" || -n "${USING_UEP:-}" || -n "${REBUILD_UEP:-}" ]]; then
    exit 0
fi

# Hooks receive: <group> <name> [full original args...]
if [[ $# -ge 2 ]]; then
    shift 2
fi

get_arg_value() {
    local key="$1"
    local default_value="$2"
    shift 2

    local value="$default_value"
    while [[ $# -gt 0 ]]; do
        if [[ "$1" == "$key" ]]; then
            if [[ $# -ge 2 ]]; then
                value="$2"
                shift 2
                continue
            fi
            value="1"
            break
        fi
        shift
    done

    printf '%s\n' "$value"
}

is_truthy() {
    case "${1,,}" in
        1|true|yes|on)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

use_turbo_deepep="$(get_arg_value --use_turbo_deepep false "$@")"
turbo_sync_free_moe_stage="$(get_arg_value --turbo_sync_free_moe_stage 0 "$@")"
tensor_model_parallel_size="$(get_arg_value --tensor_model_parallel_size 1 "$@")"
expert_model_parallel_size="$(get_arg_value --expert_model_parallel_size 1 "$@")"

want_turbo_deepep=0
if is_truthy "$use_turbo_deepep"; then
    want_turbo_deepep=1
elif [[ "$turbo_sync_free_moe_stage" =~ ^[0-9]+$ ]] && [[ "$turbo_sync_free_moe_stage" -gt 1 ]]; then
    want_turbo_deepep=1
fi

if [[ "$want_turbo_deepep" != "1" ]]; then
    exit 0
fi

if ! [[ "$tensor_model_parallel_size" =~ ^[0-9]+$ ]]; then
    tensor_model_parallel_size=1
fi
if ! [[ "$expert_model_parallel_size" =~ ^[0-9]+$ ]]; then
    expert_model_parallel_size=1
fi

tp_ep_size=$(( tensor_model_parallel_size * expert_model_parallel_size ))
if [[ "$tp_ep_size" -le 8 ]]; then
    exit 0
fi

# Detect whether the installed Primus-Turbo extension was compiled with
# DISABLE_ROCSHMEM.
if probe_detail="$(python3 - <<'PY'
import pathlib
import sys

try:
    import primus_turbo
except Exception:
    sys.exit(1)

root = pathlib.Path(primus_turbo.__file__).resolve().parent
needle = b"rocSHMEM is disabled during compilation"

for so_path in root.rglob("*.so"):
    try:
        if needle in so_path.read_bytes():
            print(f"string:{so_path.name}")
            sys.exit(0)
    except Exception:
        continue

sys.exit(1)
PY
)"
then
    log_info_rank0 "[hook system] Auto-enable UEP fallback for turbo deepep: tp*ep=${tp_ep_size}, installed Primus-Turbo lacks rocSHMEM internode support (${probe_detail})."
    log_info_rank0 "[hook system] Behavior change: this run will use the UEP/DEEP_EP fallback path instead of the default TURBO MoE dispatch/combine backend."
    log_info_rank0 "[hook system] Setting env.USING_UEP=1. The later 05_using_uep.sh hook will switch PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND from TURBO to DEEP_EP."
    emit_env_default_if_unset "UCCL_REF" "${UCCL_REF_FOR_AUTO_UEP_FALLBACK}"
    emit_env_default_if_unset "UCCL_EP_ENABLE_AGGRESSIVE_ATOMIC" "${UCCL_EP_ENABLE_AGGRESSIVE_ATOMIC_FOR_AUTO_UEP_FALLBACK}"
    emit_env_default_if_unset "UCCL_EP_FORCE_CURRENT_STREAM" "${UCCL_EP_FORCE_CURRENT_STREAM_FOR_AUTO_UEP_FALLBACK}"

    if python3 - <<'PY' >/dev/null 2>&1
import importlib.util
import sys

mods = ("uccl", "deep_ep")
sys.exit(0 if all(importlib.util.find_spec(name) for name in mods) else 1)
PY
    then
        log_info_rank0 "[hook system] uccl/deep_ep are already installed. No extra build hook will be triggered."
        echo "env.USING_UEP=1"
    else
        log_info_rank0 "[hook system] uccl/deep_ep are missing in the current image."
        log_info_rank0 "[hook system] Setting env.REBUILD_UEP=1. The later 04_rebuild_uep.sh hook will build/install uccl and deep_ep before 05_using_uep.sh switches backend to DEEP_EP."
        echo "env.USING_UEP=1"
        echo "env.REBUILD_UEP=1"
    fi
else
    exit 0
fi
