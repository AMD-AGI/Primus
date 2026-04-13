#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

path_prepend_unique() {
    local var_name="$1"
    shift
    local current="${!var_name:-}"
    local result=""
    local entry candidate
    local -a entries=()

    IFS=':' read -r -a entries <<< "$current"

    for candidate in "$@"; do
        [[ -n "$candidate" && -d "$candidate" ]] || continue
        case ":$result:" in
            *":$candidate:"*) ;;
            *) result="${result:+$result:}$candidate" ;;
        esac
    done

    for entry in "${entries[@]}"; do
        [[ -n "$entry" ]] || continue
        case ":$result:" in
            *":$entry:"*) ;;
            *) result="${result:+$result:}$entry" ;;
        esac
    done

    printf -v "$var_name" '%s' "$result"
    declare -x "${var_name}=${!var_name}"
}

path_append_unique() {
    local var_name="$1"
    shift
    local result="${!var_name:-}"
    local candidate

    for candidate in "$@"; do
        [[ -n "$candidate" && -d "$candidate" ]] || continue
        case ":$result:" in
            *":$candidate:"*) ;;
            *) result="${result:+$result:}$candidate" ;;
        esac
    done

    printf -v "$var_name" '%s' "$result"
    declare -x "${var_name}=${!var_name}"
}

ensure_rocm_ld_library_path() {
    local rocm_root="${ROCM_HOME:-/opt/rocm}"
    local rocm_lib="${rocm_root}/lib"

    if [[ ! -d "$rocm_lib" ]]; then
        rocm_lib="$(compgen -G '/opt/rocm-*/lib' | head -n1 || true)"
    fi

    [[ -n "$rocm_lib" ]] && path_prepend_unique LD_LIBRARY_PATH "$rocm_lib"
}
