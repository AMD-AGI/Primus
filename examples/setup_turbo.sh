#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

set -euo pipefail

TURBO_COMMIT="${PRIMUS_TURBO_COMMIT:-6d39647350baa180cccc61e701cf75281dc7e878}"
PRIMUS_TURBO_PATH="${PRIMUS_TURBO_PATH:-}"
REPO_URL="https://github.com/AMD-AGI/primus-turbo.git"

function get_installed_commit() {
    python3 -c "
try:
    import primus_turbo
    print(getattr(primus_turbo, '__commit__', ''), end='')
except Exception:
    try:
        from importlib.metadata import version
        ver = version('primus-turbo')
        print(ver.split('+')[1] if '+' in ver else '', end='')
    except Exception:
        print('', end='')
"
}

function install_turbo_from_path() {
    local path="$1"
    echo "[Primus-Turbo] Installing from path: $path"
    (
        cd "$path"
        pip3 install -r requirements.txt
        pip3 install --no-build-isolation .
    )
}

function install_turbo_from_commit() {
    local commit="$1"

    # Resolve PRIMUS_PATH if not already exported
    local primus_path="${PRIMUS_PATH:-$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")}"
    local hostname
    hostname="$(hostname)"
    local cache_dir="$primus_path/build/$hostname/primus-turbo-$commit"

    echo "[Primus-Turbo] Requested install from commit: $commit"
    echo "[Primus-Turbo] Target cache dir: $cache_dir"

    if [[ -d "$cache_dir/.git" ]]; then
        echo "[Primus-Turbo] Found existing clone. Skipping download."
    else
        echo "[Primus-Turbo] Cloning Primus-Turbo..."
        rm -rf "$cache_dir"
        git clone "${REPO_URL}" --recursive "$cache_dir"
        git -C "$cache_dir" checkout "$commit"
    fi

    echo "[Primus-Turbo] Installing from $cache_dir"
    (
        cd "$cache_dir"
        pip3 install -r requirements.txt
        pip3 install --no-build-isolation .
    )
}

function install_turbo_default() {
    local default_commit="main"
    echo "[Primus-Turbo] Installing default commit: $default_commit"
    install_turbo_from_commit "$default_commit"
}


if [[ -n "$PRIMUS_TURBO_PATH" ]]; then
    resolved_path=$(realpath "$PRIMUS_TURBO_PATH")

    if [[ ! -d "$resolved_path" ]]; then
        echo "[Primus-Turbo][ERROR] PRIMUS_TURBO_PATH does not exist or is not a directory: $resolved_path"
        exit 1
    fi

    echo "[Primus-Turbo] Installing from user-specified PRIMUS_TURBO_PATH: $resolved_path"
    install_turbo_from_path "$resolved_path"
    exit 0
fi

if [[ -n "$TURBO_COMMIT" ]]; then
    INSTALLED_COMMIT="$(get_installed_commit)"
    if [[ "$INSTALLED_COMMIT" == "$TURBO_COMMIT" && -n "$INSTALLED_COMMIT" ]]; then
        echo "[Primus-Turbo] Already installed commit $INSTALLED_COMMIT, skip install."
        exit 0
    fi
    install_turbo_from_commit "$TURBO_COMMIT"
    exit 0
fi

INSTALLED_COMMIT="$(get_installed_commit)"
if [[ -n "$INSTALLED_COMMIT" ]]; then
    echo "[Primus-Turbo] Found installed primus-turbo (commit: $INSTALLED_COMMIT), using it."
    exit 0
fi

install_turbo_default
