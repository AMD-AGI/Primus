#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# primus-cli deps: clone the third_party sources pinned in the packaged
# _thirdparty.lock to a local directory (full source, incl. nested submodules),
# so backends like Megatron / TorchTitan can run from source -- this avoids the
# missing-Makefile / missing-.so problems that `pip install git+...` hits.
#

set -euo pipefail

print_usage() {
cat << 'EOF'
Primus CLI - deps mode

Usage:
    primus-cli deps sync [--dir DIR] [--dry-run]

Description:
    Clone each third_party dependency pinned in the packaged _thirdparty.lock
    (shipped inside the installed wheel) to a local directory, using
    `git clone --recurse-submodules` + `git checkout <commit>`. This yields the
    full source tree (Makefiles, .cpp helpers, nested submodules), then prints
    the PYTHONPATH to use for running the backends.

Options:
    --dir DIR     Target directory
                  (default: $PRIMUS_THIRDPARTY_DIR or ~/.cache/Primus/third_party)
    --dry-run     Show what would be cloned without doing it
    -h, --help    Show this help
EOF
}

RUNNER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck disable=SC1091
source "$RUNNER_DIR/lib/common.sh" || {
    echo "[ERROR] Failed to load common library: $RUNNER_DIR/lib/common.sh" >&2
    exit 1
}

# Drop the leading 'deps' subcommand token if present.
[[ "${1:-}" == "deps" ]] && shift

SUB="${1:-sync}"
if [[ "$SUB" == "-h" || "$SUB" == "--help" ]]; then
    print_usage
    exit 0
fi
if [[ $# -gt 0 ]]; then shift; fi

# Lock file ships next to the primus package (installed: primus/_thirdparty.lock,
# i.e. one level above runner/).
LOCK_FILE="${RUNNER_DIR}/../_thirdparty.lock"

case "$SUB" in
    sync)
        DEST="${PRIMUS_THIRDPARTY_DIR:-$HOME/.cache/Primus/third_party}"
        DRY_RUN=0
        while [[ $# -gt 0 ]]; do
            case "$1" in
                --dir)
                    DEST="${2:?--dir requires a path}"
                    shift 2
                    ;;
                --dry-run)
                    DRY_RUN=1
                    shift
                    ;;
                -h|--help)
                    print_usage
                    exit 0
                    ;;
                *)
                    LOG_ERROR "[deps] Unknown argument: $1"
                    exit 2
                    ;;
            esac
        done

        if [[ ! -f "$LOCK_FILE" ]]; then
            LOG_ERROR "[deps] Lock file not found: $LOCK_FILE"
            LOG_ERROR "[deps] It is generated at build time into the wheel; only available in installed builds."
            exit 1
        fi

        require_command git "Install git to sync third_party sources."
        require_command python3 "python3 is required to parse the lock file."

        mkdir -p "$DEST"
        LOG_INFO_RANK0 "[deps] Syncing third_party sources into: $DEST"

        # Emit "name<TAB>url<TAB>commit" per entry from the JSON lock.
        mapfile -t ENTRIES < <(python3 -c '
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
for e in data.get("third_party", []):
    print("\t".join([e["name"], e["url"], e["commit"]]))
' "$LOCK_FILE")

        PYTHONPATHS=()
        for entry in "${ENTRIES[@]}"; do
            [[ -n "$entry" ]] || continue
            IFS=$'\t' read -r name url commit <<< "$entry"
            target="$DEST/$name"
            PYTHONPATHS+=("$target")

            if [[ "$DRY_RUN" == "1" ]]; then
                LOG_INFO_RANK0 "[deps] would sync ${name} -> ${target} @ ${commit} (${url})"
                continue
            fi

            if [[ -d "$target/.git" ]]; then
                LOG_INFO_RANK0 "[deps] Updating ${name} @ ${commit}"
                git -C "$target" fetch origin || true
                git -C "$target" checkout -q "$commit"
                git -C "$target" submodule update --init --recursive
            else
                LOG_INFO_RANK0 "[deps] Cloning ${name} @ ${commit}"
                git clone --recurse-submodules "$url" "$target"
                git -C "$target" checkout -q "$commit"
                git -C "$target" submodule update --init --recursive
            fi
        done

        joined="$(IFS=:; echo "${PYTHONPATHS[*]}")"
        PRINT_INFO_RANK0 ""
        PRINT_INFO_RANK0 "[deps] Done. Add the sources to PYTHONPATH:"
        PRINT_INFO_RANK0 "    export PYTHONPATH=\"${joined}\${PYTHONPATH:+:\$PYTHONPATH}\""
        ;;
    *)
        LOG_ERROR "[deps] Unknown subcommand: $SUB"
        print_usage
        exit 2
        ;;
esac
