#!/bin/bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# Run the labeled container, preserve its status, and remove only containers
# created by this invocation when the launcher fails or receives a signal.
# This owns the launcher's final wait and exits the calling shell.
run_container_with_cleanup() {
    local launch_id="$1"
    shift
    local runtime="$1"
    local client_pid=""

    # Invoked by the EXIT trap while the launch function's locals are in scope.
    # shellcheck disable=SC2317
    _primus_cleanup_container() {
        local status=$?
        local ids="" cid
        trap - EXIT HUP INT TERM
        # A successful detached launch intentionally outlives the client.
        [[ "$status" -ne 0 ]] || return 0
        # Stop the client before querying so it cannot start more work after
        # cleanup. Killing a Docker client alone does not stop its container.
        if [[ -n "$client_pid" ]]; then
            kill -KILL "$client_pid" 2>/dev/null || true
            wait "$client_pid" 2>/dev/null || true
        fi
        if ! ids=$("$runtime" ps -aq --filter "label=io.primus.launch=$launch_id"); then
            echo "[container][WARN] Could not inspect containers for launch $launch_id" >&2
            return "$status"
        fi
        while IFS= read -r cid; do
            [[ "$cid" =~ ^[a-f0-9]{12,64}$ ]] || continue
            "$runtime" rm -f "$cid" >/dev/null || \
                echo "[container][WARN] Could not remove container $cid" >&2
        done <<< "$ids"
        return "$status"
    }

    trap '_primus_cleanup_container' EXIT
    trap 'exit 129' HUP
    trap 'exit 130' INT
    trap 'exit 143' TERM

    # A foreground wait would defer Bash's traps until the client finishes.
    # Preserve stdin for callers using an interactive container.
    "$@" <&0 &
    client_pid=$!
    local status=0
    wait "$client_pid" || status=$?
    client_pid=""
    exit "$status"
}
