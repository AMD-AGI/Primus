#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Run control shared by run_batch.sh and run_batch_multinode.sh: launching a
# run under a watchdog, labelling and stopping training containers, the checks
# made before each run, the interrupt handler, and DRY_RUN.
#
# Callers source lib/common.sh and lib/run.sh first, set PERF_MULTINODE, and
# define build_launch_argv (which sets LAUNCH_ARGV for one config).
###############################################################################

# Exit codes, so a caller (or an agent) can tell outcomes apart.
PERF_EXIT_OK=0          # every run passed
PERF_EXIT_RUN_FAILED=1  # at least one run failed or was not started
PERF_EXIT_REFUSED=2     # bad input or a failed pre-run check; nothing launched
PERF_EXIT_INTERRUPTED=130

# Print a line, and append it to the batch summary once that file exists. The
# checks below log this way rather than being piped through tee by the caller,
# so they run in the caller's shell and their results (PERF_GPU_CHECK, an
# exported NCCL_SOCKET_IFNAME) stay visible to it.
perf_log() {
    printf '%s\n' "$*" | perf_log_stream
}

perf_log_stream() {
    if [ -n "${SUMMARY_FILE:-}" ] && [ -f "$SUMMARY_FILE" ]; then
        tee -a "$SUMMARY_FILE"
    else
        cat
    fi
}

# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------

# Every run is stopped when it exceeds MAX_RUN_SECONDS, or when its log has not
# grown for MAX_IDLE_SECONDS; 0 disables either limit. POLL_INTERVAL is how
# often the log is checked.
perf_check_run_limits() {
    local name
    for name in MAX_RUN_SECONDS MAX_IDLE_SECONDS POLL_INTERVAL GPU_BUSY_MIB GPU_BUSY_WAIT_SECONDS; do
        if ! [[ "${!name}" =~ ^[0-9]+$ ]]; then
            echo "[ERROR] $name must be a non-negative integer, got: '${!name}'" >&2
            return 1
        fi
    done
    if [ "$POLL_INTERVAL" -lt 1 ]; then
        echo "[ERROR] POLL_INTERVAL must be at least 1, got: '$POLL_INTERVAL'" >&2
        return 1
    fi
}

perf_run_limits_text() {
    echo "max_run=${MAX_RUN_SECONDS}s max_idle=${MAX_IDLE_SECONDS}s poll=${POLL_INTERVAL}s (0 = no limit)"
}

# ---------------------------------------------------------------------------
# Containers
#
# Every container this tool starts carries one label,
#   primus.perf=<user>/<batch stamp>/<runner host>/<runner pid>
# so cleanup can target exactly this batch's containers, and a later batch can
# find the ones an earlier batch left behind. primus-cli keeps only the last
# value of a repeated docker option, hence a single label.
# ---------------------------------------------------------------------------

PERF_LABEL_KEY="primus.perf"
PERF_LABEL_VALUE=""

perf_init_container_label() {
    PERF_LABEL_VALUE="$(id -un)/${BATCH_TS}/$(hostname)/$$"
}

# Multi-node always launches through the container entry. Single node does
# when PRIMUS_MODE is "container" (the default); "direct" runs in-process,
# e.g. when the runner itself is already inside a container.
perf_uses_containers() {
    [ "${PERF_MULTINODE:-0}" = 1 ] && return 0
    [ "${PRIMUS_MODE%% *}" = "container" ]
}

# Run a command on every node of the batch: locally for single node, once per
# node through srun for multi-node.
perf_on_nodes() {
    if [ "${PERF_MULTINODE:-0}" = 1 ]; then
        srun -N "$NNODES" "$@"
    else
        "$@"
    fi
}

# Stop this batch's training containers on every node.
perf_stop_batch_containers() {
    perf_uses_containers || return 0
    # Single quotes are deliberate: the body runs on the target node and gets
    # the label as $1.
    # shellcheck disable=SC2016
    perf_on_nodes bash -c '
        command -v docker >/dev/null 2>&1 || exit 0
        ids=$(docker ps -q --filter "label=$1")
        [ -n "$ids" ] || exit 0
        docker stop -t 20 $ids >/dev/null 2>&1
        # --rm removes the container asynchronously, and the next run reuses
        # its name (primus-training-<job id>), so wait until it is gone.
        for _ in $(seq 1 30); do
            [ -z "$(docker ps -aq --filter "label=$1")" ] && break
            sleep 1
        done
        exit 0
    ' _ "${PERF_LABEL_KEY}=${PERF_LABEL_VALUE}" >/dev/null 2>&1 || true
}

# Stop containers that earlier batches of the current user left running, e.g.
# after a SIGKILL or a scheduler time limit: docker containers live outside
# the job, so nothing else stops them. A container whose batch is still alive
# on this host is left alone; the GPU check then refuses to share the GPUs.
perf_stop_leftover_containers() {
    perf_uses_containers || return 0
    # shellcheck disable=SC2016
    perf_on_nodes bash -c '
        command -v docker >/dev/null 2>&1 || exit 0
        key=$1 me=$2 current=$3 stopped=""
        while IFS="|" read -r id name value; do
            [ -n "$id" ] || continue
            IFS=/ read -r user stamp host pid <<< "$value"
            [ "$user" = "$me" ] || continue
            [ "$value" = "$current" ] && continue
            if [ "$host" = "$(hostname)" ] && grep -qs run_batch "/proc/$pid/cmdline"; then
                echo "[WARN] $(hostname): container $name belongs to batch $stamp, which is still running (pid $pid); not stopping it."
                continue
            fi
            echo "[INFO] $(hostname): stopping container $name left over from batch $stamp."
            docker stop -t 20 "$id" >/dev/null 2>&1
            stopped="$stopped $id"
        done < <(docker ps --filter "label=$key" --format "{{.ID}}|{{.Names}}|{{.Label \"$key\"}}" 2>/dev/null)
        # Wait for --rm to finish removing them: this batch reuses the name.
        for id in $stopped; do
            for _ in $(seq 1 30); do
                [ -z "$(docker ps -aq --filter "id=$id")" ] && break
                sleep 1
            done
        done
        exit 0
    ' _ "$PERF_LABEL_KEY" "$(id -un)" "$PERF_LABEL_VALUE" 2>/dev/null | perf_log_stream
}

# Make sure every node has the image before the first run. Without this the
# first `docker run` pulls it, which both counts against the run limits (a
# silent multi-GB pull can trip MAX_IDLE_SECONDS) and leaves the image digest
# unknown until then. PULL_IMAGE=1 pulls even when the tag is present, to
# replace a stale or locally re-tagged copy with the registry's.
#
# Every node must then hold the same image: the same tag can name different
# images on different nodes, and a multi-node run mixing them measures
# neither. Mismatching image IDs refuse the batch.
PULL_IMAGE="${PULL_IMAGE:-0}"

perf_ensure_image() {
    local out rc ids
    perf_uses_containers || return 0
    [ -n "${DOCKER_IMAGE:-}" ] || return 0
    # shellcheck disable=SC2016
    out=$(perf_on_nodes bash -c '
        if ! command -v docker >/dev/null 2>&1; then
            echo "[WARN] $(hostname): docker not found; image presence not checked."
            exit 0
        fi
        if [ "$2" = 1 ] || ! docker image inspect "$1" >/dev/null 2>&1; then
            echo "[INFO] $(hostname): pulling $1 before the batch starts."
            if ! docker pull -q "$1" >/dev/null; then
                echo "[ERROR] $(hostname): docker pull $1 failed."
                exit 1
            fi
        fi
        echo "IMAGE_ID $(hostname) $(docker image inspect --format "{{.Id}}" "$1" 2>/dev/null)"
    ' _ "$DOCKER_IMAGE" "$PULL_IMAGE" 2>&1)
    rc=$?
    grep -v '^IMAGE_ID ' <<< "$out" | sed '/^$/d' | perf_log_stream
    [ "$rc" -eq 0 ] || return 1

    ids=$(awk '$1 == "IMAGE_ID" { print $3 }' <<< "$out" | sort -u)
    if [ "$(sed '/^$/d' <<< "$ids" | wc -l)" -gt 1 ]; then
        perf_log "[ERROR] $DOCKER_IMAGE is a different image on different nodes:"
        awk '$1 == "IMAGE_ID" { printf "          %s %s\n", $2, $3 }' <<< "$out" | perf_log_stream
        perf_log "        Rerun with PULL_IMAGE=1 to pull the registry's copy on every node."
        return 1
    fi
}

# ---------------------------------------------------------------------------
# GPU check
#
# A run sharing its GPUs with another process reports numbers that mean
# nothing, or dies with an out-of-memory error that looks like its own. So a
# run only starts when no GPU has more than GPU_BUSY_MIB of memory in use
# (idle MI355X GPUs show about 280 MiB). ALLOW_BUSY_GPUS=1 overrides this.
# ---------------------------------------------------------------------------

GPU_BUSY_MIB="${GPU_BUSY_MIB:-2048}"
GPU_BUSY_WAIT_SECONDS="${GPU_BUSY_WAIT_SECONDS:-60}"
ALLOW_BUSY_GPUS="${ALLOW_BUSY_GPUS:-0}"
PERF_GPU_CHECK="not checked yet"

# One "<gpu index> <MiB in use>" line per GPU on this node; nothing when no
# GPU tool works here. Shipped to remote nodes as source text, so it must
# stay self-contained.
perf_gpu_vram_used() {
    local tool
    if command -v amd-smi >/dev/null 2>&1; then
        tool=amd-smi
    elif command -v rocm-smi >/dev/null 2>&1; then
        tool=rocm-smi
    else
        return 0
    fi
    if [ "$tool" = amd-smi ]; then
        amd-smi metric --mem-usage --json 2>/dev/null
    else
        rocm-smi --showmeminfo vram --json 2>/dev/null
    fi | python3 -c '
import json, sys
tool = sys.argv[1]
try:
    data = json.load(sys.stdin)
except Exception:
    sys.exit(0)
scale = {"B": 1.0 / 2**20, "KB": 1.0 / 2**10, "KIB": 1.0 / 2**10, "MB": 1, "MIB": 1, "GB": 2**10, "GIB": 2**10}
if tool == "amd-smi":
    gpus = data.get("gpu_data", []) if isinstance(data, dict) else data
    for g in gpus:
        used = ((g.get("mem_usage") or {}).get("used_vram")) or {}
        value = used.get("value")
        if isinstance(value, (int, float)):
            print(g.get("gpu"), int(value * scale.get(str(used.get("unit", "MB")).upper(), 1)))
else:
    for card, fields in sorted(data.items()):
        used = fields.get("VRAM Total Used Memory (B)") if isinstance(fields, dict) else None
        if used is not None:
            print(card.replace("card", ""), int(int(used) / 2**20))
' "$tool" 2>/dev/null
}

# Sets PERF_GPU_CHECK to a one-line description and returns 1 if any GPU is
# busy. Not being able to check (no amd-smi / rocm-smi, or no GPUs visible)
# is recorded and does not block the run.
perf_check_gpus_idle_once() {
    local report busy max
    report=$(perf_on_nodes bash -c "$(declare -f perf_gpu_vram_used); h=\$(hostname); perf_gpu_vram_used | sed \"s/^/\$h /\"" 2>/dev/null)
    if [ -z "$report" ]; then
        PERF_GPU_CHECK="not checked (no GPU memory readings from amd-smi or rocm-smi)"
        return 0
    fi
    max=$(awk '$3 > m { m = $3 } END { print m + 0 }' <<< "$report")
    busy=$(awk -v t="$GPU_BUSY_MIB" '$3 > t { printf "%s GPU %s: %s MiB in use\n", $1, $2, $3 }' <<< "$report")
    if [ -z "$busy" ]; then
        PERF_GPU_CHECK="idle ($(wc -l <<< "$report") GPUs, at most ${max} MiB in use; threshold ${GPU_BUSY_MIB} MiB)"
        return 0
    fi
    PERF_GPU_CHECK="busy (at most ${max} MiB in use; threshold ${GPU_BUSY_MIB} MiB)"
    PERF_GPU_BUSY_REPORT="$busy"
    return 1
}

# Get the GPUs ready for the next run: stop this tool's leftover containers,
# then wait up to GPU_BUSY_WAIT_SECONDS for memory to be released (a container
# that just exited can take a few seconds). Returns 1, after explaining why,
# when the GPUs are still busy.
perf_prepare_gpus() {
    local waited=0
    PERF_GPU_BUSY_REPORT=""
    perf_stop_batch_containers
    perf_stop_leftover_containers
    until perf_check_gpus_idle_once; do
        if [ "$ALLOW_BUSY_GPUS" = 1 ]; then
            perf_log "[WARN] GPUs are in use, continuing because ALLOW_BUSY_GPUS=1:"
            sed 's/^/         /' <<< "$PERF_GPU_BUSY_REPORT" | perf_log_stream
            PERF_GPU_CHECK="$PERF_GPU_CHECK; ALLOW_BUSY_GPUS=1"
            return 0
        fi
        if [ "$waited" -ge "$GPU_BUSY_WAIT_SECONDS" ]; then
            perf_log "[ERROR] GPUs are in use by something other than this batch:"
            sed 's/^/          /' <<< "$PERF_GPU_BUSY_REPORT" | perf_log_stream
            perf_log "        Running containers:"
            # shellcheck disable=SC2016
            perf_on_nodes bash -c 'command -v docker >/dev/null 2>&1 && docker ps --format "$(hostname) {{.Names}}  {{.Image}}  {{.Status}}"' 2>/dev/null |
                sed 's/^/          /' | perf_log_stream
            perf_log "        Free the GPUs, or set ALLOW_BUSY_GPUS=1 to run anyway (results may be meaningless)."
            return 1
        fi
        sleep 5
        waited=$((waited + 5))
    done
}

# ---------------------------------------------------------------------------
# NCCL / Gloo socket interface
#
# When NCCL_SOCKET_IFNAME is unset, Primus picks the interface that owns the
# first address of `hostname -I`, which is docker0 on some nodes; RCCL then
# fails at its first collective, even on a single node. Bridges and loopback
# are refused outright, and an unusable automatic pick is replaced by the
# default-route interface.
# ---------------------------------------------------------------------------

PERF_NIC_CHECK="not checked yet"

perf_is_bad_ifname() {
    case "$1" in
        lo|docker*|br-*|veth*|virbr*|cni*|flannel*) return 0 ;;
    esac
    return 1
}

perf_check_socket_ifname() {
    local var picked route
    for var in NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME; do
        if [ -n "${!var:-}" ] && perf_is_bad_ifname "${!var}"; then
            perf_log "[ERROR] $var=${!var} is a bridge or loopback interface; RCCL/Gloo cannot bootstrap over it."
            perf_log "        Export a real NIC instead (see \`ip -br addr\`), e.g. $var=eno0."
            return 1
        fi
    done
    if [ -n "${NCCL_SOCKET_IFNAME:-}" ]; then
        PERF_NIC_CHECK="NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME (set by the user)"
        return 0
    fi
    if ! command -v ip >/dev/null 2>&1; then
        PERF_NIC_CHECK="not checked (no ip command); Primus auto-detects"
        return 0
    fi

    picked=$(bash "$PRIMUS_PATH/runner/helpers/envs/get_ip_interface.sh" 2>/dev/null | head -1)
    if [ -n "$picked" ] && ! perf_is_bad_ifname "$picked"; then
        PERF_NIC_CHECK="$picked (Primus auto-detection)"
        return 0
    fi

    route=$(ip route get 1.1.1.1 2>/dev/null | awk '{ for (i = 1; i < NF; i++) if ($i == "dev") { print $(i + 1); exit } }')
    if [ -n "$route" ] && ! perf_is_bad_ifname "$route"; then
        export NCCL_SOCKET_IFNAME="$route"
        export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-$route}"
        perf_log "[INFO] Primus would pick '${picked:-nothing}' for NCCL/Gloo; using the default-route interface '$route' instead."
        PERF_NIC_CHECK="$route (default route; Primus would have picked '${picked:-nothing}')"
        return 0
    fi

    perf_log "[ERROR] no usable NCCL/Gloo interface: Primus would pick '${picked:-nothing}' and there is no default-route interface."
    perf_log "        Export NCCL_SOCKET_IFNAME (and GLOO_SOCKET_IFNAME) with a real NIC from \`ip -br addr\`."
    return 1
}

# ---------------------------------------------------------------------------
# Checks before the first run. Logs what it does; returns 1 to refuse.
# ---------------------------------------------------------------------------

perf_preflight() {
    perf_log "[INFO] Pre-run checks: image, leftover containers, GPU memory, NCCL interface."
    if ! perf_ensure_image; then
        perf_log "[ERROR] the image is not available on every node."
        return 1
    fi
    perf_prepare_gpus || return 1
    perf_log "[INFO] GPU check: $PERF_GPU_CHECK"
    perf_check_socket_ifname || return 1
    perf_log "[INFO] NCCL interface: $PERF_NIC_CHECK"
}

# ---------------------------------------------------------------------------
# Launch under a watchdog
#
# The launcher runs in its own session (setsid), so its PID is also its
# process-group ID and one signal reaches primus-cli, docker and (multi-node)
# the local srun holding the Slurm step. Output goes through
# perf_redact_stream into the log; the terminal follows the log with tail.
#
# Sets RUN_EXIT_CODE, RUN_ELAPSED and RUN_FAILURE_REASON ("none" on success,
# otherwise "<category>: <detail>" with category timeout, idle, fail_pattern,
# signal or exit).
# ---------------------------------------------------------------------------

RUN_PID=""
TAIL_PID=""
CURRENT_RESULT_FILE=""
CURRENT_RUN_START=0
FAIL_PATTERN="${FAIL_PATTERN:-}"

perf_exit_reason() {
    local code="$1" sig
    if [ "$code" -eq 0 ]; then
        echo "none"
    elif [ "$code" -gt 128 ] && [ "$code" -lt 160 ]; then
        sig=$(kill -l $((code - 128)) 2>/dev/null)
        echo "signal: killed by SIG${sig:-$((code - 128))} (exit $code)"
    else
        echo "exit: launcher exited with code $code"
    fi
}

# Stop a run: SIGTERM to its process group, SIGKILL after 10 s, then stop its
# containers, since `docker run` losing its client leaves the container up.
perf_kill_run() {
    local pid="$1" i
    if [ -n "$pid" ]; then
        kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
        for i in 1 2 3 4 5 6 7 8 9 10; do
            kill -0 "$pid" 2>/dev/null || break
            sleep 1
        done
        kill -KILL -- "-$pid" 2>/dev/null
        kill -KILL "$pid" 2>/dev/null
    fi
    perf_stop_batch_containers
}

# perf_launch_with_watchdog <result_file> <first output line of the log>
perf_launch_with_watchdog() {
    local result_file="$1" first_line="$2"
    local start now size last_size=-1 last_change tick=0 reason=""
    # Run inside the new session: the launcher's output, with secret values
    # masked, appended to the log. PIPESTATUS[0] is the launcher's exit code.
    local runner_cmd
    runner_cmd="$(declare -f perf_redact_stream)"'
        set -o pipefail
        "$@" 2>&1 | perf_redact_stream >> "$PERF_RUN_LOG"
        exit "${PIPESTATUS[0]}"'

    tail -n +"$first_line" -F "$result_file" 2>/dev/null &
    TAIL_PID=$!

    start=$(date +%s)
    last_change=$start
    CURRENT_RESULT_FILE="$result_file"
    CURRENT_RUN_START=$start

    if command -v setsid >/dev/null 2>&1; then
        PERF_RUN_LOG="$result_file" setsid bash -c "$runner_cmd" _ "${LAUNCH_ARGV[@]}" &
    else
        # Without setsid only the launcher itself can be signalled; the time
        # limits and container cleanup still apply.
        PERF_RUN_LOG="$result_file" bash -c "$runner_cmd" _ "${LAUNCH_ARGV[@]}" &
    fi
    RUN_PID=$!

    while kill -0 "$RUN_PID" 2>/dev/null; do
        sleep 1
        tick=$((tick + 1))
        [ $((tick % POLL_INTERVAL)) -eq 0 ] || continue

        now=$(date +%s)
        size=$(stat -c %s "$result_file" 2>/dev/null || echo 0)
        if [ "$size" != "$last_size" ]; then
            last_size=$size
            last_change=$now
        fi

        # FAIL_PATTERN is matched against the run output only, never the
        # banner or the config dump above it.
        if [ -n "$FAIL_PATTERN" ] && tail -n +"$first_line" "$result_file" 2>/dev/null | grep -qE -- "$FAIL_PATTERN"; then
            reason="fail_pattern: failure text matching FAIL_PATTERN appeared in the output"
        elif [ "$MAX_RUN_SECONDS" -gt 0 ] && [ $((now - start)) -ge "$MAX_RUN_SECONDS" ]; then
            reason="timeout: still running after MAX_RUN_SECONDS=${MAX_RUN_SECONDS}s"
        elif [ "$MAX_IDLE_SECONDS" -gt 0 ] && [ $((now - last_change)) -ge "$MAX_IDLE_SECONDS" ]; then
            reason="idle: no new output for MAX_IDLE_SECONDS=${MAX_IDLE_SECONDS}s"
        fi

        if [ -n "$reason" ]; then
            # The run may have ended between the checks; then it decides.
            if ! kill -0 "$RUN_PID" 2>/dev/null; then
                reason=""
                break
            fi
            {
                echo ""
                echo ">>> [$(date '+%H:%M:%S')] Stopping the run: $reason"
            } | tee -a "$result_file" >> "$SUMMARY_FILE"
            perf_kill_run "$RUN_PID"
            break
        fi
    done

    wait "$RUN_PID" 2>/dev/null
    RUN_EXIT_CODE=$?
    RUN_PID=""
    # Let tail print the last lines before stopping it.
    sleep 1
    kill "$TAIL_PID" 2>/dev/null
    wait "$TAIL_PID" 2>/dev/null
    TAIL_PID=""

    RUN_ELAPSED=$(( $(date +%s) - start ))
    RUN_FAILURE_REASON="${reason:-$(perf_exit_reason "$RUN_EXIT_CODE")}"
    CURRENT_RESULT_FILE=""
}

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

# Refresh results.csv / results.json from every log in RESULT_DIR. EXTRACT=0
# skips it. A failure is reported but never changes the batch's exit code.
perf_extract_results() {
    local out
    [ "${EXTRACT:-1}" = 1 ] || return 0
    if ! command -v python3 >/dev/null 2>&1; then
        perf_log "[WARN] python3 not found; results.csv / results.json not written."
        return 0
    fi
    if out=$(python3 "$SCRIPT_DIR/extract_results.py" --quiet "$RESULT_DIR" 2>&1); then
        perf_log "$out"
    else
        perf_log "$out"
        perf_log "[WARN] result extraction failed; rerun: python3 $SCRIPT_DIR/extract_results.py $RESULT_DIR"
    fi
}

# ---------------------------------------------------------------------------
# Interrupt handling
#
# Ctrl+C, SIGTERM or SIGHUP stops the run in flight and its containers, closes
# its log, and abandons the rest of the batch. With the launcher in the
# background the trap fires within a second. A SIGKILL cannot be handled: the
# summary then shows which runs started, and the next batch stops the
# container that was left behind.
# ---------------------------------------------------------------------------

# Invoked through `trap`, so shellcheck sees no caller.
# shellcheck disable=SC2317,SC2329
perf_on_interrupt() {
    trap '' INT TERM HUP
    echo "" >&2
    echo ">>> Interrupted. Stopping the current run; the rest of the batch is abandoned." >&2

    [ -n "$TAIL_PID" ] && kill "$TAIL_PID" 2>/dev/null
    if [ -n "$RUN_PID" ]; then
        perf_kill_run "$RUN_PID"
    else
        perf_stop_batch_containers
    fi

    if [ -n "$CURRENT_RESULT_FILE" ]; then
        print_per_run_footer "$CURRENT_RESULT_FILE" "$PERF_EXIT_INTERRUPTED" \
            "$(( $(date +%s) - CURRENT_RUN_START ))" "interrupted: the batch was stopped by a signal"
        echo ">>> Interrupted run log: $CURRENT_RESULT_FILE" >&2
    fi

    {
        echo ""
        echo "################################################################################"
        echo "# Primus Benchmark Batch INTERRUPTED"
        echo "################################################################################"
        echo "# Interrupted at: $(date '+%Y-%m-%d %H:%M:%S %Z')"
        # Defaulted because an interrupt can land before the run loop starts.
        echo "# Runs started  : ${RUNS_STARTED:-0}"
        echo "# Runs failed   : ${RUNS_FAILED:-0}"
        echo "# In-flight run : ${CURRENT_RESULT_FILE:-<none>}"
        echo "################################################################################"
    } | tee -a "$SUMMARY_FILE"

    perf_extract_results
    exit "$PERF_EXIT_INTERRUPTED"
}

# ---------------------------------------------------------------------------
# DRY_RUN=1: show what would run, without docker, GPUs or HF_TOKEN.
# ---------------------------------------------------------------------------

perf_print_plan() {
    local config_file n=0 total
    total=$(( ${#CONFIG_FILES[@]} * NUM_REPS ))
    echo "DRY RUN -- nothing is launched."
    echo "Config source: $CONFIG_SOURCE"
    echo "RESULT_DIR   : $RESULT_DIR"
    echo "Run limits   : $(perf_run_limits_text)"
    echo ""
    for config_file in "${CONFIG_FILES[@]}"; do
        n=$((n + 1))
        prepare_config "$config_file"
        echo "[$n/${#CONFIG_FILES[@]}] ${FRAMEWORK}/${TRAIN_SUITE:-?} ${config_name} [${config_hash}]"
        echo "      MBS=${MBS} GBS=${GBS} seq=${SEQ_LEN} steps=${STEPS} reps=${NUM_REPS}"
        if build_launch_argv "$config_file" "$FRAMEWORK" "$TRAIN_SUITE" 2>/dev/null; then
            echo "      log: $(run_log_name 1)"
            echo "      cmd: ${LAUNCH_ARGV[*]}"
        else
            echo "      cannot build a launch command for this config (see the error when run)."
        fi
    done
    echo ""
    echo "${#CONFIG_FILES[@]} configs, $total runs."
}
