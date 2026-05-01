#!/usr/bin/env bash
# Monitor rocm-smi VRAM and/or GPU utilization across a set of Slurm nodes.
#
# Usage:
#   monitor_rocm_mem.sh                          # METRIC=all by default
#   METRIC=mem  monitor_rocm_mem.sh              # only VRAM (used/total GiB, pct)
#   METRIC=util monitor_rocm_mem.sh              # only GPU util %
#   METRIC=all  monitor_rocm_mem.sh              # util% / mem%
#   NODES="mi355-gpu-7 mi355-gpu-8" monitor_rocm_mem.sh
#   CONTAINER="" monitor_rocm_mem.sh             # call rocm-smi directly on the host
#   INTERVAL=10 monitor_rocm_mem.sh
#   ONCE=1 monitor_rocm_mem.sh                   # single snapshot then exit
#   CSV=/tmp/gpu.csv monitor_rocm_mem.sh         # also append samples to CSV
#   JOB=12345 monitor_rocm_mem.sh                # take nodes from this Slurm job

set -u

NODES=${NODES:-}
CONTAINER=${CONTAINER-primus-training}
INTERVAL=${INTERVAL:-5}
ONCE=${ONCE:-0}
CSV=${CSV:-}
JOB=${JOB:-}
METRIC=${METRIC:-all}
SSH_OPTS=(-o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=5 -o LogLevel=ERROR)

case "$METRIC" in
    mem|util|all) ;;
    *) echo "ERROR: METRIC must be one of: mem, util, all (got '$METRIC')" >&2; exit 2 ;;
esac

resolve_nodes() {
    if [[ -n "$NODES" ]]; then
        echo "$NODES"
        return
    fi
    local nodelist=""
    if [[ -n "$JOB" ]]; then
        nodelist=$(squeue -h -j "$JOB" -o '%N' 2>/dev/null | tr -d '[:space:]')
    fi
    if [[ -z "$nodelist" ]] && command -v squeue >/dev/null 2>&1; then
        nodelist=$(squeue --me -h -t R -o '%N' 2>/dev/null | head -1 | tr -d '[:space:]')
    fi
    if [[ -z "$nodelist" ]]; then
        echo "ERROR: no NODES env and no running Slurm job found. Set NODES=\"node-a node-b\"." >&2
        exit 2
    fi
    if command -v scontrol >/dev/null 2>&1; then
        scontrol show hostnames "$nodelist" 2>/dev/null | tr '\n' ' '
    else
        echo "$nodelist"
    fi
}

remote_cmd() {
    local args="--showuse --showmeminfo vram"
    if [[ -n "$CONTAINER" ]]; then
        echo "podman exec $CONTAINER rocm-smi $args 2>/dev/null \
              || docker exec $CONTAINER rocm-smi $args 2>/dev/null"
    else
        echo "rocm-smi $args 2>/dev/null"
    fi
}

# Sample one node -> lines: "<node> <gpu_idx> <util_pct> <used_bytes> <total_bytes>"
sample_node() {
    local node=$1
    local out
    # shellcheck disable=SC2029  # remote_cmd intentionally expands locally so $CONTAINER/$args are baked into the command string sent to ssh.
    out=$(ssh "${SSH_OPTS[@]}" "$node" "$(remote_cmd)" 2>/dev/null) || {
        echo "$node ERR ssh-or-rocm-smi-failed"
        return
    }
    awk -v node="$node" '
        match($0, /GPU\[([0-9]+)\][^:]*: GPU use \(%\): ([0-9]+)/, m) {
            util[m[1]] = m[2]
        }
        match($0, /GPU\[([0-9]+)\][^:]*: VRAM Total Memory \(B\): ([0-9]+)/, m) {
            total[m[1]] = m[2]
        }
        match($0, /GPU\[([0-9]+)\][^:]*: VRAM Total Used Memory \(B\): ([0-9]+)/, m) {
            used[m[1]] = m[2]
        }
        END {
            n = 0
            for (k in total) n++
            if (n == 0) {
                # No memory info; try util-only.
                for (k in util) n++
                if (n == 0) { print node " ERR no-data"; exit }
                for (i = 0; i < n; i++) {
                    printf "%s %d %s 0 0\n", node, i, (i in util ? util[i] : 0)
                }
                exit
            }
            for (i = 0; i < n; i++) {
                printf "%s %d %s %s %s\n", node, i,
                    (i in util ? util[i] : -1),
                    (i in used ? used[i] : 0),
                    total[i]
            }
        }
    ' <<<"$out"
}

# Format one cell based on METRIC.
fmt_cell() {
    local util=$1 used=$2 total=$3
    awk -v metric="$METRIC" -v u="$util" -v ub="$used" -v tb="$total" 'BEGIN{
        gib=1024*1024*1024
        memp = (tb+0==0) ? 0 : 100.0*ub/tb
        if (metric == "mem") {
            if (tb+0==0) { printf "%-17s", "n/a"; exit }
            printf "%5.1f/%-5.1f %4.1f%%", ub/gib, tb/gib, memp
        } else if (metric == "util") {
            if (u+0 < 0)  { printf "%-7s", "n/a"; exit }
            printf "%4d%%   ", u
        } else { # all
            uu = (u+0 < 0) ? "  n/a" : sprintf("%4d%%", u)
            mm = (tb+0==0) ? "  n/a" : sprintf("%4.1f%%", memp)
            printf "%s / %s", uu, mm
        }
    }'
}

# Fixed cell width (chars) per metric for header alignment.
cell_width() {
    case "$METRIC" in
        mem)  echo 17 ;;
        util) echo 7  ;;
        all)  echo 13 ;;
    esac
}

print_row() {
    local node=$1 ; shift
    printf "%-15s" "$node"
    while (($# >= 4)); do
        local idx=$1 util=$2 used=$3 total=$4 ; shift 4
        printf " | %s" "$(fmt_cell "$util" "$used" "$total")"
    done
    printf "\n"
}

NODES=$(resolve_nodes)

detect_max_gpus() {
    local n
    for n in $NODES; do
        local out
        out=$(sample_node "$n")
        if ! grep -q ' ERR ' <<<"$out"; then
            awk '{print $2}' <<<"$out" | sort -n | tail -1
            return
        fi
    done
    echo 7
}
MAX_IDX=$(detect_max_gpus)
W=$(cell_width)

run_once() {
    local ts
    ts=$(date '+%Y-%m-%d %H:%M:%S')

    declare -A pids=() outs=()
    local n
    for n in $NODES; do
        local f
        f=$(mktemp)
        outs[$n]=$f
        sample_node "$n" >"$f" &
        pids[$n]=$!
    done
    for n in $NODES; do
        wait "${pids[$n]}" 2>/dev/null || true
    done

    [[ "$ONCE" == "1" ]] || clear
    local label="container=${CONTAINER:-<host>}"
    local title
    case "$METRIC" in
        mem)  title="VRAM (used/total GiB, pct)" ;;
        util) title="GPU util %" ;;
        all)  title="util% / mem%" ;;
    esac
    echo "[${ts}]  rocm-smi $title  ${label}  every ${INTERVAL}s"
    printf "%-15s" "NODE"
    local i
    for ((i=0; i<=MAX_IDX; i++)); do
        printf " | %-${W}s" "GPU$i"
    done
    echo

    for n in $NODES; do
        local f="${outs[$n]}"
        if grep -q ' ERR ' "$f"; then
            local reason
            reason=$(awk '$2=="ERR"{print $3}' "$f" | head -1)
            printf "%-15s | %s\n" "$n" "ERROR: $reason"
            rm -f "$f"
            continue
        fi
        local args=()
        while read -r _node idx util used total; do
            args+=("$idx" "$util" "$used" "$total")
            if [[ -n "$CSV" ]]; then
                echo "$ts,$n,$idx,$util,$used,$total" >>"$CSV"
            fi
        done < <(sort -k2,2n "$f")
        print_row "$n" "${args[@]}"
        rm -f "$f"
    done
}

if [[ -n "$CSV" && ! -s "$CSV" ]]; then
    echo "timestamp,node,gpu,util_pct,used_bytes,total_bytes" >"$CSV"
fi

if [[ "$ONCE" == "1" ]]; then
    run_once
    exit 0
fi

while true; do
    run_once
    sleep "$INTERVAL"
done
