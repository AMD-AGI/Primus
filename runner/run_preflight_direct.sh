#!/bin/bash
###############################################################################
# run_preflight_direct.sh
#
# Wrapper around `primus-cli direct -- preflight ...` that sets up the
# distributed environment variables (NNODES, NODE_RANK, MASTER_ADDR, ...)
# which `primus-cli-direct.sh` does NOT derive from SLURM on its own.
#
# Behavior:
#   - If running inside a SLURM allocation (SLURM_JOB_ID set), distributed
#     variables are derived from SLURM_* automatically.
#   - Otherwise, sensible single-node defaults are used; any pre-exported
#     NNODES / NODE_RANK / MASTER_ADDR / MASTER_PORT / GPUS_PER_NODE values
#     are preserved.
#   - NCCL/GLOO socket interface and IB HCAs default to the values used on
#     this cluster but can be overridden via environment.
#   - Extra arguments to this script are forwarded verbatim to `preflight`.
#
# Wrapper-only flags (consumed here, NOT forwarded to preflight):
#   --silent       Suppress all stdout from the wrapper and the underlying
#                  primus-cli/preflight run. Stderr is preserved so real
#                  errors still surface. Exit code is propagated as usual.
#                  The final report file path is still printed at the end
#                  even under --silent.
#
# Usage:
#   ./run_preflight_direct.sh                      # default report name
#   ./run_preflight_direct.sh --report-file-name myrun
#   ./run_preflight_direct.sh --silent --report-file-name myrun
#   NCCL_SOCKET_IFNAME=eth0 ./run_preflight_direct.sh
#   srun -N4 --ntasks-per-node=1 ./run_preflight_direct.sh
###############################################################################

set -euo pipefail

###############################################################################
# Parse wrapper-only flags (must run BEFORE any output / logging)
###############################################################################
SILENT=0
_passthrough_args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --silent)
            SILENT=1
            shift
            ;;
        --)
            shift
            _passthrough_args+=("$@")
            break
            ;;
        *)
            _passthrough_args+=("$1")
            shift
            ;;
    esac
done
set -- "${_passthrough_args[@]}"

# Save the original stdout on fd 3 so that, even under --silent, we can still
# print a small summary (e.g. the report file path) at the end of the run.
exec 3>&1

# Apply silent mode by redirecting stdout to /dev/null for the rest of the
# script (and, transitively, for the launched primus-cli command).
if [[ "$SILENT" == "1" ]]; then
    exec >/dev/null
fi

# Always-visible logger: writes to the saved original stdout (fd 3).
log_always() { echo "[run_preflight_direct] $*" >&3; }

###############################################################################
# Paths and helpers
###############################################################################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRIMUS_CLI="${PRIMUS_CLI:-$SCRIPT_DIR/primus-cli}"

log() { echo "[run_preflight_direct] $*"; }
die() { echo "[run_preflight_direct][ERROR] $*" >&2; exit 1; }

# Invoked indirectly via `trap ... ERR` below; ShellCheck SC2317 cannot
# follow the trap handler so it flags the body as unreachable -- silence it.
# shellcheck disable=SC2317
on_error() {
    local ec=$?
    local line=${1:-?}
    echo "[run_preflight_direct][ERROR] failed at line ${line} (exit=${ec})" >&2
    exit "$ec"
}
trap 'on_error $LINENO' ERR

###############################################################################
# Activate Python virtualenv
###############################################################################
if [[ -z "${VENV_ACTIVATE:-}" ]]; then
    die "VENV_ACTIVATE is not set. Please provide the path to your Python virtualenv activation script (e.g., VENV_ACTIVATE=/path/to/venv/bin/activate)."
fi
if [[ ! -f "$VENV_ACTIVATE" ]]; then
    die "Virtualenv activate script not found at: $VENV_ACTIVATE"
fi
# shellcheck disable=SC1090
source "$VENV_ACTIVATE"
log "Activated virtualenv: $VENV_ACTIVATE"

###############################################################################
# NCCL / network defaults
#
# primus-cli direct sources runner/helpers/envs/base_env.sh, which already
# applies sensible defaults and auto-detects NCCL_IB_HCA / NCCL_SOCKET_IFNAME
# using ${VAR:-default}, so any value exported here (or in the parent shell)
# wins. Uncomment / customize if you need to override the defaults.
###############################################################################
# export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-tw-eth3}"
# export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-$NCCL_SOCKET_IFNAME}"
# export NCCL_IB_HCA="${NCCL_IB_HCA:-rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7}"
# export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
# export NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-1}"
# export NCCL_PXN_DISABLE="${NCCL_PXN_DISABLE:-0}"
# export NCCL_DEBUG=INFO   # uncomment for verbose NCCL logging

###############################################################################
# Distributed environment: derive from SLURM if available, else defaults
###############################################################################
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
export MASTER_PORT="${MASTER_PORT:-1234}"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    if ! command -v scontrol >/dev/null 2>&1; then
        die "SLURM_JOB_ID is set but 'scontrol' is not on PATH"
    fi
    if [[ -z "${SLURM_NODELIST:-}" ]]; then
        die "SLURM_JOB_ID is set but SLURM_NODELIST is empty"
    fi

    export NNODES="${SLURM_NNODES:-${SLURM_JOB_NUM_NODES:-${NNODES:-1}}}"
    export NODE_RANK="${SLURM_NODEID:-${SLURM_PROCID:-${NODE_RANK:-0}}}"

    # Resolve master address from the first hostname in the nodelist unless
    # explicitly provided (and not the default 'localhost').
    if [[ -z "${MASTER_ADDR:-}" || "${MASTER_ADDR}" == "localhost" ]]; then
        if ! MASTER_ADDR="$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)" \
           || [[ -z "$MASTER_ADDR" ]]; then
            die "Failed to resolve MASTER_ADDR from SLURM_NODELIST=$SLURM_NODELIST"
        fi
        export MASTER_ADDR
    fi

    log "SLURM detected: JOB_ID=$SLURM_JOB_ID NODELIST=$SLURM_NODELIST"
else
    export NNODES="${NNODES:-1}"
    export NODE_RANK="${NODE_RANK:-0}"
    export MASTER_ADDR="${MASTER_ADDR:-localhost}"
    log "No SLURM allocation detected; using NNODES=$NNODES NODE_RANK=$NODE_RANK"
fi

# Sanity checks
[[ "$NNODES"      =~ ^[1-9][0-9]*$ ]] || die "NNODES must be a positive integer (got '$NNODES')"
[[ "$NODE_RANK"   =~ ^[0-9]+$ ]]      || die "NODE_RANK must be a non-negative integer (got '$NODE_RANK')"
[[ "$MASTER_PORT" =~ ^[0-9]+$ ]]      || die "MASTER_PORT must be a non-negative integer (got '$MASTER_PORT')"
[[ -n "$MASTER_ADDR" ]]               || die "MASTER_ADDR is empty"
if (( NODE_RANK >= NNODES )); then
    die "NODE_RANK ($NODE_RANK) must be < NNODES ($NNODES)"
fi

###############################################################################
# Build preflight invocation
###############################################################################
[[ -x "$PRIMUS_CLI" ]] || die "primus-cli not found or not executable: $PRIMUS_CLI"

# Auto-generated unique report name (used unless the user explicitly overrides
# via --report-file-name). The timestamp + node count guarantees that each run
# writes to a fresh, never-before-used file path, which prevents the wrapper
# from announcing stale leftovers from earlier runs as if they were fresh.
DEFAULT_REPORT_NAME="preflight-${NNODES}N-$(date +%Y%m%d-%H%M%S)"
PREFLIGHT_ARGS=("$@")

# Scan PREFLIGHT_ARGS for any user-provided --dump-path / --report-file-name
# so we can (a) inject our auto-generated unique name when missing, and
# (b) know exactly where to look for output files when announcing them.
DUMP_PATH=""
REPORT_NAME=""
_i=0
while (( _i < ${#PREFLIGHT_ARGS[@]} )); do
    case "${PREFLIGHT_ARGS[$_i]}" in
        --dump-path)          DUMP_PATH="${PREFLIGHT_ARGS[$((_i+1))]:-}";    _i=$((_i+2)) ;;
        --dump-path=*)        DUMP_PATH="${PREFLIGHT_ARGS[$_i]#--dump-path=}"; _i=$((_i+1)) ;;
        --report-file-name)   REPORT_NAME="${PREFLIGHT_ARGS[$((_i+1))]:-}";  _i=$((_i+2)) ;;
        --report-file-name=*) REPORT_NAME="${PREFLIGHT_ARGS[$_i]#--report-file-name=}"; _i=$((_i+1)) ;;
        *)                    _i=$((_i+1)) ;;
    esac
done

# Inject auto-generated unique name if user didn't supply one. This is the
# critical fix: without it, multiple runs collide on preflight's default name
# 'preflight_report', causing the wrapper to announce stale files from
# previous runs (e.g. an old PDF still on disk after --disable-pdf, or an
# old info report still present after --perf-test).
if [[ -z "$REPORT_NAME" ]]; then
    REPORT_NAME="$DEFAULT_REPORT_NAME"
    PREFLIGHT_ARGS+=(--report-file-name "$REPORT_NAME")
fi
# Match preflight's own default for announcement purposes if not user-set.
DUMP_PATH="${DUMP_PATH:-output/preflight}"

###############################################################################
# Print summary (rank 0 only) and execute
###############################################################################
if [[ "${NODE_RANK}" == "0" ]]; then
    log "------------------------------------------------------------"
    log "Host             : $(hostname)"
    log "PRIMUS_CLI       : $PRIMUS_CLI"
    log "NNODES           : $NNODES"
    log "NODE_RANK        : $NODE_RANK"
    log "GPUS_PER_NODE    : $GPUS_PER_NODE"
    log "MASTER_ADDR      : $MASTER_ADDR"
    log "MASTER_PORT      : $MASTER_PORT"
    log "preflight args   : ${PREFLIGHT_ARGS[*]}"
    log "------------------------------------------------------------"
fi

###############################################################################
# Run preflight, then announce the report file path(s) on the original stdout
###############################################################################
# Don't leak our private fd 3 into the child process; preflight has no need
# for it and closing it keeps `lsof` / `/proc/<pid>/fd` output clean.
ec=0
"$PRIMUS_CLI" direct -- preflight "${PREFLIGHT_ARGS[@]}" 3>&- || ec=$?

# Only rank 0 writes the preflight report, so only rank 0 announces it.
if [[ "$NODE_RANK" == "0" ]]; then
    _printed=0
    for _suffix in "" "_perf"; do
        for _ext in md pdf; do
            _f="${DUMP_PATH}/${REPORT_NAME}${_suffix}.${_ext}"
            if [[ -f "$_f" ]]; then
                # Resolve to absolute path so the user can copy/paste it directly.
                _abs="$(readlink -f -- "$_f" 2>/dev/null || echo "$_f")"
                log_always "Report: $_abs"
                _printed=1
            fi
        done
    done
    if [[ "$_printed" -eq 0 ]]; then
        log_always "WARN: no report files found at ${DUMP_PATH}/${REPORT_NAME}{,_perf}.{md,pdf}"
    fi
fi

# Close fd 3 cleanly before exit (cosmetic; the kernel does this anyway).
exec 3>&-
exit "$ec"
