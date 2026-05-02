#!/bin/bash
###############################################################################
# run_node_smoke_direct.sh
#
# Wrapper around `python -m primus.tools.preflight.node_smoke` that runs the
# node-local preflight smoke test on EACH node in parallel with NO global
# rendezvous, then aggregates per-node JSON verdicts on rank 0.
#
# How to use
# ----------
#
#   # Inside an existing SLURM allocation (the normal case):
#   srun -N "$SLURM_NNODES" --ntasks-per-node=1 ./run_node_smoke_direct.sh
#
#   # Single-node local check (no SLURM):
#   ./run_node_smoke_direct.sh
#
# Behavior
# --------
#   - One Python process per node (use --ntasks-per-node=1 with srun).
#   - Each node writes <dump>/smoke/<host>.json. No torch.distributed across
#     nodes is ever initialized -- a bad node cannot stall the rest.
#   - When NODE_RANK == 0 (or in single-node mode), this script also runs the
#     aggregator after its own per-node check completes. The aggregator polls
#     <dump>/smoke for the expected number of JSONs (with a timeout) and
#     emits:
#         <dump>/smoke_report.md
#         <dump>/passing_nodes.txt   (newline-separated, ready for SLURM)
#         <dump>/failing_nodes.txt   (newline-separated, ready for SLURM)
#
# Wrapper-only flags (consumed here, NOT forwarded to node_smoke):
#   --silent                 Suppress wrapper stdout (real errors still go to
#                            stderr, exit code preserved). Final report path is
#                            still printed under --silent.
#   --aggregate-only         Only run the aggregator on this node; skip the
#                            local per-node smoke step.
#   --no-aggregate           Skip the aggregator on rank 0 (useful if you want
#                            to aggregate later from a separate command).
#   --wait-timeout-sec SEC   Override aggregator wait timeout (default 60).
#
# Anything else is forwarded verbatim to `node_smoke run` (e.g. --tier2-perf,
# --gemm-tflops-min, --dump-path, --expected-gpus, ...).
###############################################################################

set -euo pipefail

###############################################################################
# Parse wrapper-only flags
###############################################################################
SILENT=0
AGGREGATE_ONLY=0
RUN_AGGREGATE=1
WAIT_TIMEOUT_SEC=60
_passthrough_args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --silent)            SILENT=1; shift ;;
        --aggregate-only)    AGGREGATE_ONLY=1; shift ;;
        --no-aggregate)      RUN_AGGREGATE=0; shift ;;
        --wait-timeout-sec)  WAIT_TIMEOUT_SEC="${2:-60}"; shift 2 ;;
        --wait-timeout-sec=*) WAIT_TIMEOUT_SEC="${1#--wait-timeout-sec=}"; shift ;;
        --)                  shift; _passthrough_args+=("$@"); break ;;
        *)                   _passthrough_args+=("$1"); shift ;;
    esac
done
set -- "${_passthrough_args[@]}"

# Save original stdout on fd 3 so --silent can still print a final summary.
exec 3>&1
if [[ "$SILENT" == "1" ]]; then
    exec >/dev/null
fi

log_always() { echo "[run_node_smoke_direct] $*" >&3; }

###############################################################################
# Paths and helpers
###############################################################################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log() { echo "[run_node_smoke_direct] $*"; }
die() { echo "[run_node_smoke_direct][ERROR] $*" >&2; exit 1; }

on_error() {
    local ec=$?
    local line=${1:-?}
    echo "[run_node_smoke_direct][ERROR] failed at line ${line} (exit=${ec})" >&2
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
# Discover SLURM topology (we do NOT need MASTER_ADDR/MASTER_PORT here -- the
# node-local smoke test never opens an inter-node process group).
###############################################################################
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    NNODES="${SLURM_NNODES:-${SLURM_JOB_NUM_NODES:-${NNODES:-1}}}"
    NODE_RANK="${SLURM_NODEID:-${SLURM_PROCID:-${NODE_RANK:-0}}}"
    log "SLURM detected: JOB_ID=$SLURM_JOB_ID NNODES=$NNODES NODE_RANK=$NODE_RANK"
else
    NNODES="${NNODES:-1}"
    NODE_RANK="${NODE_RANK:-0}"
    log "No SLURM allocation detected; NNODES=$NNODES NODE_RANK=$NODE_RANK"
fi
export NNODES NODE_RANK

[[ "$NNODES"    =~ ^[1-9][0-9]*$ ]] || die "NNODES must be a positive integer (got '$NNODES')"
[[ "$NODE_RANK" =~ ^[0-9]+$ ]]      || die "NODE_RANK must be a non-negative integer (got '$NODE_RANK')"

###############################################################################
# Determine --dump-path so we can hand it to the aggregator. If the user did
# not pass it, mirror node_smoke's default (output/preflight).
###############################################################################
DUMP_PATH=""
SMOKE_ARGS=("$@")
_i=0
while (( _i < ${#SMOKE_ARGS[@]} )); do
    case "${SMOKE_ARGS[$_i]}" in
        --dump-path)   DUMP_PATH="${SMOKE_ARGS[$((_i+1))]:-}"; _i=$((_i+2)) ;;
        --dump-path=*) DUMP_PATH="${SMOKE_ARGS[$_i]#--dump-path=}"; _i=$((_i+1)) ;;
        *)             _i=$((_i+1)) ;;
    esac
done
DUMP_PATH="${DUMP_PATH:-output/preflight}"
mkdir -p "${DUMP_PATH}/smoke"

###############################################################################
# Per-node smoke run (skipped under --aggregate-only)
###############################################################################
ec_run=0
if [[ "$AGGREGATE_ONLY" == "0" ]]; then
    if [[ "${NODE_RANK}" == "0" ]]; then
        log "------------------------------------------------------------"
        log "Host             : $(hostname)"
        log "NNODES           : $NNODES"
        log "NODE_RANK        : $NODE_RANK"
        log "DUMP_PATH        : $DUMP_PATH"
        log "smoke args       : ${SMOKE_ARGS[*]:-}"
        log "------------------------------------------------------------"
    fi
    python -m primus.tools.preflight.node_smoke run "${SMOKE_ARGS[@]}" 3>&- || ec_run=$?
fi

###############################################################################
# Aggregator (rank 0 only, unless --no-aggregate)
###############################################################################
ec_agg=0
if [[ "${NODE_RANK}" == "0" && "$RUN_AGGREGATE" == "1" ]]; then
    # Resolve SLURM's compressed nodelist (e.g. "tus1-p3-g[14,15,25]") into
    # one short hostname per line, so the aggregator can name nodes that
    # never produced a JSON (e.g. tasks marked "unknown" by srun) instead of
    # falling back to <missing-N> placeholders. Best effort: if scontrol is
    # unavailable or fails (non-SLURM run, container without slurm-client),
    # we silently fall back to the count-only behaviour.
    EXPECTED_NODELIST_FILE=""
    if [[ -n "${SLURM_JOB_NODELIST:-}" ]] && command -v scontrol >/dev/null 2>&1; then
        _candidate="${DUMP_PATH}/expected_nodes.txt"
        if scontrol show hostnames "$SLURM_JOB_NODELIST" > "$_candidate" 2>/dev/null \
                && [[ -s "$_candidate" ]]; then
            EXPECTED_NODELIST_FILE="$_candidate"
            log "Resolved expected nodelist ($(wc -l < "$EXPECTED_NODELIST_FILE") nodes) -> $EXPECTED_NODELIST_FILE"
        else
            log "WARN: 'scontrol show hostnames' failed; aggregator will use <missing-N> for nodes that never reported"
            rm -f "$_candidate"
        fi
    fi

    log "Aggregating: expected_nodes=$NNODES wait_timeout_sec=$WAIT_TIMEOUT_SEC"
    AGG_ARGS=(
        --dump-path "$DUMP_PATH"
        --expected-nodes "$NNODES"
        --wait-timeout-sec "$WAIT_TIMEOUT_SEC"
    )
    if [[ -n "$EXPECTED_NODELIST_FILE" ]]; then
        AGG_ARGS+=(--expected-nodelist-file "$EXPECTED_NODELIST_FILE")
    fi
    python -m primus.tools.preflight.node_smoke aggregate "${AGG_ARGS[@]}" 3>&- || ec_agg=$?

    REPORT_MD="${DUMP_PATH}/smoke_report.md"
    PASS_TXT="${DUMP_PATH}/passing_nodes.txt"
    FAIL_TXT="${DUMP_PATH}/failing_nodes.txt"
    [[ -f "$REPORT_MD" ]] && log_always "Report:  $(readlink -f -- "$REPORT_MD" 2>/dev/null || echo "$REPORT_MD")"
    [[ -f "$PASS_TXT"  ]] && log_always "Passing: $(readlink -f -- "$PASS_TXT"  2>/dev/null || echo "$PASS_TXT")"
    [[ -f "$FAIL_TXT"  ]] && log_always "Failing: $(readlink -f -- "$FAIL_TXT"  2>/dev/null || echo "$FAIL_TXT")"
fi

# Close fd 3 cleanly before exit (cosmetic).
exec 3>&-

# Aggregator's exit code wins on rank 0 (it knows about MISSING nodes).
# Other ranks just propagate their own per-node exit code.
if [[ "${NODE_RANK}" == "0" && "$RUN_AGGREGATE" == "1" ]]; then
    if (( ec_agg != 0 )); then exit "$ec_agg"; fi
fi
exit "$ec_run"
