#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# validate_baremetal.sh -- the release-docs skill's bare-metal check: install the
# environment from this checkout's tools/installation*/setup.sh into a fresh
# base, compare the installed packages with the image snapshot, and train a few
# configs for a handful of steps with tools/perf/run_batch.sh in direct mode.
#
# Run it on a GPU node, from the Primus checkout the release documents (the
# release branch with the docs branch's install files), never on a login node.
#
# Usage:
#   export HF_TOKEN=...                       # or HF_TOKEN_FILE=<file>
#   bash tools/release_docs/validate_baremetal.sh --family primus \
#       --base /tmp/primus-baremetal-v26.9 --result-dir ~/baremetal-v26.9 \
#       --config examples/megatron/configs/MI325X/llama3.1_8B-BF16-pretrain.yaml \
#       --config examples/torchtitan/configs/MI325X/llama3.1_8B-BF16-pretrain.yaml
#
# Options:
#   --family primus|jax   which install (tools/installation or tools/installation-jax)
#   --base DIR            PRIMUS_BASE / PRIMUS_JAX_BASE; must not exist yet (a fresh
#                         install is the point), unless --reuse
#   --result-dir DIR      logs and summary.json (default: <base>-results)
#   --config PATH         a config to train (repeatable), relative to the checkout;
#                         the backend is read from examples/<backend>/...
#   --gpu NAME            catalog GPU key (default: from the first config's path)
#   --train-steps N       steps per config (default 10)
#   --min-free-gb N       refuse when the base's filesystem has less (default 200)
#   --reuse               install into an existing base (resume after a failure)
#   --skip-install        only train, with an existing base
#   --dry-run             no install, no training: print the plan (run_batch.sh DRY_RUN=1)
#
# Outputs in --result-dir:
#   install.log, install-warnings.txt   the full setup.sh log and its warning lines
#   parity.json                         installed packages vs the image snapshot
#   train/                              run_batch.sh logs, results.csv / results.json
#   summary.json                        everything above in one record
#
# Exit codes: 0 install and every config passed, 1 something failed, 2 bad input.
#
# Lessons from the v26.7 run, built in: the environment is cleaned before the
# install (VENV_DIR and friends are inherited and silently redirect it), the base
# is checked for space first, and the log is grepped for warnings, not tailed.

set -uo pipefail

# Launch with a clean environment: another stack's env.sh exports these, and
# setup.sh then installs into the wrong venv.
if [ -z "${_VALIDATE_BAREMETAL_CLEAN:-}" ]; then
    exec env -u VENV_DIR -u WORKSPACE_DIR -u UV_PYTHON_INSTALL_DIR -u PRIMUS_PIP_CONSTRAINTS \
        -u VIRTUAL_ENV -u PRIMUS_BASE -u PRIMUS_JAX_BASE _VALIDATE_BAREMETAL_CLEAN=1 bash "$0" "$@"
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FAMILY="" BASE="" RESULT_DIR="" GPU="" TRAIN_STEPS=10 MIN_FREE_GB=200 REUSE=0 SKIP_INSTALL=0 DRY_RUN=0
CONFIGS=()

die() { echo "[validate][ERROR] $*" >&2; exit 2; }
log() { echo "[validate] $(date -u +%FT%TZ) $*"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --family) FAMILY="$2"; shift 2 ;;
        --base) BASE="$2"; shift 2 ;;
        --result-dir) RESULT_DIR="$2"; shift 2 ;;
        --config) CONFIGS+=("$2"); shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --train-steps) TRAIN_STEPS="$2"; shift 2 ;;
        --min-free-gb) MIN_FREE_GB="$2"; shift 2 ;;
        --reuse) REUSE=1; shift ;;
        --skip-install) SKIP_INSTALL=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) sed -n '9,45p' "$0"; exit 0 ;;
        *) die "unknown option $1" ;;
    esac
done

case "$FAMILY" in
    primus) INSTALL_DIR="$ROOT/tools/installation"; BASE_VAR=PRIMUS_BASE ;;
    jax) INSTALL_DIR="$ROOT/tools/installation-jax"; BASE_VAR=PRIMUS_JAX_BASE ;;
    *) die "--family must be primus or jax" ;;
esac
[ -n "$BASE" ] || die "--base is required"
case "$BASE" in /*) ;; *) die "--base must be an absolute path" ;; esac
RESULT_DIR="${RESULT_DIR:-$BASE-results}"
[ "${#CONFIGS[@]}" -gt 0 ] || die "at least one --config is required"
for cfg in "${CONFIGS[@]}"; do
    [ -f "$ROOT/$cfg" ] || die "config not found in this checkout: $cfg"
done
if [ -z "$GPU" ]; then
    GPU=$(sed -n 's#.*/configs/\([^/]*\)/.*#\1#p' <<< "${CONFIGS[0]}")
    [ -n "$GPU" ] || die "cannot read the GPU from ${CONFIGS[0]}; pass --gpu"
fi
if [ -z "${HF_TOKEN:-}" ] && [ -n "${HF_TOKEN_FILE:-}" ] && [ -s "$HF_TOKEN_FILE" ]; then
    HF_TOKEN="$(tr -d '[:space:]' < "$HF_TOKEN_FILE")"
    export HF_TOKEN
fi
[ -n "${HF_TOKEN:-}" ] || die "HF_TOKEN (or HF_TOKEN_FILE) is required for the training runs"

mkdir -p "$RESULT_DIR" || die "cannot create $RESULT_DIR"
SUMMARY="$RESULT_DIR/summary.json"
INSTALL_RC="" WARNINGS=0 TRAIN_RC=""

write_summary() {
    python3 - "$SUMMARY" "$FAMILY" "$BASE" "$GPU" "$TRAIN_STEPS" "${INSTALL_RC:-}" "$WARNINGS" \
        "${TRAIN_RC:-}" "$RESULT_DIR" "$1" "${CONFIGS[@]}" <<'EOF'
import json, os, sys
path, family, base, gpu, steps, install_rc, warnings, train_rc, result_dir, stage, *configs = sys.argv[1:]
parity = os.path.join(result_dir, "parity.json")
summary = {
    "family": family,
    "base": base,
    "gpu": gpu,
    "train_steps": int(steps),
    "stage": stage,
    "install_rc": int(install_rc) if install_rc else None,
    "install_warnings": int(warnings),
    "parity": json.load(open(parity)) if os.path.exists(parity) else None,
    "train_rc": int(train_rc) if train_rc else None,
    "train_dir": os.path.join(result_dir, "train"),
    "configs": configs,
}
with open(path + ".tmp", "w") as f:
    json.dump(summary, f, indent=2)
os.replace(path + ".tmp", path)
EOF
}

# ---- install -------------------------------------------------------------------
if [ "$DRY_RUN" = 1 ]; then
    log "dry run: no install"
elif [ "$SKIP_INSTALL" = 0 ]; then
    if [ -e "$BASE" ] && [ "$REUSE" = 0 ] && [ -n "$(ls -A "$BASE" 2>/dev/null)" ]; then
        die "$BASE exists and is not empty; a fresh install is the point (--reuse to resume)"
    fi
    mkdir -p "$BASE" || die "cannot create $BASE"
    free_gb=$(df -P -BG "$BASE" | awk 'NR==2 {gsub("G","",$4); print $4}')
    if [ "${free_gb:-0}" -lt "$MIN_FREE_GB" ]; then
        die "only ${free_gb}G free under $BASE (need $MIN_FREE_GB; --min-free-gb to change)"
    fi
    log "installing $FAMILY into $BASE (${free_gb}G free)"
    write_summary installing
    export "$BASE_VAR=$BASE"
    bash "$INSTALL_DIR/setup.sh" > "$RESULT_DIR/install.log" 2>&1
    INSTALL_RC=$?
    grep -inE '\bwarn(ing)?\b' "$RESULT_DIR/install.log" > "$RESULT_DIR/install-warnings.txt"
    WARNINGS=$(wc -l < "$RESULT_DIR/install-warnings.txt")
    log "install exit $INSTALL_RC, $WARNINGS warning line(s)"
    if [ "$INSTALL_RC" != 0 ]; then
        write_summary install_failed
        tail -40 "$RESULT_DIR/install.log" >&2
        exit 1
    fi
else
    [ -d "$BASE" ] || die "--skip-install needs an existing $BASE"
    INSTALL_RC=0
fi

# ---- installed packages vs the image ------------------------------------------
manifest="$BASE/workspace/.manifest/requirements.txt"
if [ -f "$manifest" ]; then
    python3 - "$ROOT" "$FAMILY" "$manifest" "$RESULT_DIR/parity.json" <<'EOF'
import json, sys
root, family, manifest, out = sys.argv[1:]
sys.path.insert(0, f"{root}/tools/release_docs")
import install_parity as P
release = P.mirrored_release(family)
diff, error = P.compare_manifest(family, release, manifest)
result = {"release": release, "error": error}
if diff:
    result.update(
        differing=[{"package": n, "image": i, "metal": m} for n, i, m in diff["differing"]],
        only_in_image=diff["only_in_image"],
        only_on_metal=diff["only_on_metal"],
        common=diff["common"],
    )
json.dump(result, open(out, "w"), indent=2)
print(f"[validate] packages vs {release} snapshot: "
      + (error or f"{len(diff['differing'])} differing, {len(diff['only_in_image'])} only in the image, "
                  f"{len(diff['only_on_metal'])} only on bare metal"))
EOF
else
    log "no install manifest at $manifest (stage_manifest did not run)"
fi

# ---- training --------------------------------------------------------------------
catalog="$RESULT_DIR/catalog.yaml"
{
    echo "$GPU:"
    for backend in $(printf '%s\n' "${CONFIGS[@]}" | sed -n 's#^examples/\([^/]*\)/.*#\1#p' | sort -u); do
        echo "  $backend:"
        for cfg in "${CONFIGS[@]}"; do
            case "$cfg" in examples/"$backend"/*) echo "    - $cfg" ;; esac
        done
    done
} > "$catalog"
backend_list=$(printf '%s\n' "${CONFIGS[@]}" | sed -n 's#^examples/\([^/]*\)/.*#\1#p' | sort -u | paste -sd, -)

log "training ${#CONFIGS[@]} config(s) for $TRAIN_STEPS steps (backends $backend_list)"
write_summary training
(
    export "$BASE_VAR=$BASE"
    if [ "$DRY_RUN" = 1 ]; then
        export DRY_RUN=1
    else
        # shellcheck disable=SC1091
        source "$INSTALL_DIR/env.sh" || exit 3
    fi
    cd "$ROOT" || exit 3
    PRIMUS_MODE=direct CONFIG_FILE="$catalog" BACKEND="$backend_list" GPU="$GPU" \
        TRAIN_STEPS="$TRAIN_STEPS" NUM_REPS=1 RESULT_DIR="$RESULT_DIR/train" \
        DOCKER_IMAGE="bare-metal:$FAMILY" REPORT=0 \
        bash tools/perf/run_batch.sh
) > "$RESULT_DIR/train.log" 2>&1
TRAIN_RC=$?
log "training exit $TRAIN_RC (logs in $RESULT_DIR/train)"
write_summary finished
[ "$TRAIN_RC" = 0 ] || exit 1
exit 0
