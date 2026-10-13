#!/bin/bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# One-time preparation, inside the container, as root (CPU only):
#
#   docker exec -e TURBO_DIR=/path/to/Primus-Turbo <container> \
#       bash examples/megatron/models/deepseek-v3/prepare_container_mi455x.sh
#
# 1. Force Triton's gfx1250 expert scheduling off (GFX1250 runbook v3 §1.2): with
#    it, Triton kernels fault or return NaN at random, and
#    TRITON_HIP_USE_EXPERT_SCHEDULING=0 misses torch inductor kernels. compiler.py
#    is patched; the original is kept as compiler.py.orig.
# 2. With TURBO_DIR: build that Primus-Turbo checkout in place for gfx1250, as its
#    owner and without /opt/openmpi (else rocSHMEM and MPI get linked in), and put
#    the flydsl its setup.py pins into $TURBO_DIR/build/pydeps. The image's own
#    primus-turbo and flydsl stay untouched; run_deepseek_v3_proxy_mi455x.sh puts
#    the checkout first. Rerun after pulling C++ changes into the checkout.
#
# Env: TURBO_DIR, FLYDSL_SPEC (requirement or wheel path; default: the pin in
#      $TURBO_DIR/setup.py, needs network), MAX_JOBS (the image sets 128)
###############################################################################
set -euo pipefail

# ---------- 1. Triton expert scheduling --------------------------------------
# is_expert_scheduling_enabled() honours TRITON_HIP_USE_EXPERT_SCHEDULING, so
# check with the variable unset.
expert_sched_off() {
    env -u TRITON_HIP_USE_EXPERT_SCHEDULING python3 -c \
        "import sys; from triton.backends.amd import compiler as c; sys.exit(c.is_expert_scheduling_enabled('gfx1250'))"
}
COMPILER_PY=$(python3 -c "import os, triton; print(os.path.join(os.path.dirname(triton.__file__), 'backends', 'amd', 'compiler.py'))")
if expert_sched_off; then
    echo "[prepare] Triton expert scheduling is already off for gfx1250 (${COMPILER_PY})"
elif [[ "$(id -u)" == 0 ]]; then
    [[ -e "${COMPILER_PY}.orig" ]] || cp "${COMPILER_PY}" "${COMPILER_PY}.orig"
    sed -i '/^def is_expert_scheduling_enabled(arch):/a\    return False' "${COMPILER_PY}"
    if ! expert_sched_off; then
        echo "[prepare] patching ${COMPILER_PY} did not take; is_expert_scheduling_enabled() changed?" >&2
        exit 1
    fi
    echo "[prepare] patched ${COMPILER_PY}: gfx1250 expert scheduling off"
else
    echo "[prepare] Triton enables gfx1250 expert scheduling; rerun as root (docker exec without -u)" \
        "to patch ${COMPILER_PY}" >&2
    exit 1
fi

[[ -n "${TURBO_DIR:-}" ]] || exit 0

# ---------- 2. Primus-Turbo checkout + its flydsl ----------------------------
TURBO_DIR="$(cd "${TURBO_DIR}" && pwd)"
[[ -f "${TURBO_DIR}/setup.py" ]] || { echo "[prepare] ${TURBO_DIR} has no setup.py" >&2; exit 1; }
FLYDSL_SPEC=${FLYDSL_SPEC:-$(grep -oE 'flydsl==[0-9][0-9.]*' "${TURBO_DIR}/setup.py" | head -n1 || true)}
OWNER_UID=$(stat -c %u "${TURBO_DIR}")
OWNER_GID=$(stat -c %g "${TURBO_DIR}")

build() {
    # `bash -c build` starts a new shell, which does not inherit the caller's set -e
    set -euo pipefail
    HOME="/tmp/primus-home-$(id -u)"
    export HOME
    mkdir -p "${HOME}"
    PATH=$(tr ':' '\n' <<< "${PATH}" | grep -v openmpi | paste -sd: || true)
    LD_LIBRARY_PATH=$(tr ':' '\n' <<< "${LD_LIBRARY_PATH:-}" | grep -v openmpi | paste -sd: || true)
    export PATH LD_LIBRARY_PATH
    cd "${TURBO_DIR}"
    echo "[prepare] building ${TURBO_DIR} ($(git rev-parse --short HEAD 2>/dev/null || echo '?')) for gfx1250"
    # explicitly: the image's GPU_ARCHS=native would probe the GPU
    GPU_ARCHS=gfx1250 python3 setup.py build_ext --inplace
    for so in primus_turbo/pytorch/_C*.so primus_turbo/lib/libprimus_turbo_kernels.so; do
        [[ -e "${so}" ]] || { echo "[prepare] the build produced no ${so}" >&2; return 1; }
    done
    if [[ -n "${FLYDSL_SPEC}" ]]; then
        echo "[prepare] installing ${FLYDSL_SPEC} into ${TURBO_DIR}/build/pydeps"
        python3 -m pip install --quiet --no-deps --upgrade --target "${TURBO_DIR}/build/pydeps" "${FLYDSL_SPEC}"
    fi
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${TURBO_DIR}:${TURBO_DIR}/build/pydeps${PYTHONPATH:+:${PYTHONPATH}}" \
        python3 -c "
import os, flydsl, primus_turbo
print('[prepare] primus_turbo', getattr(primus_turbo, '__version__', '?'), 'from', os.path.dirname(primus_turbo.__file__))
print('[prepare] flydsl', getattr(flydsl, '__version__', '?'), 'from', os.path.dirname(flydsl.__file__))
"
}
export -f build
# build() runs from TURBO_DIR, so make a wheel path given relative to the caller absolute
if [[ -n "${FLYDSL_SPEC}" && -f "${FLYDSL_SPEC}" ]]; then
    FLYDSL_SPEC="$(realpath "${FLYDSL_SPEC}")"
fi
export TURBO_DIR FLYDSL_SPEC
if [[ "$(id -u)" == 0 && "${OWNER_UID}" != 0 ]]; then
    setpriv --reuid "${OWNER_UID}" --regid "${OWNER_GID}" --clear-groups bash -c build
else
    bash -c build
fi
