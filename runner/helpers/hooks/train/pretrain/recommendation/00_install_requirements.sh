#!/bin/bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Recommendation (DLRMv4) additions to the rocm/primus image, once per container:
#   - requirements-recommendation.txt and the MLPerf logging pin, all with --no-deps;
#   - requirements-recommendation-private.txt into $PRIMUS_RECOMMENDATION_SITE, a
#     directory only the trainer puts first on its path (versions that differ from
#     the image's, e.g. FlyDSL);
#   - /opt/rocm -> $ROCM_PATH: the trainer builds HIP kernels at run time with
#     /opt/rocm/bin/hipcc, and TheRock images keep ROCm inside the venv.
# Fails if torch, FBGEMM or torchrec changed (the recipe runs on the image's stack).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ_FILE="${SCRIPT_DIR}/requirements-recommendation.txt"
PRIVATE_REQ_FILE="${SCRIPT_DIR}/requirements-recommendation-private.txt"
# Set and empty: no private packages (the trainer then uses the image's versions).
PRIVATE_SITE="${PRIMUS_RECOMMENDATION_SITE-/opt/primus-recommendation/site-packages}"
MLPERF_LOGGING_REF="${MLPERF_LOGGING_REF:-6.1.0-rc2}"

if [[ ! -e /opt/rocm/bin/hipcc ]]; then
    rocm="${ROCM_PATH:-}"
    if [[ -z "$rocm" || ! -x "$rocm/bin/hipcc" ]]; then
        echo "[ERROR] recommendation: no /opt/rocm/bin/hipcc and no \$ROCM_PATH/bin/hipcc (ROCM_PATH='$rocm')" >&2
        exit 1
    fi
    if [[ -e /opt/rocm || -L /opt/rocm ]]; then
        echo "[ERROR] recommendation: /opt/rocm exists but has no bin/hipcc; expected a link to $rocm" >&2
        exit 1
    fi
    ln -s "$rocm" /opt/rocm
    echo "[INFO] recommendation: linked /opt/rocm -> $rocm"
fi

if [[ "${PRIMUS_SKIP_PIP:-0}" == "1" ]]; then
    echo "[INFO] PRIMUS_SKIP_PIP=1: skipping DLRMv4 dependency installation"
    exit 0
fi

# The private directory: one marker per requirements file content, so an edit reinstalls.
private_marker="${PRIVATE_SITE}/.installed-$(sha256sum "$PRIVATE_REQ_FILE" | cut -c1-16)"
if [[ -z "$PRIVATE_SITE" ]]; then
    echo "[INFO] recommendation: PRIMUS_RECOMMENDATION_SITE is empty: no private packages"
elif [[ -f "$private_marker" ]]; then
    echo "[INFO] recommendation: private packages already in $PRIVATE_SITE"
else
    echo "[+] Installing private recommendation packages into $PRIVATE_SITE (--no-deps)..."
    # Replace only a directory this hook made (it holds a marker) or an empty one.
    if [[ -d "$PRIVATE_SITE" && -n "$(ls -A "$PRIVATE_SITE")" ]]; then
        if ! compgen -G "${PRIVATE_SITE}/.installed-*" >/dev/null; then
            echo "[ERROR] recommendation: $PRIVATE_SITE is not empty and was not made by this hook" >&2
            exit 1
        fi
        rm -rf "$PRIVATE_SITE"
    fi
    mkdir -p "$PRIVATE_SITE"
    pip install --no-deps --target "$PRIVATE_SITE" -r "$PRIVATE_REQ_FILE"
    touch "$private_marker"
fi

# Installed already (a persistent container, or an image that has them): skip pip.
if python3 - "$MLPERF_LOGGING_REF" <<'PY'
import importlib.metadata as m, sys
want = {"polars-u64-idx": "1.33.1", "gin-config": "0.5.0"}
try:
    ok = all(m.version(name) == version for name, version in want.items())
    url = m.distribution("mlperf-logging").read_text("direct_url.json") or ""
    ok = ok and sys.argv[1] in url
except m.PackageNotFoundError:
    ok = False
sys.exit(0 if ok else 1)
PY
then
    echo "[INFO] recommendation: dependencies already installed"
    exit 0
fi

stack() {
    python3 -c 'import importlib.metadata as m
for n in ("torch", "torchrec", "fbgemm-gpu", "fbgemm-gpu-nightly-rocm", "triton", "flydsl"):
    try: print(n, m.version(n))
    except m.PackageNotFoundError: pass'
}
before=$(stack)
echo "[+] Installing recommendation dependencies (--no-deps)..."
pip install --no-deps -r "$REQ_FILE"
pip install --no-deps --force-reinstall "git+https://github.com/mlcommons/logging.git@${MLPERF_LOGGING_REF}"
after=$(stack)
if [[ "$before" != "$after" ]]; then
    echo "[ERROR] recommendation: the install changed the image's stack:" >&2
    diff <(echo "$before") <(echo "$after") >&2 || true
    exit 1
fi
echo "[OK] recommendation dependencies installed"
