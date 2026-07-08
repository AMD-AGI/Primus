#!/bin/bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
# Check that Primus requirements.txt packages are installed.
# Warns on missing packages by default. Set PRIMUS_STRICT_REQUIREMENTS=1 to
# make it a fatal error.
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRIMUS_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
REQ_FILE="${PRIMUS_ROOT}/requirements.txt"

if [[ ! -f "$REQ_FILE" ]]; then
    exit 0
fi

MISSING=$(python3 - "$REQ_FILE" << 'PYEOF'
import importlib.metadata, re, sys, pathlib

req_file = pathlib.Path(sys.argv[1])
missing = []

for line in req_file.read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#") or line.startswith("-"):
        continue
    pkg = re.split(r"[><=!;\[\s]", line)[0].strip()
    if not pkg:
        continue
    normalized = re.sub(r"[-_.]+", "-", pkg).lower()
    try:
        importlib.metadata.distribution(normalized)
    except importlib.metadata.PackageNotFoundError:
        missing.append(pkg)

for m in missing:
    print(m)
PYEOF
)

if [[ -n "$MISSING" ]]; then
    echo ""
    echo "[WARN] Primus requirements not satisfied. Missing packages:"
    while IFS= read -r pkg; do
        echo "         - $pkg"
    done <<< "$MISSING"
    echo ""
    echo "       To install:  pip install -r ${REQ_FILE}"
    echo ""

    if [[ "${PRIMUS_STRICT_REQUIREMENTS:-0}" == "1" ]]; then
        echo "[ERROR] PRIMUS_STRICT_REQUIREMENTS=1 — aborting. Install missing packages first."
        exit 1
    fi

    if [[ "${PRIMUS_AUTO_INSTALL:-0}" == "1" ]]; then
        echo "[INFO] PRIMUS_AUTO_INSTALL=1 — installing missing packages..."
        if pip install -r "${REQ_FILE}" --quiet; then
            echo "[INFO] Successfully installed missing packages."
        else
            echo "[ERROR] Failed to install packages. Please install manually."
            exit 1
        fi
    fi
fi
