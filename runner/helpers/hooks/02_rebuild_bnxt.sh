#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# System hook: optionally rebuild bnxt from a tar package.
#
# Equivalent behavior to the legacy logic in examples/run_pretrain.sh.
#
# Trigger:
#   export REBUILD_BNXT=1
#
# Required:
#   export PATH_TO_BNXT_TAR_PACKAGE=/path/to/libbnxt_re-*.tar.gz
#
# Implementation:
#   Inline rebuild steps (no separate helper script).
###############################################################################

set -euo pipefail

REBUILD_BNXT="${REBUILD_BNXT:-0}"
PATH_TO_BNXT_TAR_PACKAGE="${PATH_TO_BNXT_TAR_PACKAGE:-}"

if [[ "${REBUILD_BNXT}" != "1" ]]; then
    exit 0
fi

if [[ -z "${PRIMUS_PATH:-}" ]]; then
    # Best-effort fallback: infer PRIMUS_PATH from this file location
    PRIMUS_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    export PRIMUS_PATH
fi

if [[ -z "${PATH_TO_BNXT_TAR_PACKAGE}" || ! -f "${PATH_TO_BNXT_TAR_PACKAGE}" ]]; then
    LOG_INFO_RANK0 "[hook system] Skip bnxt rebuild. REBUILD_BNXT=${REBUILD_BNXT}, PATH_TO_BNXT_TAR_PACKAGE=${PATH_TO_BNXT_TAR_PACKAGE}"
    exit 0
fi

LOG_INFO_RANK0 "[hook system] REBUILD_BNXT=1 → rebuilding bnxt from ${PATH_TO_BNXT_TAR_PACKAGE}"

# Inline implementation (previously runner/helpers/rebuild_bnxt.sh)
tar xzf "${PATH_TO_BNXT_TAR_PACKAGE}" -C /tmp/
mv /tmp/libbnxt_re-* /tmp/libbnxt
mv /usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so /usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so.inbox

cd /tmp/libbnxt/
sh ./autogen.sh
./configure
make clean all install

echo '/usr/local/lib' > /etc/ld.so.conf.d/libbnxt_re.conf
ldconfig
cp -f /tmp/libbnxt/bnxt_re.driver /etc/libibverbs.d/

cd "${PRIMUS_PATH}"
LOG_INFO_RANK0 "[hook system] Rebuilding libbnxt done."
