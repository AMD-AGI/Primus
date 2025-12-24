#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Rebuild bnxt from a tar package, intended to be run via:
#   runner/helpers/execute_patches.sh runner/helpers/rebuild_nbxt.sh
#
# Control via environment variables:
#   PATH_TO_BNXT_TAR_PACKAGE=/path  # path to libbnxt_re-*.tar.gz
#
# Exit codes (for execute_patches.sh):
#   0  - success (patch applied)
#   2  - skipped (no work to do; not an error)
#   >2 - failure (stop patch pipeline)
###############################################################################

set -euo pipefail

PATH_TO_BNXT_TAR_PACKAGE="${PATH_TO_BNXT_TAR_PACKAGE:-}"

if [[ -z "$PATH_TO_BNXT_TAR_PACKAGE" || ! -f "$PATH_TO_BNXT_TAR_PACKAGE" ]]; then
    LOG_INFO "Skip bnxt rebuild. PATH_TO_BNXT_TAR_PACKAGE=$PATH_TO_BNXT_TAR_PACKAGE"
    exit 2
fi

LOG_INFO "Rebuilding bnxt from $PATH_TO_BNXT_TAR_PACKAGE ..."

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

cd "${PRIMUS_PATH:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
LOG_INFO "Rebuilding libbnxt done."

exit 0
