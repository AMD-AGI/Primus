#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# System hook: make the RDMA userspace provider (libibverbs driver) match the
# NIC kernel driver on the host, so RCCL uses the NICs instead of TCP sockets.
#
# Covers Broadcom (libbnxt_re) and AMD Pensando AINIC (libionic). A no-op when
# the provider already works; runner/helpers/nic_userspace_driver.py documents
# how a matching provider is found, installed and verified.
#
# Controls:
#   PRIMUS_NIC_USERSPACE_DRIVER=auto|force|off  (default: auto)
#   PRIMUS_NIC_DRIVER_SEARCH_PATH=dir:dir:...    vendor bundles to search
#       (default: /opt/broadcom:/opt/bnxt-bundles:/opt/amd/ainic)
#   PRIMUS_NIC_DRIVER_STRICT=1                   abort the launch if RDMA stays unusable
#   PRIMUS_AINIC_REPO_URL=<url>|none             AINIC package repository
#   PRIMUS_AINIC_BUNDLE_VERSION=<bundle>         AINIC bundle to install when fw_ver does not name it
#
# Legacy (still honored):
#   REBUILD_BNXT=1                               force a reinstall for Broadcom NICs
#   PATH_TO_BNXT_TAR_PACKAGE=/path/to/libbnxt_re-<version>.tar.gz
#
# Inspect without changing anything:
#   python3 runner/helpers/nic_userspace_driver.py status
# Troubleshooting: docs/04-technical-guides/multi-node-networking.md, section 5.
###############################################################################

set -euo pipefail

case "${PRIMUS_NIC_USERSPACE_DRIVER:-auto}" in
    off|OFF|0|false|no) exit 0 ;;
esac

# No RDMA devices: nothing to match, and no reason to start Python.
compgen -G "/sys/class/infiniband/*" >/dev/null || exit 0

strict=0
case "${PRIMUS_NIC_DRIVER_STRICT:-0}" in
    1|true|TRUE|yes|on) strict=1 ;;
esac

if ! python3 -c 'import sys; sys.exit(sys.version_info < (3, 7))' 2>/dev/null; then
    echo "[WARN] [nic-driver] python3 >= 3.7 not found; skipping the NIC userspace driver check" >&2
    exit "$strict"
fi

# The check is best effort: an unexpected failure must not stop a launch unless strict mode asks for it.
rc=0
python3 "$(dirname "${BASH_SOURCE[0]}")/../nic_userspace_driver.py" setup || rc=$?
if [[ $rc -ne 0 && $strict -eq 0 ]]; then
    echo "[WARN] [nic-driver] the NIC userspace driver check exited with $rc; continuing" >&2
    exit 0
fi
exit "$rc"
