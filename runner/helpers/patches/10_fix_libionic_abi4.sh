#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# primus-cli --patch script: swap the ionic libibverbs provider (.so) for an
# ABI-4-capable build at container launch (before torchrun starts).
#
# Why: some AINIC container images ship a libionic provider built against stock
# rdma-core that only advertises ionic uverbs ABI 1, while the host kernel ionic
# driver exposes ABI 4 -- so libibverbs rejects every ionic_* device and RDMA
# falls back to TCP. Copying in an ABI-4 provider .so fixes device enumeration
# without rebuilding the image.
#
# Controlled by PRIMUS_LIBIONIC_SRC_ABI4_SO (set it in your launch script, e.g.
# run_flash.sh). If unset/empty the patch is skipped. The PRIMUS_ prefix makes it
# auto-forward into the container (see primus-cli-container.sh env passthrough).
###############################################################################
set -euo pipefail

if [[ -z "${PRIMUS_LIBIONIC_SRC_ABI4_SO:-}" ]]; then
    echo "[fix_libionic_abi4] PRIMUS_LIBIONIC_SRC_ABI4_SO not set -- skipping"
    exit 2  # 2 = skip (not an error), per runner/helpers/execute_patches.sh
fi

SRC="$PRIMUS_LIBIONIC_SRC_ABI4_SO"
if [[ ! -f "$SRC" ]]; then
    echo "[fix_libionic_abi4] source .so not found: $SRC" >&2
    exit 1
fi

PROVIDER_LINK=/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so
if [[ ! -e "$PROVIDER_LINK" ]]; then
    echo "[fix_libionic_abi4] ionic provider not present ($PROVIDER_LINK); nothing to patch" >&2
    exit 2
fi
DST="$(readlink -f "$PROVIDER_LINK")"

if cmp -s "$SRC" "$DST"; then
    echo "[fix_libionic_abi4] provider already matches source -- skipping"
    exit 2
fi

cp --remove-destination "$SRC" "$DST"
echo "[fix_libionic_abi4] swapped ionic provider: $DST <- $SRC"
