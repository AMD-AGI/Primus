#!/bin/bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
# Package-level check of an AINIC-rebuilt image. On a node with AINIC
# hardware this also runs ibv_devices; elsewhere that list will be empty.
# Usage: test.sh <docker image>
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Syntax: $0 <docker image>" >&2
    exit 1
fi

IMAGE=$1
MOUNTS=()
if [ -d /dev/infiniband ]; then
    MOUNTS+=(-v /dev/infiniband:/dev/infiniband)
fi

docker run --rm --privileged --network host --cap-add=IPC_LOCK \
  "${MOUNTS[@]}" "$IMAGE" bash -c '
    dpkg-query -W -f="libionic1 \${Version}\n" libionic1
    readlink -f /usr/lib/x86_64-linux-gnu/libionic.so.1
    readlink -f /usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so
    dpkg -C && echo "dpkg consistent"
    ibv_devices
  '
