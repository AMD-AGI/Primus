#!/bin/bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
# Rebuild BASE against a pre-downloaded AINIC bundle tarball.
# Usage: build.sh <base docker image> <ainic bundle file>
set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Syntax: $0 <base docker image> <ainic bundle file>" >&2
    exit 1
fi

BASE=$1
BUNDLE=$2
if [ ! -f "$BUNDLE" ]; then
    echo "error: bundle file not found: $BUNDLE" >&2
    exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
BUNDLE_NAME=$(basename "$BUNDLE")
BUNDLE_VER=$(echo "$BUNDLE_NAME" | sed -E 's/ainic_bundle_(.*)\.tar\.gz/\1/')
CONTEXT=$(cd "$(dirname "$BUNDLE")" && pwd)

set -x
docker build --network host \
  -f "$SCRIPT_DIR/Dockerfile" \
  --build-arg BASE_IMAGE="$BASE" \
  --build-arg AINIC_BUNDLE="$BUNDLE_NAME" \
  -t "${BASE##*/}-ainic-${BUNDLE_VER}" \
  "$CONTEXT"
