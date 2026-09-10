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
BUNDLE_VER=$(echo "$BUNDLE_NAME" | sed -E \
    's/^ainic_bundle_//; s/\.tar(\.(gz|xz|bz2|zst))?$//')

# Send only the requested bundle to Docker. Using the bundle's parent directory
# as the context can be both very large and unreadable because of unrelated
# files. Prefer a hard link to avoid copying a large bundle when /tmp shares the
# same filesystem, and fall back to a regular copy otherwise.
CONTEXT=$(mktemp -d)
trap 'rm -rf "$CONTEXT"' EXIT
ln "$BUNDLE" "$CONTEXT/$BUNDLE_NAME" 2>/dev/null || \
    cp --reflink=auto "$BUNDLE" "$CONTEXT/$BUNDLE_NAME"

set -x
docker build --network host \
  -f "$SCRIPT_DIR/Dockerfile" \
  --build-arg BASE_IMAGE="$BASE" \
  --build-arg AINIC_BUNDLE="$BUNDLE_NAME" \
  -t "${BASE##*/}-ainic-${BUNDLE_VER}" \
  "$CONTEXT"
