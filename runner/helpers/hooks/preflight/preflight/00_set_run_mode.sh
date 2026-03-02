#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Preflight run-mode policy:
# - Default to single-process in direct mode to avoid torchrun rendezvous hang
#   when cluster networking is unhealthy.
# - Allow opt-in torchrun via PRIMUS_PREFLIGHT_ALLOW_TORCHRUN=1.
#
# Hook protocol:
#   print "env.RUN_MODE=<value>" to export environment variables in launcher.
#

set -euo pipefail

allow_torchrun="${PRIMUS_PREFLIGHT_ALLOW_TORCHRUN:-0}"
if [[ "$allow_torchrun" == "1" ]]; then
    echo "env.RUN_MODE=torchrun"
else
    echo "env.RUN_MODE=single"
fi
