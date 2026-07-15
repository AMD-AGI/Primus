#!/bin/bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Global hook: enable AITER.
#
# Trigger:
#   export AITER_LOG_LEVEL=ERROR
#
# This hook emits env.* lines which will be exported by execute_hooks.sh.
#

set -euo pipefail

# Set AITER log level to ERROR to suppress the verbose logs.
echo "env.AITER_LOG_LEVEL=ERROR"
