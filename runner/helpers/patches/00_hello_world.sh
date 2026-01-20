#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Minimal example patch script for Primus runner.
#
# This patch does not modify anything; it just prints a message so you can verify
# patch execution order and log capture.
###############################################################################

set -euo pipefail

echo "[primus patch] hello world"
exit 0
