#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# AMD MI350X GPU-specific optimizations
# Note: MI350X is gfx950, same family as MI355X. Common settings are in
# base_env.sh. This file only contains MI350X-specific overrides.
#

LOG_INFO_RANK0 "Loading MI350X-specific optimizations..."

# ----------------- MI350X RCCL optimizations -----------------
# Disable RCCL warp-speed auto-tuning on gfx950 (MI350X). WarpSpeed is
# default-on in gfx950 RCCL builds and can produce NaN losses.
# Same override as MI355X.sh; primus-env.sh loads this file when
# PRIMUS_GPU_MODEL=MI350X (it does not fall back to MI355X.sh).
export RCCL_WARP_SPEED_AUTO=${RCCL_WARP_SPEED_AUTO:-0}

# log_exported_vars "MI350X-specific optimizations" \
#     RCCL_WARP_SPEED_AUTO
