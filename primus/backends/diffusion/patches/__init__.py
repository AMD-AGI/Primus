###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Diffusion-backend patches.

The trainer imports the modules it needs. This package does not auto-import
them, so a normal diffusion job does not load Flux MLPerf logging.
"""
