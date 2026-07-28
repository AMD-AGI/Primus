###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""Trainer registrations for the Primus diffusion backend."""

from .fsdp2 import build_fsdp2_trainer

__all__ = ["build_fsdp2_trainer"]
