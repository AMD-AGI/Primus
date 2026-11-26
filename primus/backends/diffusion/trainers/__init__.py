###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""Trainer registrations for the Primus diffusion backend."""

from .benchmark import build_amdb_benchmark_trainer
from .fsdp2 import build_fsdp2_trainer

__all__ = ["build_amdb_benchmark_trainer", "build_fsdp2_trainer"]
