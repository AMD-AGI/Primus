###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
Primus diffusion distributed training utilities.

Modules:
  mesh        - Device mesh creation, distributed setup
  checkpoint  - DTCP sharded checkpoint save/load
  ulysses     - Ulysses Sequence Parallel primitives
"""

from .checkpoint import load_checkpoint_dtcp, save_checkpoint_dtcp
from .mesh import create_device_mesh, setup_distributed
from .ulysses import distributed_attention, sp_gather, sp_split, sp_unpad

__all__ = [
    # mesh
    "setup_distributed",
    "create_device_mesh",
    # checkpoint
    "save_checkpoint_dtcp",
    "load_checkpoint_dtcp",
    # ulysses
    "distributed_attention",
    "sp_split",
    "sp_gather",
    "sp_unpad",
]
