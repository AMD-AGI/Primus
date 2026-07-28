###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
wan: Components + TaskPipeline design for Wan PyTorch/FSDP training.

Goal:
- Keep modeling close to official Wan2.2 (pure torch modules).
- Keep trainer generic: trainer calls model(batch, scheduler) -> {"loss": ...}.
- Decouple training/inference workflow (pipeline) from modeling (modules).
"""

from .adapter import WanForTraining

__all__ = [
    "WanForTraining",
]
