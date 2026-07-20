###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Triton kernels for the Primus Megatron extensions package.

Currently contains:

* :mod:`stack_grouped_weight` — plan-6 P34's fused
  ``torch.stack + transpose(1, 2) + contiguous`` for the per-expert weight
  tensors of :class:`PrimusTurboGroupedMLP`.
"""
