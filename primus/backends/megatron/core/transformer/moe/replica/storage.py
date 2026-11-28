###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Replica slot storage: ``B`` slots of one expert weight shape, their weights and their gradients."""

import torch


class SlotStorage:
    """``w`` ``[B, *shape]`` in the weights' dtype and ``main_grad`` ``[B, *shape]`` in the owners' ``main_grad``
    dtype (one fused grouped GEMM accumulates into both), allocated by the transport. Plain tensors: not parameters
    or buffers, so not checkpointed and without optimizer state."""

    def __init__(self, transport, num_slots: int, shape, dtype: torch.dtype, grad_dtype: torch.dtype, device):
        self.w = transport.alloc((num_slots, *shape), dtype, device)
        self.main_grad = transport.alloc((num_slots, *shape), grad_dtype, device)

    def claim(self, user) -> bool:
        """Make the slots hold ``user``'s slot table; False if another user's weights are in them now, so the
        caller pushes all of its slots again."""
        raise NotImplementedError


class LayerwiseStorage(SlotStorage):
    """One set of slots per expert linear: it only ever holds its own user's weights."""

    def claim(self, user) -> bool:
        return True
