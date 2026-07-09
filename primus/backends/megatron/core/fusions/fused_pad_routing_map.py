###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Primus override for the fused routing-map padding used by MoE quantization.

Rewrites Megatron's ``fused_pad_routing_map`` as a self-contained Triton kernel
that operates directly on the native ``[num_tokens, num_experts]`` layout, so no
``transpose`` / ``contiguous`` / intermediate copy is needed. It is intentionally
*not* wrapped with ``@jit_fuser`` (``torch.compile``): the kernel is already the
fused region, and wrapping user-written Triton kernels with ``torch.compile``
triggers a functionalization failure on some torch/triton combos.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import torch
from packaging import version

from megatron.core.utils import null_decorator

try:
    import triton
    import triton.language as tl

    if version.parse(triton.__version__) < version.parse("3.4.0") and not torch.cuda.is_available():
        HAVE_TRITON = False
    else:
        HAVE_TRITON = tl.constexpr(version.parse(triton.__version__) >= version.parse("2.0.0"))
except ImportError:
    HAVE_TRITON = False

if not HAVE_TRITON:
    triton = MagicMock()
    triton.jit = null_decorator
    triton.autotune = null_decorator
    triton.heuristics = null_decorator
    tl = MagicMock()


@triton.jit
def _pad_routing_map_kernel(
    routing_map_ptr,  # *Pointer* to [num_tokens, num_experts] row-major routing map
    output_ptr,  # *Pointer* to [num_tokens, num_experts] row-major output
    num_tokens,
    num_experts,
    pad_multiple: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program instance handles one expert, i.e. one *column* of the
    # [num_tokens, num_experts] routing map. This lets us operate directly on the
    # native (row-major) layout, without transposing/copying to [num_experts, num_tokens].
    expert_idx = tl.program_id(axis=0)

    # Token indices for this block
    token_indices = tl.arange(0, BLOCK_SIZE)
    token_mask = token_indices < num_tokens

    # Column-strided access: element (t, expert_idx) lives at t * num_experts + expert_idx.
    offsets = token_indices * num_experts + expert_idx

    # Load this expert's column; out-of-bounds tokens read as 0 and are masked on store.
    row = tl.load(routing_map_ptr + offsets, mask=token_mask, other=0).to(tl.int32)

    # 1. Number of tokens currently routed to this expert.
    num_ones = tl.sum(row, axis=0)

    # 2. How many zeros must be flipped to 1 to reach the next multiple of pad_multiple.
    remainder = num_ones % pad_multiple
    num_to_pad = tl.where(remainder != 0, pad_multiple - remainder, 0)

    # 3. 1-based cumulative rank of each zero within the column.
    is_zero = row == 0
    zero_ranks = tl.cumsum(is_zero.to(tl.int32), axis=0)

    # 4. Flip only the first `num_to_pad` zeros to 1.
    mask_to_flip = (zero_ranks <= num_to_pad) & is_zero
    output_row = tl.where(mask_to_flip, 1, row)

    # 5. Store back in the same native layout, masking out-of-bounds tokens.
    tl.store(output_ptr + offsets, output_row, mask=token_mask)


def fused_pad_routing_map(routing_map: torch.Tensor, pad_multiple: int) -> torch.Tensor:
    """Fused version of pad_routing_map.
    Args:
        routing_map (torch.Tensor): A boolean or integer tensor of shape [num_tokens,
            num_experts] indicating which tokens are routed to which experts.
        pad_multiple (int): The multiple to pad each expert's token count to.

    Returns:
        torch.Tensor: The padded routing map of shape [num_tokens, num_experts].
    """
    num_tokens, num_experts = routing_map.shape
    if num_tokens == 0:
        return routing_map

    # Operate directly on the native [num_tokens, num_experts] layout: the kernel reads
    # each expert's column with a stride of num_experts, so no transpose/copy is needed.
    routing_map = routing_map.contiguous()
    output_map = torch.empty_like(routing_map, dtype=torch.int32)

    # One program instance per expert (column).
    grid = (num_experts,)
    BLOCK_SIZE = triton.next_power_of_2(num_tokens)

    _pad_routing_map_kernel[grid](
        routing_map, output_map, num_tokens, num_experts, pad_multiple, BLOCK_SIZE=BLOCK_SIZE
    )

    return output_map  # [num_tokens, num_experts]
