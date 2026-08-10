###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Eager reference for the Kimi K3 attention-residual mixer.

A verbatim extraction of what ``AttentionResidualMixer.forward`` used to do
inline, so that it can serve as the permanent numerical oracle for the fused
FlyDSL kernel — the same role :mod:`...kda_kernels._eager.reference` plays for
KDA. ``test_attention_residual.py`` already pins this arithmetic bit-exactly
against an independent transcription of ``_apply_attn_res``
(``modeling_kimi_linear.py:1075-1088``), and that test keeps passing unchanged,
which is what makes the extraction safe.

Three details, restated here because this is now the file a kernel is checked
against:

1. **The output mixes the un-normalised candidates.** ``v`` is the raw
   ``cat([block_residual, prefix_sum])``; the RMS normalisation feeds the
   *scores* only (``:1083`` vs ``:1087``).
2. **The scorer is rank-1**, the elementwise product of the RMSNorm gain and
   the ``[1, hidden]`` projection row (``:1084``).
3. **All of it runs in fp32** and casts back once at the end (``:1081``,
   ``:1088``).
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

__all__ = ["accum_dtype", "fused_score_weight", "eager_attn_res_mix"]


def accum_dtype(dtype: torch.dtype) -> torch.dtype:
    """fp32, unless the operand is already wider.

    The reference spells this ``.float()`` unconditionally
    (``modeling_kimi_linear.py:1081``), which is an up-cast for every dtype
    Kimi K3 actually trains in (bf16 / fp16 / fp32) and a *down*-cast in fp64.
    Promoting instead is bit-identical on all three real cases and keeps the
    module differentiable under ``torch.autograd.gradcheck``, which needs fp64
    end to end.
    """
    return torch.promote_types(dtype, torch.float32)


def fused_score_weight(
    norm_weight: Tensor, proj_weight: Tensor, dtype: Optional[torch.dtype] = None
) -> Tensor:
    """The rank-1 scoring vector, ``[hidden]`` (``modeling_kimi_linear.py:1084``).

    Kept as a separate function because both backends need it and because the
    FlyDSL backend forms it with plain autograd-visible torch ops *outside* the
    kernel — one auditable copy of the factorisation, and the kernel then takes
    a single ``[hidden]`` vector. Same division of labour as KDA's
    ``use_qk_l2norm_in_kernel``.
    """
    if dtype is None:
        dtype = accum_dtype(proj_weight.dtype)
    return norm_weight.to(dtype) * proj_weight.squeeze(0).to(dtype)


def eager_attn_res_mix(
    prefix_sum: Tensor,
    block_residual: Tensor,
    norm_weight: Tensor,
    proj_weight: Tensor,
    eps: float,
) -> Tensor:
    """``_apply_attn_res``: softmax-mix the stream with the block checkpoints.

    Args:
        prefix_sum: ``[*, hidden]`` — the running residual stream.
        block_residual: ``[*, num_blocks, hidden]`` — the cross-layer
            checkpoints. ``num_blocks == 0`` is legal and degenerates to a
            softmax over a single candidate, i.e. ``prefix_sum`` itself.
        norm_weight: ``[hidden]`` RMSNorm gain.
        proj_weight: ``[1, hidden]`` projection row.
        eps: RMSNorm epsilon.

    Returns:
        ``[*, hidden]``, in ``prefix_sum``'s dtype.
    """
    # [*, num_blocks + 1, hidden]: the checkpoints, then the stream.
    v = torch.cat((block_residual, prefix_sum.unsqueeze(-2)), dim=-2)
    compute_dtype = accum_dtype(v.dtype)
    v_float = v.to(compute_dtype)

    # RMS-normalise each candidate. This feeds the SCORES ONLY.
    variance = v_float.pow(2).mean(-1, keepdim=True)
    k = v_float * torch.rsqrt(variance + eps)

    scores = (k * fused_score_weight(norm_weight, proj_weight, compute_dtype)).sum(-1)
    probs = scores.softmax(-1).unsqueeze(-2)  # [*, 1, num_blocks + 1]

    # Convex combination of the UN-normalised candidates.
    mixed = torch.matmul(probs, v_float).squeeze(-2)  # [*, hidden]
    return mixed.to(prefix_sum.dtype)
